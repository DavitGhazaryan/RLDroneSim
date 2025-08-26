import gymnasium as gym
from gymnasium.utils import seeding
from gymnasium import spaces

import numpy as np
import logging
import sys
import asyncio
import math

sys.path.insert(0, "/home/student/Dev/pid_rl")

from rl_training.utils.gazebo_interface import GazeboInterface
from rl_training.utils.ardupilot_sitl import ArduPilotSITL
from rl_training.utils.utils import euler_to_quaternion
logger = logging.getLogger("Env")


class ArdupilotEnv(gym.Env):
    """
    Initializes Gazebo and Ardupilot SITL, and provides a Gymnasium-compatible interface.
    Environment is intended to enable training if an RL agent that will find the optimal PID gains to put on an agent.  
    """
    
    def __init__(self, config):
        super().__init__()
        self.np_random, _ = seeding.np_random(None)  # init
        self.config = config.get('environment_config', {})
        self.max_episode_steps = self.config.get('max_episode_steps')
        self.mode = self.config.get('mode')
        self.observable_gains = self.config['observable_gains'].split('+')
        self.observable_states = self.config['observable_states'].split('+')
        self.action_gains = self.config['action_gains'].split('+')
        self.reward_config = self.config["reward_config"]
        self.action_dt = self.config.get('action_dt')
        self.takeoff_altitude = self.config['takeoff_altitude']

        self.verbose = self.config.get("verbose")

        if self.verbose:
            logger.setLevel(logging.INFO)
        else:
            logger.setLevel(logging.ERROR)

        self.gazebo = GazeboInterface(config['gazebo_config'], self.verbose)
        self.sitl = ArduPilotSITL(config['ardupilot_config'], self.verbose)
        self.loop = asyncio.get_event_loop()

        # Episode tracking
        self.initialized = False
        self.episode_step = 0

        self.ep_initial_pose = None       # at the episode start {lat_deg:, lon_deg:, rel_alt_m:}
        self.ep_initial_attitude = None   # {pitch_deg:, roll_deg:, yaw_deg:}
        self.ep_initial_gains = {}      # {gain_name: value}
        
        self._async_mission_function = None
        self.goal_orientation = None   # {pitch_deg:, roll_deg:, yaw_deg:}
        self.goal_pose = None          # {latitude_deg:, longitude_deg:, relative_altitude_m:}
        self.eps_stable_time = 0
        self.max_stable_time = 0
        self.accumulated_huber_error = 0.0  # Track Huber errors for timeout reward
        
        # Initialize spaces
        self.observation_space = self._define_observation_space()
        self.action_space = self._define_action_space()
        
        # Log space information for debugging
        logger.info(f"🔧 Environment spaces initialized:")
        logger.info(f"   Observation space: {self.observation_space}")
        logger.info(f"   Action space: {self.action_space}")
        logger.info(f"   Observable gains: {self.observable_gains}")
        logger.info(f"   Observable states: {self.observable_states}")
        logger.info(f"   Action gains: {self.action_gains}")
        
        # Print mapping information
        obs_mapping = self.get_observation_key_mapping()
        action_mapping = self.get_action_key_mapping()
        logger.info(f"   Observation mapping: {obs_mapping}")
        logger.info(f"   Action mapping: {action_mapping}")
        
    def _define_observation_space(self):
        """
        Observations are flattened into a single Box space for Stable Baselines compatibility.
        Order: [observable_gains, observable_states]
        """
        # Calculate total dimension
        total_dim = len(self.observable_gains) + len(self.observable_states)
        
        # Create bounds arrays
        lows = []
        highs = []
        
        # Add bounds for gains (0 to 100)
        for _ in self.observable_gains:
            lows.append(0.0)
            highs.append(100.0)
        
        # Add bounds for states (-1000 to 1000)
        for _ in self.observable_states:
            lows.append(-1000.0)
            highs.append(1000.0)
        
        return spaces.Box(
            low=np.array(lows, dtype=np.float32),
            high=np.array(highs, dtype=np.float32),
            dtype=np.float32
        )
    
    def _define_action_space(self):
        """
        Action space is flattened into a single Box space for Stable Baselines compatibility.
        Actions represent PID gain adjustments.
        """
        # Calculate total dimension
        total_dim = len(self.action_gains)
        
        # Create bounds arrays (all actions are -0.1 to 0.1)
        lows = np.array([-5.0] * total_dim, dtype=np.float32)
        highs = np.array([5.0] * total_dim, dtype=np.float32)
        
        return spaces.Box(
            low=lows,
            high=highs,
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)             # sets self.np_random
        if hasattr(self.action_space, "seed"):
            self.action_space.seed(seed)
        if hasattr(self.observation_space, "seed"):
            self.observation_space.seed(seed)
        self.episode_step = 0

        if not self.initialized:
            logger.info("🌎 Launching Gazebo simulation...")
            self.gazebo.start_simulation()
            self.gazebo._wait_for_startup()
            self.gazebo.resume_simulation()
            logger.debug("✅ Gazebo initialized")
            logger.debug("🚁 Starting ArduPilot SITL...")
            self.sitl.start_sitl()
            info = self.sitl.get_process_info()
            logger.debug(f"✅ SITL running (PID {info['pid']})")
            self.initialized = True
            self.loop.run_until_complete(self._async_arm())
            ## Resetting Ends Here: Drone is Armed

            ## Setup Mission
            pose_ned = self.loop.run_until_complete(self.sitl.get_pose_NED_async())

            for gain in self.action_gains:
                self.ep_initial_gains[gain] = self.loop.run_until_complete(self.sitl.get_param_async(gain))
                

            self.ep_initial_pose = {
                'x_m': pose_ned.east_m,
                'y_m': pose_ned.north_m,
                'z_m': - pose_ned.down_m
            }
            attitude = self.loop.run_until_complete(self.sitl.get_attitude_async())
            self.ep_initial_attitude = {
                'pitch_deg': attitude.pitch_deg,
                'roll_deg': attitude.roll_deg,
                'yaw_deg': attitude.yaw_deg
            }

            self._async_mission_function = self._setup_mission()   # same x, y, z
            self.loop.run_until_complete(self._async_mission_function())
        else:
            logger.info("Resetting the Environment")
            self.ep_initial_pose, self.ep_initial_attitude, self.ep_initial_gains = self.get_random_initial_state()
            self.eps_stable_time = 0
            self.max_stable_time = 0
            self.accumulated_huber_error = 0.0

            logger.info(f"Setting gains to {self.ep_initial_gains}")

            for gain in self.action_gains:
                self.loop.run_until_complete(self.sitl.set_param_async(gain, self.ep_initial_gains[gain]))

            logger.info(f"Setting drone to {self.ep_initial_pose}")
            
            self.gazebo.pause_simulation()
            self.gazebo.transport_position(self.sitl.name, [self.ep_initial_pose["x_m"], self.ep_initial_pose["y_m"], self.ep_initial_pose["z_m"]], euler_to_quaternion(None))
            self.gazebo.resume_simulation()

            self.loop.run_until_complete(self.sitl.transport_and_reset_async([self.ep_initial_pose["x_m"], self.ep_initial_pose["y_m"], self.ep_initial_pose["z_m"]], euler_to_quaternion(None)))
        
        observation, info = self.loop.run_until_complete(self._async_get_observation())
        return observation, info  # observation, info

    async def _async_get_observation(self):
        """
        Get the observations of the environment as a flattened array.
        This function is called when the environment is reset.
        Returns flattened observations for Stable Baselines compatibility.
        """
        drone = await self.sitl._get_mavsdk_connection()

        # Initialize flattened observation array
        total_dim = len(self.observable_gains) + len(self.observable_states)
        observation = np.zeros(total_dim, dtype=np.float32)
        
        # Fill gains first
        for i, observable_gain in enumerate(self.observable_gains):
            gain_value = await drone.param.get_param_float(observable_gain)
            observation[i] = gain_value
        
        # Fill states
        pose_ned = (await anext(drone.telemetry.position_velocity_ned())).position
        
        for i, observable_state in enumerate(self.observable_states):
            idx = len(self.observable_gains) + i
            if observable_state == 'altitude_m':
                state_value = -pose_ned.down_m  # Convert down_m (negative) to relative altitude (positive)
                
            elif observable_state == 'x_m':
                state_value = pose_ned.east_m
            elif observable_state == 'y_m':
                state_value = pose_ned.north_m
            else:
                raise NotImplemented("Observation not available")
                # For other states, try to get from NED position
            observation[idx] = state_value
        
        info = {}
        return observation, info 
    
    def _setup_mission(self):
        match self.mode:
            case 'altitude':
                self.goal_pose = {
                    'x_m': self.ep_initial_pose['x_m'],
                    'y_m': self.ep_initial_pose['y_m'],
                    'z_m': self.takeoff_altitude + 0.19
                }
                self.goal_orientation = self.ep_initial_attitude.copy()

                return self._async_takeoff
            
            case 'position' | 'attitude' | 'stabilize' | 'althold':
                raise NotImplementedError("Position, attitude, stabilize, and althold modes are not implemented yet")
            case _:
                raise ValueError(f"Invalid mode: {self.mode}")

    def get_random_initial_state(self):
        initial_gains = {}
        for gain in self.action_gains:
            # initial_gains[gain] = self.np_random.uniform(0.8, 5.2)
            initial_gains[gain] = max(self.ep_initial_gains[gain] + self.np_random.uniform(-3.0, 3.0), 0)
        return {
            'x_m': self.goal_pose['x_m'],
            'y_m': self.goal_pose['y_m'],    
            'z_m': max(self.goal_pose['z_m'], 0.3) + self.np_random.uniform(-2.0, 2.0)
        }, self.ep_initial_attitude, initial_gains
    
    def step(self, action):
        self.episode_step += 1
        obs, reward, done, truncated, info = self.loop.run_until_complete(self._async_step(action))
        # self.old_observation = obs
        return obs, reward, done, truncated, info

    async def _async_step(self, action):
        """
        Handle flattened actions for Stable Baselines compatibility.
        Actions are changes to PID parameters that need to be applied.
        """
        drone = await self.sitl._get_mavsdk_connection()
        # Validate action dimensions
        if len(action) != len(self.action_gains):
            raise ValueError(f"Expected action of length {len(self.action_gains)}, got {len(action)}")
        
        # Get current gains
        new_gains = {}
        for variable in self.action_gains:
            new_gains[variable] = await drone.param.get_param_float(variable)
        
        # Apply the flattened action to the gains
        for i, var in enumerate(self.action_gains):
            new_gains[var] += action[i]
            new_gains[var] = max(new_gains[var], 0)
            await drone.param.set_param_float(var, new_gains[var])
        
        print(f"SENDING {new_gains}")

        await self._gazebo_sleep(self.action_dt)   # no need to normalize the sleep time with speedup

        # get the new observation
        observation, info = await self._async_get_observation()
        print(observation)
        pose_vel = await anext(drone.telemetry.position_velocity_ned())
        pose_ned = pose_vel.position
        attitude = await anext(drone.telemetry.attitude_euler())
        

        def _check_terminated(pose_vel, attitude):
            pose = pose_vel.position
            vel = pose_vel.velocity
            # Assume pose : [north_m:, east_m:, down_m:] object
            #             attitude: [pitch_deg:, roll_deg:, yaw_deg:]
            
            # 1. Crash: the difference between current and target attitude is more that 30 degree
            if abs(attitude.pitch_deg - self.goal_orientation['pitch_deg']) > 15 or abs(attitude.roll_deg - self.goal_orientation['roll_deg']) > 15:
                return True, "crash: excessive pitch/roll"

            # Velocity checks
            if abs(vel.north_m_s) > 3 or abs(vel.east_m_s) > 3 or abs(vel.down_m_s) > 3:
                return True, "vel exceeds limits"
            

            # 2. Flip (pitch or roll > 90 deg)
            if abs(attitude.pitch_deg) > 90 or abs(attitude.roll_deg) > 90:
                return True, "crash: flip"

            # 3. 2x farther from goal than originally
            dist_init = np.linalg.norm(np.array([self.ep_initial_pose["x_m"], self.ep_initial_pose["y_m"], self.ep_initial_pose["z_m"]]) 
                                       - np.array([self.goal_pose["x_m"], self.goal_pose["y_m"], self.goal_pose["z_m"]]))
            dist_now = np.linalg.norm(np.array([pose.north_m, pose.east_m, -pose.down_m]) 
                                      - np.array([self.goal_pose["x_m"], self.goal_pose["y_m"], self.goal_pose["z_m"]]))
            
            if dist_now > 4.0 * dist_init and dist_now > 0.01:
                return True, "too far from goal"
            
            # 4. goal is reached - check both position and altitude
            pos_error = np.linalg.norm(np.array([pose.north_m, pose.east_m]) - np.array([self.goal_pose["x_m"], self.goal_pose["y_m"]]))
            alt_error = abs(- pose.down_m - self.goal_pose["z_m"])
            
            # Check if in vicinity using helper method
            in_vicinity = self._check_vicinity_status(pos_error, alt_error)
            
            # Update stable time tracking (consolidated logic)
            if in_vicinity:
                self.eps_stable_time += 1
                if self.eps_stable_time >= self.max_episode_steps/2:
                    return True, f"stable for {self.eps_stable_time} steps"
            else:
                if self.eps_stable_time > self.max_stable_time:
                    self.max_stable_time = self.eps_stable_time
                self.eps_stable_time = 0
            return False, None

        terminated, reason = _check_terminated(pose_vel, attitude)

        if terminated:
            truncated = False
            logger.info(f"Terminating the episode {reason}")
        else:
            terminated = False
            truncated = self.episode_step >= self.max_episode_steps
            if truncated:
                logger.info(f"Truncating the Episode")
        reward = self._compute_reward(pose_ned, attitude)
        
        # Create proper info dictionary
        info = {
            'reason': reason,
            'episode_step': self.episode_step,
            'stable_time': self.eps_stable_time,
            'max_stable_time': self.max_stable_time,
            'pos_error': float(np.linalg.norm(np.array([pose_ned.north_m, pose_ned.east_m]) - np.array([self.goal_pose["x_m"], self.goal_pose["y_m"]]))),
            'alt_error': float(abs(-pose_ned.down_m - self.goal_pose["z_m"])),
            'accumulated_huber_error': float(self.accumulated_huber_error),
        }

        for i, var in enumerate(self.action_gains):
            info[var] = new_gains[var]

        return observation, reward, terminated, truncated, info

    def get_observation_key_mapping(self):
        """Get mapping from observation keys to array indices."""
        mapping = {}
        all_keys = self.observable_gains + self.observable_states
        for i, key in enumerate(all_keys):
            mapping[key] = i
        return mapping
    
    def get_action_key_mapping(self):
        """Get mapping from action keys to array indices."""
        mapping = {}
        for i, key in enumerate(self.action_gains):
            mapping[key] = i
        return mapping
    
    def get_observation_description(self):
        """Get description of what each observation index represents."""
        description = {}
        all_keys = self.observable_gains + self.observable_states
        for i, key in enumerate(all_keys):
            if i < len(self.observable_gains):
                description[f"obs_{i}"] = f"Gain: {key}"
            else:
                description[f"obs_{i}"] = f"State: {key}"
        return description
    
    def get_action_description(self):
        """Get description of what each action index represents."""
        description = {}
        for i, key in enumerate(self.action_gains):
            description[f"action_{i}"] = f"Gain adjustment: {key}"
        return description

    async def _async_arm(self):
        logger.info("Arming...")
        drone = await self.sitl._get_mavsdk_connection()

        logger.info("Waiting for vehicle to become armable...")
        async for health in drone.telemetry.health():
            if health.is_armable and health.is_global_position_ok:
                logger.info("Vehicle is armable and has GPS fix!")
                break

        await drone.action.arm()
        await asyncio.sleep(1/self.sitl.speedup)

    async def _async_takeoff(self):

        drone = await self.sitl._get_mavsdk_connection()
        await drone.action.set_takeoff_altitude(self.takeoff_altitude)
        await drone.action.takeoff()
        await asyncio.sleep(15/self.sitl.speedup)

    async def _gazebo_sleep(self, duration):
        """
        Sleep for the given duration (in seconds) using Gazebo simulation time.
        """
        start_time = self.gazebo.get_sim_time()
        while True:
            await asyncio.sleep(0.0001)
            current_time = self.gazebo.get_sim_time()
            if current_time - start_time >= duration:
                break

    def _check_vicinity_status(self, pos_error, alt_error):
        """
        Check if the drone is in vicinity of the goal.
        Uses hysteresis to prevent flickering between vicinity states.
        """
        eps_in = 0.03  # Inner vicinity threshold
        eps_out = 0.05  # Outer vicinity threshold
        
        prev_in_vicinity = self.eps_stable_time > 0
        in_vicinity = (pos_error <= eps_in and alt_error <= eps_in) if prev_in_vicinity else (pos_error <= eps_out and alt_error <= eps_out)
        
        return in_vicinity
    
    def _compute_reward(self, pose, attitude=None):
        """
        Compute reward based on stable-time mechanism.
        
        Reward components:
        - Dense penalty: -Huber(e_t, delta) where e_t is the error
        - Vicinity bonus: γ_s + η*log(1 + stable_time) when in vicinity
        - Success bonus: +R_succ when episode terminates successfully
        - Timeout reward: κ*max_stable_time - ν*sum_huber_error when timing out
        - Crash penalty: Large negative reward for dangerous attitudes
        """
        
        # Calculate position error (2D distance)
        pos_error = np.linalg.norm(np.array([pose.north_m, pose.east_m]) - np.array([self.goal_pose["x_m"], self.goal_pose["y_m"]]))
        
        # Calculate altitude error
        alt_error = abs(-pose.down_m - self.goal_pose["z_m"])
        
        # Use the larger error for the main reward calculation  ????
        e_t = max(pos_error, alt_error)
        
        # Huber function for primary penalty
        def huber(e, delta):
            a = abs(e)/delta
            return 0.5*a*a if a <= 1.0 else a - 0.5
        
        # Vicinity parameters (consistent with _check_terminated)
        delta = 0.05   # Huber parameter (≈ vicinity radius)
        
        # Check vicinity status using helper method
        in_vicinity = self._check_vicinity_status(pos_error, alt_error)
        
        # Base reward: primary penalty using Huber
        r = -huber(e_t, delta)
        
        # Accumulate Huber error for timeout reward calculation
        self.accumulated_huber_error += huber(e_t, delta)
        
        # Crash penalty (if attitude is provided)
        if attitude is not None:
            # Penalize excessive pitch/roll deviations from goal
            pitch_error = abs(attitude.pitch_deg - self.goal_orientation['pitch_deg'])
            roll_error = abs(attitude.roll_deg - self.goal_orientation['roll_deg'])
            
            # Large penalty for dangerous attitudes
            if pitch_error > 30 or roll_error > 20:
                r -= 50.0  # Severe crash penalty
            elif pitch_error > 20 or roll_error > 15:
                r -= 20.0  # Moderate penalty
            elif pitch_error > 10 or roll_error > 10:
                r -= 5.0   # Light penalty
        
        # Vicinity bonus
        if in_vicinity:
            gamma_s = 0.1  # Base vicinity bonus
            eta = 0.05     # Log scaling factor
            r += gamma_s + eta * math.log1p(self.eps_stable_time)
        
        # Success termination bonus
        T_stable = self.max_episode_steps // 2  # Success threshold
        if self.eps_stable_time >= T_stable:
            R_succ = 100.0  # Success bonus
            r += R_succ
        
        # Timeout reward (when episode ends due to max steps)
        if self.episode_step >= self.max_episode_steps:
            kappa = 1    # Stable time coefficient
            nu = 0.1      # Huber error coefficient
            r += kappa * self.max_stable_time - nu * self.accumulated_huber_error
        return r
    
    def close(self):
        """Clean up resources."""
        logger.info("Closing environment...")
        self.gazebo.close()
        self.sitl.close()


if __name__ == "__main__":
    from gymnasium.utils.env_checker import check_env, env_reset_passive_checker, env_step_passive_checker, check_step_determinism  
    from rl_training.utils.utils import load_config
    config = load_config('/home/pid_rl/rl_training/configs/default_config.yaml')

    env = ArdupilotEnv(config)
    # check_env(env)
    # env_reset_passive_checker(env)
    # env_step_passive_checker(env, env.action_space.sample())

    # ==== Check the step method ====
    # check_step_determinism(env)
    env.close()