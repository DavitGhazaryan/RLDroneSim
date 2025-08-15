"""
Main Ardupilot environment implementing Gymnasium API.
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import logging
import sys
import asyncio
import math

sys.path.insert(0, "/home/student/Dev/pid_rl")

from rl_training.utils.gazebo_interface import GazeboInterface
from rl_training.utils.ardupilot_sitl import ArduPilotSITL
from rl_training.utils.utils import euler_to_quaternion, lat_lon_to_xy_meters, xy_meters_to_lat_lon

logger = logging.getLogger("Env")
logger.setLevel(logging.INFO)

PID_KEYS = [
    # Not Complete
    "ATC_ANG_PIT_P",     # Attitude angle PID Gains
    "ATC_ANG_RLL_P",
    "ATC_ANG_YAW_P",
    "ATC_RAT_PIT_P",     # Attitude rate PID Gains
    "ATC_RAT_RLL_P",
    "ATC_RAT_YAW_P",
    "PSC_POSZ_P",       # Position Z PID Gains
    "PSC_VELZ_P",       # Velocity Z PID Gains
]

class ArdupilotEnv(gym.Env):
    """
    Initializes Gazebo and Ardupilot SITL, and provides a Gymnasium-compatible interface.
    Environment is intended to enable training if an RL agent that will find the optimal PID gains to put on an agent.  
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.__compatibility_checks()

        self.gazebo = GazeboInterface(self.config['gazebo_config'])
        self.sitl = ArduPilotSITL(self.config['ardupilot_config'])
        self.loop = asyncio.get_event_loop()


        # Episode tracking
        self.initialized = False
        self.episode_step = 0
        self.environment_config = config.get('environment_config', {})
        self.observable_gains = self.environment_config['observable_gains'].split('+')
        self.observable_states = self.environment_config['observable_states'].split('+')
        self.action_gains = self.environment_config['action_gains'].split('+')
        self.max_episode_steps = self.environment_config.get('max_episode_steps', 50)
        
        self.origin_pose = None       # {latitude_deg:, longitude_deg:, relative_altitude_m:}
        self.initial_pose = None       # {latitude_deg:, longitude_deg:, relative_altitude_m:}
        self.initial_attitude = None   # {pitch_deg:, roll_deg:, yaw_deg:}
        self.initial_gains = None      # {gain_name: value}
        
        self._async_mission_function = None
        self.action_dt = self.environment_config.get('action_dt', 1.0)
        self.goal_orientation = None   # {pitch_deg:, roll_deg:, yaw_deg:}
        self.goal_pose = None          # {latitude_deg:, longitude_deg:, relative_altitude_m:}
        self.stable_time = 0
        self.max_stable_time = 0
        self.accumulated_huber_error = 0.0  # Track Huber errors for timeout reward
        
        # Initialize spaces
        self.observation_space = self._define_observation_space()
        self.action_space = self._define_action_space()
        
    def _define_observation_space(self):
        """
        Observations are of two parts: observable gains and observable states
        They should be stored as a dict with keys: observable_gains and observable_states
        """
        dictionary = spaces.Dict()
        for observable_gain in self.observable_gains:
            dictionary[observable_gain] = spaces.Box(low=0, high=100, dtype=np.float32)
        for observable_state in self.observable_states:
            dictionary[observable_state] = spaces.Box(low=-np.inf, high=np.inf, dtype=np.float32)
        return dictionary
    
    def _define_action_space(self):
        """
        Action space is a dictionary with keys: action_gains
        """
        dictionary = spaces.Dict()
        for action_gain in self.action_gains:
            dictionary[action_gain] = spaces.Box(low=-0.1, high=0.1, dtype=np.float32)
        return dictionary

    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)

        self.episode_step = 0

        if not self.initialized:
            logger.info("🌎 Launching Gazebo simulation...")
            self.gazebo.start_simulation()
            self.gazebo._wait_for_startup()
            self.gazebo.resume_simulation()
            logger.info("✅ Gazebo initialized")
            logger.info("🚁 Starting ArduPilot SITL...")
            self.sitl.start_sitl()
            info = self.sitl.get_process_info()
            logger.info(f"✅ SITL running (PID {info['pid']})")
            self.initialized = True
            self.loop.run_until_complete(self._async_arm())
            ## Resetting Ends Here: Drone is Armed

            ## Setup Mission
            pose = self.loop.run_until_complete(self.sitl.get_pose_async())
            self.origin_pose = {
                'latitude_deg': pose.latitude_deg,
                'longitude_deg': pose.longitude_deg,
                'absolute_altitude_m': pose.absolute_altitude_m,
                'relative_altitude_m': pose.relative_altitude_m
            }
            self.initial_pose = {
                'latitude_deg': pose.latitude_deg,
                'longitude_deg': pose.longitude_deg,
                'relative_altitude_m': pose.relative_altitude_m
            }
            attitude = self.loop.run_until_complete(self.sitl.get_attitude_async())
            self.initial_attitude = {
                'pitch_deg': attitude.pitch_deg,
                'roll_deg': attitude.roll_deg,
                'yaw_deg': attitude.yaw_deg
            }

            self._async_mission_function = self._setup_mission()
            self.loop.run_until_complete(self._async_mission_function())
        else:
            self.initial_pose, self.initial_attitude, self.initial_gains = self.get_random_initial_state()
            self.stable_time = 0
            self.max_stable_time = 0
            self.accumulated_huber_error = 0.0

            #TODO  Position needs to be moved to odometry
            # self.gazebo.transport_position(self.sitl.name, self.initial_pose, self.initial_attitude)
            # new_xy = lat_lon_to_xy_meters(self.origin_pose, self.initial_pose['latitude_deg'], self.initial_pose['longitude_deg'])
            for gain in self.action_gains:
                self.loop.run_until_complete(self.sitl.set_param_async(gain, self.initial_gains[gain]))
            self.gazebo.transport_position(self.sitl.name, [0, 0, self.initial_pose['relative_altitude_m']], euler_to_quaternion(self.initial_attitude))
        
        observation, info = self.loop.run_until_complete(self._async_get_observation())
        return observation, info  # observation, info


    async def _async_get_observation(self):
        """
        Get the observations of the environment.
        This function is called when the environment is reset.
        """
        drone = await self.sitl._get_mavsdk_connection()
        observation = {}
        for observable_gain in self.observable_gains:
            observation[observable_gain] = await drone.param.get_param_float(observable_gain)
        position = await anext(drone.telemetry.position())
        for observable_state in self.observable_states:
            observation[observable_state] = getattr(position, observable_state)
        info = {}
        return observation, info 
    
    def _setup_mission(self):
        match self.config['environment_config']['mode']:
            case 'altitude':
                self.takeoff_altitude = self.config['environment_config']['takeoff_altitude']
                self.goal_pose = {
                    'latitude_deg': self.initial_pose['latitude_deg'],
                    'longitude_deg': self.initial_pose['longitude_deg'],
                    'relative_altitude_m': self.takeoff_altitude
                }
                self.goal_orientation = self.initial_attitude.copy()
                return self._async_takeoff
            
            case 'position' | 'attitude' | 'stabilize' | 'althold':
                raise NotImplementedError("Position, attitude, stabilize, and althold modes are not implemented yet")
            case _:
                raise ValueError(f"Invalid mode: {self.config['environment_config']['mode']}")

    def get_random_initial_state(self):
        # TODO: Implement random initial state
        initial_gains = {}
        for gain in self.action_gains:
            initial_gains[gain] = np.random.uniform(0.8, 5.2)
        return {
            'latitude_deg': self.initial_pose['latitude_deg'],
            'longitude_deg': self.initial_pose['longitude_deg'],    
            'relative_altitude_m': 1.2 + np.random.uniform(-0.5, 0.5)
            # 'relative_altitude_m': self.initial_pose['relative_altitude_m'] + np.random.uniform(-0.5, 0.5)
        }, self.initial_attitude, initial_gains
    
    def step(self, action):
        self.episode_step += 1
        obs, reward, done, truncated, info = self.loop.run_until_complete(self._async_step(action))
        # self.old_observation = obs
        return obs, reward, done, truncated, info

    async def _async_step(self, action):
        """
        get ths action which is the changes of that need to be done in the PID params
        we need to compute the new PID params by adding the action to the old PID params
        and set them
        """
        drone = await self.sitl._get_mavsdk_connection()
        # take the current gains
        new_gains = {
            variable: await drone.param.get_param_float(variable)
            for variable in self.action_gains                }
        
        # apply the action to the new gains and set them
        for var in self.action_gains:
            new_gains[var] += action[var]
            await drone.param.set_param_float(var, new_gains[var].item())

        await self._gazebo_sleep(self.action_dt)   # no need to normalize the sleep time with speedup

        # get the new observation
        observation, info = await self._async_get_observation()
        pose = await anext(drone.telemetry.position())
        attitude = await anext(drone.telemetry.attitude_euler())
        

        def _check_terminated(pose, attitude):
            # Assume pose : [latitude_deg:, longitude_deg:, absolute_altitude_m:, relative_altitude_m:]
            #             attitude: [pitch_deg:, roll_deg:, yaw_deg:]
            
            # 1. Crash: the difference between current and target attitude is more that 30 degree
            if abs(attitude.pitch_deg - self.goal_orientation['pitch_deg']) > 30 or abs(attitude.roll_deg - self.goal_orientation['roll_deg']) > 30:
                return True, "crash: excessive pitch/roll"

            # 2. Flip (pitch or roll > 90 deg)
            if abs(attitude.pitch_deg) > 90 or abs(attitude.roll_deg) > 90:
                return True, "crash: flip"

            # 3. 2x farther from goal than originally
            dist_init = np.linalg.norm(np.array([self.initial_pose["latitude_deg"], self.initial_pose["longitude_deg"]]) - np.array([self.goal_pose["latitude_deg"], self.goal_pose["longitude_deg"]]))
            dist_now = np.linalg.norm(np.array([pose.latitude_deg, pose.longitude_deg]) - np.array([self.goal_pose["latitude_deg"], self.goal_pose["longitude_deg"]]))
            
            if dist_now > 1.5 * dist_init:
                return True, "too far from goal"
            
            # 4. goal is reached - check both position and altitude
            pos_error = np.linalg.norm(np.array([pose.latitude_deg, pose.longitude_deg]) - np.array([self.goal_pose["latitude_deg"], self.goal_pose["longitude_deg"]]))
            alt_error = abs(pose.relative_altitude_m - self.goal_pose["relative_altitude_m"])
            
            # Check if in vicinity using helper method
            in_vicinity = self._check_vicinity_status(pos_error, alt_error)
            
            # Update stable time tracking (consolidated logic)
            if in_vicinity:
                self.stable_time += 1
                if self.stable_time >= self.max_episode_steps/2:
                    return True, f"stable for {self.stable_time} steps"
            else:
                if self.stable_time > self.max_stable_time:
                    self.max_stable_time = self.stable_time
                self.stable_time = 0
            return False, None

        terminated, reason = _check_terminated(pose, attitude)
        if terminated:
            truncated = False
        else:
            terminated = False
            truncated = self.episode_step >= self.max_episode_steps
        reward = self._compute_reward(pose, attitude)
        # info modify
        # new_info = observation_info + reason
        
        return observation, reward, terminated, truncated, reason

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
        drone.action.set_takeoff_altitude(self.takeoff_altitude)
        await drone.action.takeoff()
        await asyncio.sleep(5/self.sitl.speedup)

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
        
        prev_in_vicinity = self.stable_time > 0
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
        pos_error = np.linalg.norm(np.array([pose.latitude_deg, pose.longitude_deg]) - np.array([self.goal_pose["latitude_deg"], self.goal_pose["longitude_deg"]]))
        
        # Calculate altitude error
        alt_error = abs(pose.relative_altitude_m - self.goal_pose["relative_altitude_m"])
        
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
            r += gamma_s + eta * math.log1p(self.stable_time)
        
        # Success termination bonus
        T_stable = self.max_episode_steps // 2  # Success threshold
        if self.stable_time >= T_stable:
            R_succ = 100.0  # Success bonus
            r += R_succ
        
        # Timeout reward (when episode ends due to max steps)
        if self.episode_step >= self.max_episode_steps:
            kappa = 1    # Stable time coefficient
            nu = 0.1      # Huber error coefficient
            r += kappa * self.max_stable_time - nu * self.accumulated_huber_error
        return r

    def __compatibility_checks(self):
        if self.config['environment_config']['mode'] not in ['position', 'attitude', 'stabilize', 'althold', 'altitude']:
            raise ValueError(f"Invalid mode: {self.config['environment_config']['mode']}")
        logger.warning(f"compatibility checks is not implemented")

    def close(self):
        """Clean up resources."""
        logger.info("Closing environment...")
        self.gazebo.close()
        self.sitl.close()
