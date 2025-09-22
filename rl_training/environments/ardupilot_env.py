import enum
import gymnasium as gym
from gymnasium.utils import seeding
from gymnasium import spaces

import numpy as np
import logging
import sys
import math
import time
from pymavlink import mavutil
sys.path.insert(0, "/home/student/Dev/pid_rl")

from rl_training.utils.gazebo_interface import GazeboInterface
from rl_training.utils.ardupilot_sitl import ArduPilotSITL
from rl_training.utils.utils import euler_to_quaternion, nrm
logger = logging.getLogger("Env")
from enum import Enum, auto

class Termination(Enum):
    ATTITUDE_ERR = auto()   # excessive attitude error
    VEL_EXC = auto()        # velocity exceeded
    FLIP = auto()           # flip detected
    FAR = auto()            # too far from target

class ArdupilotEnv(gym.Env):
    """
    Initializes Gazebo and Ardupilot SITL, and provides a Gymnasium-compatible interface.
    Environment is intended to enable training if an RL agent that will find the optimal PID gains to put on an agent.  
    """
    
    def __init__(self, config, eval=False, instance=1):
        super().__init__()
        self.np_random, _ = seeding.np_random(None)  # init

        self.eval = eval

        self.config = config.get('environment_config', {})
        self.max_episode_steps = self.config.get('max_episode_steps')
        self.mode = self.config.get('mode')
        self.observable_gains = self.config['observable_gains'].split('+')
        self.observable_states = self.config['observable_states'].split('+')
        self.action_gains = self.config['action_gains'].split('+')
        self.reward_config = self.config["reward_config"]
        self.reward_coefs = config.get('reward_config').get(self.reward_config)

        self.action_dt = self.config.get('action_dt')
        self.takeoff_altitude = self.config['takeoff_altitude']

        self.verbose = self.config.get("verbose")

        if self.verbose:
            logger.setLevel(logging.INFO)
        else:
            logger.setLevel(logging.ERROR)

        self.gazebo = GazeboInterface(config['gazebo_config'], instance, self.verbose)
        self.sitl = ArduPilotSITL(config['ardupilot_config'], instance, self.verbose)

        # Episode tracking
        self.initialized = False
        self.episode_step = 0

        self.ep_initial_pose = None       # at the episode start {lat_deg:, lon_deg:, rel_alt_m:}
        self.ep_initial_attitude = None   # {pitch_deg:, roll_deg:, yaw_deg:}
        self.ep_initial_gains = {}      # {gain_name: value}
        self.first_initial_gains = {}      # {gain_name: value}
        
        self.mission_function = None
        self.goal_orientation = None   # {pitch_deg:, roll_deg:, yaw_deg:}
        self.goal_pose = None          # {latitude_deg:, longitude_deg:, relative_altitude_m:}
        self.eps_stable_time = 0
        self.max_stable_time = 0

        # Initialize spaces
        self.observation_space = self._define_observation_space()
        self.action_space = self._define_action_space()
    
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
        
        # Add bounds for states (-100 to 100)
        for _ in self.observable_states:
            lows.append(-100.0)
            highs.append(100.0)
        
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
        if self.eval:  # for eval only 0 actions
            lows = np.array([0.0] * total_dim, dtype=np.float32)
            highs = np.array([0.0] * total_dim, dtype=np.float32)
        else:
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
            #logger.info("🌎 Launching Gazebo simulation...")
            self.gazebo.start_simulation()   # waiting is done internally.
            self.gazebo.resume_simulation()
            #logger.debug("✅ Gazebo initialized")
            #logger.debug("🚁 Starting ArduPilot SITL...")
            self.sitl.start_sitl()
            #logger.debug(f"✅ SITL running (PID {info['pid']})")
            self.initialized = True

            ## Setup Mission
            master = self.sitl._get_mavlink_connection()  
            self.arm_drone(master)
            time.sleep(10)

            hb = master.wait_heartbeat()
            messages = master.messages
            
            for gain in self.action_gains:
                self.ep_initial_gains[gain] = self.sitl.get_param(master, gain)
                self.first_initial_gains[gain] = self.sitl.get_param(master, gain)
                

            self.ep_initial_pose = {
                'x_m': messages["LOCAL_POSITION_NED"].y,
                'y_m': messages["LOCAL_POSITION_NED"].x,
                'z_m': - messages["LOCAL_POSITION_NED"].z
            }
            attitude = messages["ATTITUDE"]

            self.ep_initial_attitude = {
                'pitch_deg': math.degrees(attitude.pitch),
                'roll_deg': math.degrees(attitude.roll),
                'yaw_deg': math.degrees(attitude.yaw)
            }

            self.mission_function = self._setup_mission()   # same x, y, z
            self.mission_function()
        else:
            #logger.info("Resetting the Environment")
            self.ep_initial_pose, self.ep_initial_attitude, self.ep_initial_gains = self._get_random_initial_state()
            self.eps_stable_time = 0
            self.max_stable_time = 0

            logger.info(f"Setting gains to {self.ep_initial_gains}")
            master = self.sitl._get_mavlink_connection()
            for gain in self.action_gains:
                self.sitl.set_param_and_confirm(gain, self.ep_initial_gains[gain])

            logger.info(f"Setting drone to {self.ep_initial_pose}")
            
            self.gazebo.pause_simulation()
            self.gazebo.transport_position(self.sitl.name, [self.ep_initial_pose["x_m"], self.ep_initial_pose["y_m"], self.ep_initial_pose["z_m"]], euler_to_quaternion(None))
            self.gazebo.resume_simulation()
            self.send_reset(master, self.ep_initial_pose["y_m"], self.ep_initial_pose["x_m"], self.ep_initial_pose["z_m"])
        self._gazebo_sleep(self.action_dt)   # no need to normalize the sleep time with speedup
        observation, info = self._get_observation()
        return observation, info  # observation, info

    def send_reset(self, master, n, e, agl, seq=None, retries=3, ack_timeout=1.5):
        CMD = 31010
        if seq is None:
            seq = int(time.time() * 1000) & 0x7FFFFFFF  # monotonic-ish

        def wait_ack():
            t0 = time.time()
            while time.time() - t0 < ack_timeout:
                msg = master.recv_match(type='COMMAND_ACK', blocking=False)
                if msg and int(msg.command) == CMD:
                    return int(msg.result)  # 0 = ACCEPTED
                time.sleep(0.01)
            return None

        for _ in range(retries):
            master.mav.command_long_send(
                master.target_system, master.target_component,
                CMD, 0,          # confirmation=0
                float(n), float(e), float(agl),
                float(seq), 0, 0, 0
            )
            res = wait_ack()
            if res == 0:  # ACCEPTED
                return True
            time.sleep(0.1)
        return False
    
    def _get_observation(self, messages=None):

        # Initialize flattened observation array
        total_dim = len(self.observable_gains) + len(self.observable_states)
        observation = np.zeros(total_dim, dtype=np.float32)
        
        master = self.sitl._get_mavlink_connection()
        master.wait_heartbeat()
        
        # Fill gains first
        for i, observable_gain in enumerate(self.observable_gains):
            gain_value  = self.sitl.get_param(master, observable_gain)
            observation[i] = gain_value
        
        # Fill states
        if messages is None:
            messages = master.messages
        
        for i, observable_state in enumerate(self.observable_states):
            idx = len(self.observable_gains) + i
            if observable_state == 'alt_err':
                state_value = messages["NAV_CONTROLLER_OUTPUT"].alt_error
            elif observable_state == 'vZ_err':
                state_value = messages["DEBUG_VECT"].z
            elif observable_state == 'accZ_err':
                state_value = messages["PID_TUNING[4]"].desired - messages["PID_TUNING[4]"].achieved   # underneath it is not desired but target
            else:
                raise NotImplemented("Observation not available")
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

                return self.takeoff_drone
            
            case 'position' | 'attitude' | 'stabilize' | 'althold':
                raise NotImplementedError("Position, attitude, stabilize, and althold modes are not implemented yet")
            case _:
                raise ValueError(f"Invalid mode: {self.mode}")

    def _get_random_initial_state(self):
        initial_gains = {}
        for gain in self.action_gains:
            # initial_gains[gain] = self.np_random.uniform(0.8, 5.2)
            # initial_gains[gain] = max(self.ep_initial_gains[gain] + self.np_random.uniform(-3.0, 3.0), 0) if not self.eval else self.ep_initial_gains[gain]
            initial_gains[gain] = max(self.first_initial_gains[gain] + self.np_random.uniform(-2.0, 2.0), 0) if not self.eval else self.ep_initial_gains[gain]
        return {
            'x_m': self.goal_pose['x_m'],
            'y_m': self.goal_pose['y_m'],    
            'z_m': max(self.goal_pose['z_m']+ self.np_random.uniform(-2.0, 2.0), 0.3) 
        }, self.ep_initial_attitude, initial_gains
    
    def step(self, action):
        self.episode_step += 1
        obs, reward, done, truncated, info = self._step(action)
        return obs, reward, done, truncated, info

    def _step(self, action):
        """
        Handle flattened actions for Stable Baselines compatibility.
        Actions are changes to PID parameters that need to be applied.
        """

        if len(action) != len(self.action_gains):
            raise ValueError(f"Expected action of length {len(self.action_gains)}, got {len(action)}")
        master = self.sitl._get_mavlink_connection()
        hb = master.wait_heartbeat()

        # Get current gains
        new_gains = {}
        for variable in self.action_gains:
            new_gains[variable] = self.sitl.get_param(master, variable)
        for i, var in enumerate(self.action_gains):
            new_gains[var] += action[i]
            new_gains[var] = max(new_gains[var], 0)
            self.sitl.set_param_and_confirm(var, new_gains[var])
        self._gazebo_sleep(self.action_dt)   # no need to normalize the sleep time with speedup

        # first get more complete info then construct observation from that        
        master.wait_heartbeat()
        messages = master.messages
        # print(messages)
        observation, info = self._get_observation(messages)
        terminated, reason = self._check_terminated(messages)
        if terminated:
            truncated = False
            #logger.info(f"Terminating the episode {reason}")
        else:
            terminated = False
            truncated = self.episode_step >= self.max_episode_steps
            if truncated:
                pass
                #logger.info(f"Truncating the Episode")

        
        # Create proper info dictionary
        info = {
            'reason': reason,
            'episode_step': self.episode_step,
            'max_stable_time': self.max_stable_time,
        }
        reward = self._compute_reward(messages, reason)
        for i, var in enumerate(self.action_gains):
            info[var] = new_gains[var]

        return observation, reward, terminated, truncated, info

    def arm_drone(self, master, timeout=10):
        master.wait_heartbeat()
        t0 = time.time()
        while time.time() - t0 < timeout:
            hb = master.recv_match(type="HEARTBEAT", blocking=True, timeout=1.0)
            if not hb:
                continue
            if hb.system_status == mavutil.mavlink.MAV_STATE_STANDBY:
                break
            time.sleep(0.1)
        if time.time() - t0 >= timeout:
            raise TimeoutError("Failed to arm the drone")
        
        master.mav.command_long_send(
            master.target_system, master.target_component,
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 1, 0, 0, 0, 0, 0, 0
        )

        master.recv_match(type='COMMAND_ACK', blocking=True, timeout=10)

    def takeoff_drone(self):
        master = self.sitl._get_mavlink_connection()
        master.wait_heartbeat()
        master.mav.command_long_send(
            master.target_system,
            master.target_component,
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
            0,          # confirmation
            0, 0, 0, 0, # params 1–4 (unused here)
            0, 0,       # lat, lon (0 = current location)
            self.takeoff_altitude    # param7 = target altitude (meters, AMSL)
        )
        ack = master.recv_match(type='COMMAND_ACK', blocking=True, timeout=10)
        if ack and ack.result == mavutil.mavlink.MAV_RESULT_ACCEPTED:
            #logger.info(f"Takeoff to {self.takeoff_altitude} m commanded")
            pass
        else:
            logger.error(f"Failed to takeoff: {ack}")

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

    def _gazebo_sleep(self, duration):
        """
        Sleep for the given duration (in seconds) using Gazebo simulation time.
        """
        start_time = self.gazebo.get_sim_time()
        while True:
            time.sleep(0.001)
            current_time = self.gazebo.get_sim_time()
            if current_time - start_time >= duration:
                break

    def _check_vicinity_status(self, pos_error_cm, alt_error_cm):
        """
        Check if the drone is in vicinity of the goal.
        Uses hysteresis to prevent flickering between vicinity states.
        """
        pos_error_cm = abs(pos_error_cm)
        alt_error_cm = abs(alt_error_cm)

        eps_in = 5  # Inner vicinity threshold
        prev_in_vicinity = self.eps_stable_time > 0
        in_vicinity = (pos_error_cm <= eps_in and alt_error_cm <= eps_in) 
        return in_vicinity

    def _check_terminated(self, messages):
        message = messages["LOCAL_POSITION_NED"]
        attitude = messages["ATTITUDE"]

        # 1. Attitude Error
        if abs(math.degrees(attitude.pitch) - self.goal_orientation['pitch_deg']) > 15 or abs(math.degrees(attitude.roll) - self.goal_orientation['roll_deg']) > 15:
            return True, Termination.ATTITUDE_ERR
        
        # Velocity magnitude
        if abs(message.vx) > 4 or abs(message.vy) > 4 or abs(message.vz) > 4:
            return True, Termination.VEL_EXC
        
        # 2. Flip (pitch or roll > 90 deg)
        if abs(math.degrees(attitude.pitch)) > 90 or abs(math.degrees(attitude.roll)) > 90:
            return True, Termination.FLIP
        
        # 3. 2x farther xy from goal than originally
        dist_xy_init = np.linalg.norm(np.array([self.ep_initial_pose["x_m"], self.ep_initial_pose["y_m"], self.ep_initial_pose["z_m"]]) 
                                   - np.array([self.goal_pose["x_m"], self.goal_pose["y_m"], self.goal_pose["z_m"]]))
        dist_xy_now = np.linalg.norm(np.array([message.y, message.x, -message.z]) 
                                  - np.array([self.goal_pose["x_m"], self.goal_pose["y_m"], self.goal_pose["z_m"]]))
       
        # 4. 2x farther altitude from goal than originally
        alt_init = abs(self.ep_initial_pose["z_m"] - self.goal_pose["z_m"])
        alt_now = abs(-message.z - self.goal_pose["z_m"])
       
        if (dist_xy_now > 1.5 * dist_xy_init and dist_xy_now > 0.1) or (alt_now > alt_init * 3  and alt_now > 0.1):
            return True, Termination.FAR


        # 5. goal is reached - check both position and altitude
        pos_err_cm = messages["NAV_CONTROLLER_OUTPUT"].wp_dist   # in cm integers
        alt_err_m = messages["NAV_CONTROLLER_OUTPUT"].alt_error
        
        # Check if in vicinity using helper method
        in_vicinity = self._check_vicinity_status(pos_err_cm, alt_err_m * 100)
        
        # Update stable time tracking (consolidated logic)
        if in_vicinity:
            self.eps_stable_time += 1
        else:
            if self.eps_stable_time > self.max_stable_time:
                self.max_stable_time = self.eps_stable_time
            self.eps_stable_time = 0
        return False, None 

    def _compute_reward(self, messages, reason):
        """
        Compute reward based on stable-time mechanism.
        
        Reward components:
        """
        
        w = self.reward_coefs
        tol = w.get("tolerance")


        if self.reward_config == "hover":
            ## initialize with 100 step reward
            r = self.reward_coefs.get("step_reward")
            
            # Calculate position error 
            pos_err_cm = nrm(messages["NAV_CONTROLLER_OUTPUT"].wp_dist, tol["xy_tol"])   # in cm integers
            alt_err = nrm(messages["NAV_CONTROLLER_OUTPUT"].alt_error, tol["alt_tol"])
            
            # Velocity error components 
            vel_err_n = nrm(messages["DEBUG_VECT"].y, tol["vel_n_tol"])
            vel_err_e = nrm(messages["DEBUG_VECT"].x, tol["vel_e_tol"]) 
            vel_err_d = nrm(messages["DEBUG_VECT"].z, tol["vel_d_tol"])
            
            # Acceleration components if available
            acc_err_n = nrm(messages["PID_TUNING[1]"].desired - messages["PID_TUNING[1]"].achieved,   tol["acc_n_tol"])
            acc_err_e = nrm(messages["PID_TUNING[2]"].desired - messages["PID_TUNING[2]"].achieved,   tol["acc_e_tol"])
            acc_err_yaw = nrm(messages["PID_TUNING[3]"].desired - messages["PID_TUNING[3]"].achieved, tol["acc_yaw_tol"])
            acc_err_d = nrm(messages["PID_TUNING[4]"].desired - messages["PID_TUNING[4]"].achieved,   tol["acc_d_tol"])

            # Weighted error aggregation
            e_t = (
                  w.get("alt_w")     * alt_err 
                + w.get("xy_w")      * pos_err_cm
                + w.get("velN_w")    * vel_err_n 
                + w.get("velE_w")    * vel_err_e 
                + w.get("velZ_w")    * vel_err_d 
                + w.get("accN_w")    * acc_err_n 
                + w.get("accE_w")    * acc_err_e 
                + w.get("accZ_w")    * acc_err_d 
                + w.get("acc_yaw_w") * acc_err_yaw
            )

            # Decrease reward based on the errors
            r -= e_t

            if reason:
                match reason:
                    case Termination.ATTITUDE_ERR:
                        r = self.reward_coefs.get("crash_penalty_att")
                    case Termination.VEL_EXC:
                        r = self.reward_coefs.get("crash_penalty_vel")
                    case Termination.FLIP:
                        r = self.reward_coefs.get("crash_penalty_flip")
                    case Termination.FAR:
                        r = self.reward_coefs.get("crash_penalty_far")
                    
            #  Timeout reward and has not crashed
            if self.episode_step >= self.max_episode_steps and r > 0:  
                r += self.reward_coefs.get("success_reward")

                kappa = self.reward_coefs.get("stable_time_coef")
                r += kappa * self.max_stable_time
            
            return r
        else:
            raise NotImplementedError()
        
    def close(self):
        """Clean up resources."""
        #logger.info("Closing environment...")
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