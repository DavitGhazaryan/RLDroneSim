import gymnasium as gym
from gymnasium.utils import seeding
from gymnasium import spaces

import numpy as np
import logging
import sys
import math
import time
import atexit
from functools import partial

sys.path.insert(0, "/home/student/Dev/pid_rl")

from rl_training.utils.drone import Drone
from rl_training.utils.ardupilot_sitl import ArduPilotSITL
from rl_training.utils.utils import euler_to_quaternion, nrm
logger = logging.getLogger("Env")
from enum import Enum, auto

class Termination(Enum):    
    ATTITUDE_ERR = auto()   # excessive attitude error
    VEL_EXC = auto()        # velocity exceeded
    FLIP = auto()           # flip detected
    FAR = auto()            # too far from target

class BaseEnv(gym.Env):
    """
    Defines the main API of the Environment. main component is the drone which can be either hardware or Gazebo+SITL.
    """
    
    def __init__(self, config, hardware=False, instance=1):
        super().__init__()
        self.np_random, _ = seeding.np_random(None)  # init

        self.hardware = hardware
        
        self.config = config.get('environment_config', {})
        self.max_episode_steps = self.config.get('max_episode_steps')
        self.mode = self.config.get('mode')
        self.observable_gains = self.config['observable_gains'].split('+')
        self.observable_states = self.config['observable_states'].split('+')
        self.action_gains = self.config['action_gains'].split('+')
        self.reward_coefs = config.get('reward_config').get(self.mode)
        self.action_dt = self.config.get('action_dt')
        self.takeoff_altitude = self.config['takeoff_altitude']
        self.verbose = self.config['verbose']  # internal logging  TODO

        self.drone = Drone(config['drone_config'], self.verbose) if self.hardware else ArduPilotSITL(config['sitl_config'], config['gazebo_config'], instance, self.verbose)

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

        atexit.register(self.close)
    
    def _define_observation_space(self):
        """
        Observations are flattened into a single Box space for Stable Baselines compatibility.
        Order: [observable_gains, observable_states]
        """
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
        print("############################ RESET #############################################")
        start = time.time()

        if hasattr(self.action_space, "seed"):
            self.action_space.seed(seed)
        if hasattr(self.observation_space, "seed"):
            self.observation_space.seed(seed)
        self.episode_step = 0

        if not self.initialized:
            self.drone.start()

            master = self.drone._get_mavlink_connection()  
            hb = master.wait_heartbeat()
            messages = master.messages
            
            for gain in self.action_gains:
                self.ep_initial_gains[gain] = self.drone.get_param(gain)
                self.first_initial_gains[gain] = self.drone.get_param(gain)

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
            self.initialized = True
            
            if not self.hardware:
                self.mission_function = self._setup_mission()   # same x, y, z
                self.mission_function()

        else:
            self.eps_stable_time = 0
            self.max_stable_time = 0

            if not self.hardware:
                # works only for simulation
                self.ep_initial_pose, self.ep_initial_attitude, self.ep_initial_gains = self._get_random_initial_state()
                start_setting = time.time()
                for gain in self.action_gains:
                    self.drone.set_param_and_confirm(gain, self.ep_initial_gains[gain])
                end_setting = time.time()
                print(f"Param Setting time {end_setting-start_setting}")
                self.drone.reset([self.ep_initial_pose["x_m"], self.ep_initial_pose["y_m"], self.ep_initial_pose["z_m"]], euler_to_quaternion(None))            
            
        self.drone.wait(self.action_dt)   # no need to normalize the sleep time with speedup
        observation, info = self._get_observation(self.ep_initial_gains)
        end = time.time()

        print(f" Reset duration {end - start}")
        return observation, info  # observation, info
    
    def step(self, action):
        self.episode_step += 1
        print()
        start = time.time()
        obs, reward, done, truncated, info = self._step(action)
        end = time.time()
        print(f"Step duration {end-start}")
        return obs, reward, done, truncated, info
    
    def _get_observation(self, new_gains, messages=None):

        # Initialize flattened observation array
        total_dim = len(self.observable_gains) + len(self.observable_states)
        observation = np.zeros(total_dim, dtype=np.float32)

        
        # Fill gains first
        for i, observable_gain in enumerate(self.observable_gains):
            observation[i] = new_gains[observable_gain]

        # Fill states
        if messages is None:
            master = self.drone._get_mavlink_connection()
            master.wait_heartbeat()
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

    # for simulation related code
    def _setup_mission(self):
        if self.hardware:
            raise NotImplementedError("Mission is for sitl.")
        match self.mode:
            case 'altitude':
                self.goal_pose = {
                    'x_m': self.ep_initial_pose['x_m'],
                    'y_m': self.ep_initial_pose['y_m'],
                    'z_m': self.takeoff_altitude + 0.19
                }
                self.goal_orientation = self.ep_initial_attitude.copy()

                return partial(self.drone.takeoff_drone, self.takeoff_altitude)
            
            case 'position' | 'attitude' | 'stabilize' | 'althold':
                raise NotImplementedError("Position, attitude, stabilize, and althold modes are not implemented yet")
            case _:
                raise ValueError(f"Invalid mode: {self.mode}")

    def _get_random_initial_state(self):
        initial_gains = {}
        for gain in self.action_gains:
            initial_gains[gain] = max(self.first_initial_gains[gain] + self.np_random.uniform(-2.0, 2.0), 0)
        return {
            'x_m': self.goal_pose['x_m'],
            'y_m': self.goal_pose['y_m'],    
            'z_m': max(self.goal_pose['z_m']+ self.np_random.uniform(-2.0, 2.0), 0.3) 
        }, self.ep_initial_attitude, initial_gains
    
    def _step(self, action):
        """
        Handle flattened actions for Stable Baselines compatibility.
        Actions are changes to PID parameters that need to be applied.
        """

        if len(action) != len(self.action_gains):
            raise ValueError(f"Expected action of length {len(self.action_gains)}, got {len(action)}")
        start_getting = time.time()
        # Get current gains
        new_gains = {}
        for variable in self.action_gains:
            new_gains[variable] = self.drone.get_param(variable)
        end_getting = time.time()
        print(f"getting curr gains {end_getting-start_getting}")
        
        start_setting = time.time()
        for i, var in enumerate(self.action_gains):
            new_gains[var] += action[i]
            new_gains[var] = max(new_gains[var], 0)
            self.drone.set_param_and_confirm(var, new_gains[var])
        end_setting = time.time()

        print(f"setting new gains {end_setting-start_setting}")

        start_wait = time.time()
        self.drone.wait(self.action_dt)   # no need to normalize the sleep time with speedup
        end_wait = time.time()

        print(f"action_dt real {end_wait - start_wait}")

        start_msg = time.time()        
        # first get more complete info then construct observation from that        
        master = self.drone._get_mavlink_connection()
        master.wait_heartbeat()
        messages = master.messages    # same messages are used for single step
        end_msg = time.time()

        print(f"message reading {end_msg-start_msg}")

        start_obs = time.time()
        observation, info = self._get_observation(new_gains, messages)
        end_obs = time.time()
        print(f"get observation {end_obs-start_obs}")

        start_boil = time.time()
        terminated, reason = self._check_terminated(messages)
        if terminated:
            truncated = False
        else:
            terminated = False
            truncated = self.episode_step >= self.max_episode_steps
            if truncated:
                pass

        
        # Create proper info dictionary
        info = {
            'reason': reason,
            'episode_step': self.episode_step,
        }
        reward = self._compute_reward(messages, reason)
        print(reward)
        for i, var in enumerate(self.action_gains):
            info[var] = new_gains[var]
        end_boil = time.time()
        print(f"boilerplate code {end_boil -start_boil}")
        return observation, reward, terminated, truncated, info
        
    def close(self):
        """Clean up resources."""
        self.drone.close()

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


        if self.mode  == "altitude":
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


if __name__ == "__main__":
    from gymnasium.utils.env_checker import check_env, env_reset_passive_checker, env_step_passive_checker, check_step_determinism  
    from rl_training.utils.utils import load_config
    config = load_config('/home/pid_rl/rl_training/configs/default_config.yaml')

    env = BaseEnv(config)
    # check_env(env)
    # env_reset_passive_checker(env)
    # env_step_passive_checker(env, env.action_space.sample())

    # ==== Check the step method ====
    # check_step_determinism(env)
    env.close()