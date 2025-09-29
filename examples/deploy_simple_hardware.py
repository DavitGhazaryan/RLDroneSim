import numpy as np
import logging
import sys
import math
import time
import atexit
from functools import partial
import os 
import torch
import yaml

sys.path.insert(0, "/home/student/Dev/pid_rl")

from rl_training.utils.drone import Drone
from rl_training.utils.utils import nrm
logger = logging.getLogger("Env")
from enum import Enum, auto

class Termination(Enum):    
    ATTITUDE_ERR = auto()   # excessive attitude error
    VEL_EXC = auto()        # velocity exceeded
    FLIP = auto()           # flip detected
    FAR = auto()            # too far from target

class DeployEnv:
    def __init__(self, config, instance=1):
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

        self.drone = Drone(config['drone_config'], self.verbose)

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
    
    
    def reset(self, seed=None, options=None):
        print("############################ RESET #############################################")
        start = time.time()

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
        
        else:
            self.eps_stable_time = 0
            self.max_stable_time = 0
            
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
    
    @staticmethod
    def deploy(model_path):
        """Static method to load the model, read the config, and set up the deployment loop."""
        # 1. Load model
        model_file = os.path.join(model_path, "policy_actor_scripted.pt")
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Model file not found at {model_file}")
        actor = torch.jit.load(model_file, map_location="cpu").eval()

        # 2. Load the environment configuration from cfg.yaml
        config_file = os.path.join(model_path, "cfg.yaml")
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Config file not found at {config_file}")
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)

        # 3. Initialize the deployment environment
        env = DeployEnv(config)

        # 4. Start the deployment loop
        try:
            while True:
                # 4.1 Get observation (current state of the drone)
                current_state = env._get_observation(env.ep_initial_gains)

                # 4.2 Prepare the observation for the model
                observation = torch.tensor(current_state[0], dtype=torch.float32).unsqueeze(0)  # Add batch dimension

                # 4.3 Get the action from the actor (policy)
                with torch.no_grad():
                    action = actor(observation).squeeze(0).cpu().numpy()

                # 4.4 Send action to the drone
                env._step(action)

                # 4.5 Check termination criteria (goal reached or episode timeout)
                messages = env.drone._get_mavlink_connection().messages
                distance_to_goal = np.linalg.norm(np.array([messages["LOCAL_POSITION_NED"].x,
                                                           messages["LOCAL_POSITION_NED"].y,
                                                           -messages["LOCAL_POSITION_NED"].z]) - np.array([env.goal_pose["x_m"],
                                                                                                          env.goal_pose["y_m"],
                                                                                                          env.goal_pose["z_m"]]))
                if distance_to_goal < 1.0:  # Goal reached
                    print("Goal reached!")
                    break

                # Optional: Add a delay for control frequency
                time.sleep(0.1)

        except KeyboardInterrupt:
            print("Deployment stopped manually.")
        finally:
            env.close()


if __name__ == "__main__":
    from rl_training.utils.utils import load_config
    config = load_config('/home/pid_rl/rl_training/configs/default_config.yaml')

    env = DeployEnv(config)
    env.close()