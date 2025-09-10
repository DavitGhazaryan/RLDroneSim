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
from rl_training.utils.utils import euler_to_quaternion
logger = logging.getLogger("Env")
from enum import Enum, auto

class Termination(Enum):
    ATTITUDE_ERR = auto()   # excessive attitude error
    VEL_EXC = auto()        # velocity exceeded
    FLIP = auto()           # flip detected
    FAR = auto()            # too far from target
    SUCCESS = auto()        # task completed successfully         

class HardEnv(gym.Env):

    def __init__(self, config, eval=False, instance=1):
        super().__init__()
        self.np_random, _ = seeding.np_random(None)
        self.eval = eval

        self.config = config.get('environment_config', {})
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


        # Episode tracking
        self.initialized = False
        self.episode_step = 0

        self.ep_initial_pose = None       # at the episode start {lat_deg:, lon_deg:, rel_alt_m:}
        self.ep_initial_attitude = None   # {pitch_deg:, roll_deg:, yaw_deg:}
        self.ep_initial_gains = {}      # {gain_name: value}
        
        self.mission_function = None
        self.goal_orientation = None   # {pitch_deg:, roll_deg:, yaw_deg:}
        self.goal_pose = None          # {latitude_deg:, longitude_deg:, relative_altitude_m:}
        self.eps_stable_time = 0
        self.max_stable_time = 0
        self.accumulated_huber_error = 0.0  # Track Huber errors for timeout reward

        # Initialize spaces
        self.observation_space = self._define_observation_space()
        self.action_space = self._define_action_space()

        # Connection
        self._mavlink_master = None

        # Log space information for debugging
        #logger.info(f"🔧 Environment spaces initialized:")
        #logger.info(f"   Observation space: {self.observation_space}")
        #logger.info(f"   Action space: {self.action_space}")
        #logger.info(f"   Observable gains: {self.observable_gains}")
        #logger.info(f"   Observable states: {self.observable_states}")
        #logger.info(f"   Action gains: {self.action_gains}")
        
        
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
        
        self.episode_step = 0
        self.eps_stable_time = 0
        self.max_stable_time = 0
        self.accumulated_huber_error = 0.0

        master = self._get_mavlink_connection()  
        # time.sleep(2)

        hb = master.wait_heartbeat()
        messages = master.messages

        for gain in self.action_gains:
            self.ep_initial_gains[gain] = self.get_param(master, gain)
            
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
        if not self.initialized:
            self.mission_function = self._setup_mission()   # same x, y, z
            # self.mission_function()   # No need
            self.initialized = True

        time.sleep(self.action_dt)  

        observation, info = self._get_observation()
        return observation, info  # observation, info
    
    def _get_observation(self, messages=None):

        # Initialize flattened observation array
        total_dim = len(self.observable_gains) + len(self.observable_states)
        observation = np.zeros(total_dim, dtype=np.float32)
        
        master = self._get_mavlink_connection()
        master.wait_heartbeat()
        
        # Fill gains first
        for i, observable_gain in enumerate(self.observable_gains):
            gain_value  = self.get_param(master, observable_gain)
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

    def step(self, action):
        self.episode_step += 1
        print("new step")
        obs, reward, done, truncated, info = self._step(action)
        return obs, reward, done, truncated, info

    def _step(self, action):
        """
        Handle flattened actions for Stable Baselines compatibility.
        Actions are changes to PID parameters that need to be applied.
        """

        if len(action) != len(self.action_gains):
            raise ValueError(f"Expected action of length {len(self.action_gains)}, got {len(action)}")
        master = self._get_mavlink_connection()
        hb = master.wait_heartbeat()

        # Get current gains
        new_gains = {}
        for variable in self.action_gains:
            new_gains[variable] = self.get_param(master, variable)
        for i, var in enumerate(self.action_gains):
            new_gains[var] += action[i]
            new_gains[var] = max(new_gains[var], 0)
            self.set_param_and_confirm(master, var, new_gains[var])
        
        time.sleep(self.action_dt)  

        # first get more complete info then construct observation from that        
        master.wait_heartbeat()
        messages = master.messages
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
            'stable_time': self.eps_stable_time,
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
            time.sleep(0.01)
        if time.time() - t0 >= timeout:
            raise TimeoutError("Failed to arm the drone")
        
        master.mav.command_long_send(
            master.target_system, master.target_component,
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 1, 0, 0, 0, 0, 0, 0
        )

        master.recv_match(type='COMMAND_ACK', blocking=True, timeout=10)

    def takeoff_drone(self):
        master = self._get_mavlink_connection()
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
        print("check terminated")
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
        
        
        if dist_xy_now > 1.5 * dist_xy_init and dist_xy_now > 0.1:
            return True, Termination.FAR

        # 5. goal is reached - check both position and altitude
        pos_err_cm = messages["NAV_CONTROLLER_OUTPUT"].wp_dist   # in cm integers
        alt_err_m = messages["NAV_CONTROLLER_OUTPUT"].alt_error
        
        # Check if in vicinity using helper method
        in_vicinity = self._check_vicinity_status(pos_err_cm, alt_err_m * 100)
        
        # Update stable time tracking (consolidated logic)
        if in_vicinity:
            self.eps_stable_time += 1
            if self.eps_stable_time >= self.max_episode_steps/2:
                return True, Termination.SUCCESS
        else:
            if self.eps_stable_time > self.max_stable_time:
                self.max_stable_time = self.eps_stable_time
            self.eps_stable_time = 0
        return False, None 

    def _compute_reward(self, messages, reason):
        if self.reward_config == "hover":
            ### Abstract with ArdupilotEnv
            return 0
        else:
            raise NotImplementedError()
    
    def _get_mavlink_connection(self):
        """
        Get or create a cached MAVLink connection.
        
        Returns:
            mavutil.mavlink_connection: Active connection
            
        Raises:
            RuntimeError: If connection cannot be established
        """
        if self._mavlink_master is None or not hasattr(self._mavlink_master, 'target_system'):
            addr = f'00000000000' 
            
            try:
                self._mavlink_master = mavutil.mavlink_connection(addr)
                hb = self._mavlink_master.wait_heartbeat()
                self._mavlink_master.target_system = hb.get_srcSystem()

                self._mavlink_master.target_component = hb.get_srcComponent() or mavutil.mavlink.MAV_COMP_ID_AUTOPILOT1
                #logger.debug(f"HB from sys:{self._mavlink_master.target_system} comp:{self._mavlink_master.target_component}")

                # channels
                PID_TUNING = 194
                NAV_CONTROLLER_OUTPUT = 62

                RATE_HZ = 100    # TODO 
                self.set_message_interval(self._mavlink_master, PID_TUNING, RATE_HZ)
                self.set_message_interval(self._mavlink_master, NAV_CONTROLLER_OUTPUT, RATE_HZ)

                GCS_PID_MASK_VALUE = 0xFFFF
                self.set_param_and_confirm(self._mavlink_master, "GCS_PID_MASK", GCS_PID_MASK_VALUE)

                print(f"Established MAVLink connection to {addr}")
                #logger.debug(f"Established MAVLink connection to {addr}")
            except Exception as e:
                self._mavlink_master = None
                raise RuntimeError(f"Failed to establish MAVLink connection to {addr}: {e}")
                
        return self._mavlink_master

    def set_param_and_confirm(self, master, name_str, value, timeout=3.0):
        name_bytes = name_str.encode("ascii", "ignore")

        is_float = isinstance(value, float)
        ptype = (mavutil.mavlink.MAV_PARAM_TYPE_REAL32
                if is_float else mavutil.mavlink.MAV_PARAM_TYPE_INT32)

        master.mav.param_set_send(master.target_system, master.target_component,
                                name_bytes, float(value), ptype)

        t0 = time.time()
        while time.time() - t0 < timeout:
            msg = master.recv_match(type="PARAM_VALUE", blocking=True, timeout=timeout)
            if not msg:
                break
            pid = (msg.param_id.decode("ascii","ignore") if isinstance(msg.param_id,(bytes,bytearray))
                else str(msg.param_id)).rstrip("\x00")
            if pid == name_str:
                return True
        logger.warn("Param is NOT SET")
        return False

    def get_param(self, master, param_name, timeout=3.0, resend_every=0.5):
        name16 = param_name[:16]  # enforce MAVLink 16-char limit
        name_bytes = name16.encode("ascii", "ignore")

        t0 = time.time()
        last = 0.0
        while time.time() - t0 < timeout:
            if time.time() - last >= resend_every:
                master.mav.param_request_read_send(master.target_system, master.target_component, name_bytes, -1)
                last = time.time()
            msg = master.recv_match(type="PARAM_VALUE", blocking=True, timeout=0.2)
            if not msg:
                continue
            pid = (msg.param_id.decode("ascii", "ignore") if isinstance(msg.param_id, (bytes, bytearray))
                else str(msg.param_id)).rstrip("\x00")
            if pid == name16:
                return msg.param_value
            
        raise TimeoutError(f"Timeout: param {param_name} not received")

    def _close_mavlink_connection(self):
        """Close and cleanup cached MAVLink connection."""
        if self._mavlink_master is not None:
            try:
                self._mavlink_master.close()
            except:
                pass
            self._mavlink_master = None

    def close(self):
        """Clean up resources."""
        #logger.info("Closing environment...")
        self.gazebo.close()
        self._close_mavlink_connection()


if __name__ == "__main__":
    from gymnasium.utils.env_checker import check_env, env_reset_passive_checker, env_step_passive_checker, check_step_determinism  
    from rl_training.utils.utils import load_config
    config = load_config('/home/pid_rl/rl_training/configs/default_config.yaml')

    env = HardEnv(config)
    env.close()