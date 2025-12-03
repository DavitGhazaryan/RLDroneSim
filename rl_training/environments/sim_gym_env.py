import gymnasium as gym
from gymnasium import spaces

import numpy as np
import logging
import sys
import time
from functools import partial

sys.path.insert(0, "/home/student/Dev/pid_rl")

from rl_training.environments.base_env import BaseEnv
from rl_training.utils.ardupilot_sitl import ArduPilotSITL
from rl_training.utils.utils import euler_to_quaternion
logger = logging.getLogger("Env")

class SimGymEnv(BaseEnv, gym.Env):
    """
    Defines the main API of the Environment. main component is the drone which can be either hardware or Gazebo+SITL.
    """
    
    def __init__(self, config, instance=1, eval_baseline=False):
        BaseEnv.__init__(self, config=config, instance=instance)
        gym.Env.__init__(self)

        self.drone = ArduPilotSITL(config['sitl_config'], config['gazebo_config'], instance, self.verbose)

        # Initialize spaces
        self.observation_space = self._define_observation_space()
        self.action_space = self._define_action_space()
        self.eval_baseline = eval_baseline

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
    
    # Overwritten
    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        gym.Env.reset(self, seed=seed)         
        start = time.time()

        # if hasattr(self.action_space, "seed"):
        #     self.action_space.seed(seed)
        # if hasattr(self.observation_space, "seed"):
        #     self.observation_space.seed(seed)
        self.episode_step = 0

        if not self.initialized:
            self._initialize_system()
    
            self.mission_function = self._setup_mission()   # same x, y, z
            self.mission_function()

        else:
            self.eps_stable_time = 0
            self.max_stable_time = 0

            self.ep_initial_pose, self.ep_initial_attitude, self.ep_initial_gains = self._get_random_initial_state()
            # print("Reseted pose")
            # print(self.ep_initial_pose)
            self.curr_gains = self.ep_initial_gains.copy()
            for gain in self.action_gains:
                self.drone.set_param_and_confirm(gain, self.ep_initial_gains[gain])
            self.drone.reset([self.ep_initial_pose["x_m"], self.ep_initial_pose["y_m"], self.ep_initial_pose["z_m"]], euler_to_quaternion(None))            
                        
        self.drone.wait(self.action_dt)   # no need to normalize the sleep time with speedup
        observation, info = self._get_observation(self.ep_initial_gains)

        # print(f" Reset duration {end - start}")
        return observation, info  # observation, info

    # for simulation related code
    def _setup_mission(self):
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
        if not self.eval_baseline:
            for gain in self.action_gains:
                initial_gains[gain] = max(self.first_initial_gains[gain] + self.np_random.uniform(-2.0, 2.0), 0)
        else:
            initial_gains = self.ep_initial_gains.copy()
        
        return {
            'x_m': self.goal_pose['x_m'],
            'y_m': self.goal_pose['y_m'],    
            'z_m': max(self.goal_pose['z_m']+ self.np_random.uniform(-0.01, 0.01), 1.3) 
        }, self.ep_initial_attitude, initial_gains
    
