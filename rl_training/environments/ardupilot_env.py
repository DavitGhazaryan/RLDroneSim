"""
Main Ardupilot environment implementing Gymnasium API.
"""

import os
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import argparse
import logging
import sys
import asyncio

sys.path.insert(0, "/home/student/Dev/pid_rl")

from rl_training.utils.gazebo_interface import GazeboInterface
from rl_training.utils.ardupilot_sitl import ArduPilotSITL

from rl_training.utils.utils import load_config

logger = logging.getLogger("Env")
logger.setLevel(logging.INFO)

class ArdupilotEnv(gym.Env):
    """
    Ardupilot environment for RL training.
    
    This environment provides a Gymnasium-compatible interface for
    training RL agents on Ardupilot SITL with Gazebo simulation.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.gazebo = GazeboInterface(self.config['gazebo_config'])
        self.sitl = ArduPilotSITL(self.config['ardupilot_config'])

        self.loop = asyncio.get_event_loop()

        self.old_observation = None

        # Episode tracking
        self.initialized = False
        self.episode_step = 0
        self.environment_config = config.get('environment_config', {})
        self.max_episode_steps = self.environment_config.get('max_episode_steps', 100)

        # Initialize spaces
        self.observation_space = self._define_observation_space()
        self.action_space = self._define_action_space()
            
    def _define_observation_space(self):
        ## TODO: Define the observation space based on the environment_config
        low = np.full(6, -np.inf, dtype=np.float32)
        high = np.full(6, np.inf, dtype=np.float32)
        return spaces.Box(low=low, high=high, dtype=np.float32)
    
    def _define_action_space(self):
        ## TODO: Define the action space based on the environment_config
        # Each action is a delta for a PID gain, bounded between -0.1 and 0.1
        low = np.array([-0.1] * 6, dtype=np.float32)
        high = np.array([0.1] * 6, dtype=np.float32)
        return spaces.Box(low=low, high=high, dtype=np.float32)
    
    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        # TODO reset the world  using the reset interfaces functions

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
        
            self.old_observation, info = self.loop.run_until_complete(self._async_reset())
            return self.old_observation, info  # observation, info
        else:
            raise NotImplementedError("Resetting the environment SECOND TIME is not implemented for this environment.")

    async def _async_reset(self):
        observation = await self.sitl.get_pid_params_async()
        
        return self.dict_to_obs(observation), {}  # observation, info

    def step(self, action):
        self.episode_step += 1
        obs, reward, done, truncated, info = self.loop.run_until_complete(self._async_step(action))
        self.old_observation = obs
        return obs, reward, done, truncated, info

    async def _async_step(self, action):
        pid_params = self.old_observation + action
        await self.sitl.set_params_async(pid_params)
        
        obs = await self.sitl.get_pid_params_async()
        
        reward = await self._compute_reward(obs, pid_params)
        done = False
        truncated = self.episode_step >= self.max_episode_steps
        info = {}

        return self.dict_to_obs(obs), reward, done, truncated, info

    async def _compute_reward(self, obs, pid_params):
        print(f"Calculating reward for")
        await asyncio.sleep(20)
        return 0.0

    def close(self):
        """Clean up resources."""
        self.gazebo.close()
        self.sitl.close()

    def render(self, mode="human"):
        """Render the environment."""
        pass 

    def dict_to_obs(self, pid_dict):
        PID_KEYS = [
            "ATC_ANG_PIT_P",
            "ATC_ANG_RLL_P",
            "ATC_ANG_YAW_P",
            "ATC_RAT_PIT_P",
            "ATC_RAT_RLL_P",
            "ATC_RAT_YAW_P",
        ]
        return np.array([pid_dict[k] for k in PID_KEYS], dtype=np.float32)


if __name__ == "__main__":
    config = load_config('/home/student/Dev/pid_rl/rl_training/configs/default_config.yaml')
    env = ArdupilotEnv(config)

    env.reset()
    env.step(np.array([0.3, 0.3, 0.3, 0.3, 0.3, 0.3]))
    env.step(np.array([0.0, 0.2, 0.2, 0.2, 0.2, 0.2]))

    env.close()