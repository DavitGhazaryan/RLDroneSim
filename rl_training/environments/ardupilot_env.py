"""
Main Ardupilot environment implementing Gymnasium API.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import argparse
import logging

from ..utils.gazebo_interface import GazeboInterface
from ..utils.ardupilot_sitl import ArduPilotSITL

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
        

        # Initialize spaces
        # self.observation_space = self._define_observation_space()
        # self.action_space = self._define_action_space()
        
        # Episode tracking
        # self.episode_step = 0
        # self.max_episode_steps = config.get('max_episode_steps', 1000)
    
        
    def _define_observation_space(self):
        """Define the observation space."""
        # Placeholder - will be implemented based on your needs
        pass
    
    def _define_action_space(self):
        """Define the action space."""
        # Placeholder - will be implemented based on your needs
        pass
    
    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        # TO DO reset the world  using the reset interfaces functions

        self.episode_step = 0
        
        logger.info("🌎 Launching Gazebo simulation...")
        self.gazebo.start_simulation()
        self.gazebo._wait_for_startup()
        self.gazebo.resume_simulation()
        logger.info("✅ Gazebo initialized")

        logger.info("🚁 Starting ArduPilot SITL...")
        self.sitl.start_sitl()
        info = self.sitl.get_process_info()
        logger.info(f"✅ SITL running (PID {info['pid']})")

        return None, {}  # Placeholder
    
    def step(self, action):
        """Execute action and return next state, reward, done, truncated, info."""

        # TODO: Each step will be performing a mission under a particular
        #  PID gain set.

        # TODO: Implement action execution
        self.episode_step += 1
        
        # Placeholder return values
        observation = None
        reward = 0.0
        done = False
        truncated = self.episode_step >= self.max_episode_steps
        info = {}
        
        return observation, reward, done, truncated, info
    
    def close(self):
        """Clean up resources."""
        self.gazebo.close()
        self.sitl.close()

    def render(self, mode="human"):
        """Render the environment."""
        pass 