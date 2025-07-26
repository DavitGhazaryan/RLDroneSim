"""
Observation space wrappers for preprocessing observations.
"""

import numpy as np
import gymnasium as gym
from gymnasium import ObservationWrapper


class ObservationNormalizer(ObservationWrapper):
    """Normalize observations to zero mean and unit variance."""
    
    def __init__(self, env, mean=None, std=None):
        super().__init__(env)
        self.mean = mean
        self.std = std
    
    def observation(self, obs):
        """Normalize the observation."""
        if self.mean is not None and self.std is not None:
            return (obs - self.mean) / self.std
        return obs


class ObservationFilter(ObservationWrapper):
    """Apply filtering to observations (e.g., moving average)."""
    
    def __init__(self, env, filter_type="moving_average", window_size=5):
        super().__init__(env)
        self.filter_type = filter_type
        self.window_size = window_size
        self.observation_history = []
    
    def observation(self, obs):
        """Apply filtering to the observation."""
        self.observation_history.append(obs)
        if len(self.observation_history) > self.window_size:
            self.observation_history.pop(0)
        
        if self.filter_type == "moving_average":
            return np.mean(self.observation_history, axis=0)
        return obs 