"""
Action space wrappers for preprocessing actions.
"""

import numpy as np
import gymnasium as gym
from gymnasium import ActionWrapper


class ActionNormalizer(ActionWrapper):
    """Normalize actions to a specific range."""
    
    def __init__(self, env, target_min=-1, target_max=1):
        super().__init__(env)
        self.target_min = target_min
        self.target_max = target_max
    
    def action(self, action):
        """Normalize the action."""
        # Assuming action space is Box
        if hasattr(self.action_space, 'low') and hasattr(self.action_space, 'high'):
            action_min = self.action_space.low
            action_max = self.action_space.high
            
            # Normalize to target range
            normalized = (action - action_min) / (action_max - action_min)
            normalized = normalized * (self.target_max - self.target_min) + self.target_min
            
            return normalized
        return action


class ActionClipper(ActionWrapper):
    """Clip actions to valid range."""
    
    def __init__(self, env, clip_min=None, clip_max=None):
        super().__init__(env)
        self.clip_min = clip_min
        self.clip_max = clip_max
    
    def action(self, action):
        """Clip the action."""
        if self.clip_min is not None or self.clip_max is not None:
            return np.clip(action, self.clip_min, self.clip_max)
        return action 