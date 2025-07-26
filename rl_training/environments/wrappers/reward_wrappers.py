"""
Reward wrappers for reward shaping and normalization.
"""

import numpy as np
import gymnasium as gym
from gymnasium import RewardWrapper


class RewardShaping(RewardWrapper):
    """Apply reward shaping to encourage desired behaviors."""
    
    def __init__(self, env, shaping_weights=None):
        super().__init__(env)
        self.shaping_weights = shaping_weights or {}
    
    def reward(self, reward):
        """Apply reward shaping."""
        shaped_reward = reward
        
        # Add shaping terms based on current state
        # This is a placeholder - implement based on your specific needs
        
        return shaped_reward


class RewardNormalizer(RewardWrapper):
    """Normalize rewards using running statistics."""
    
    def __init__(self, env, gamma=0.99, epsilon=1e-8):
        super().__init__(env)
        self.gamma = gamma
        self.epsilon = epsilon
        self.ret = 0.0
        self.ret_rms = None
    
    def reward(self, reward):
        """Normalize the reward."""
        self.ret = self.ret * self.gamma + reward
        
        # Initialize running mean and std if not done
        if self.ret_rms is None:
            self.ret_rms = RunningMeanStd()
        
        self.ret_rms.update(self.ret)
        
        # Normalize reward
        normalized_reward = reward / np.sqrt(self.ret_rms.var + self.epsilon)
        
        return normalized_reward


class RunningMeanStd:
    """Running mean and standard deviation calculator."""
    
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon
    
    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = len(x)
        
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta ** 2 * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        
        self.mean = new_mean
        self.var = new_var
        self.count = tot_count 