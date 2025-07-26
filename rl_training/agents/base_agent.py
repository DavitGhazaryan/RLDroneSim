"""
Abstract base agent for RL training.
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseAgent(ABC):
    """
    Abstract base class for RL agents.
    
    This class defines the interface that all RL agents
    must implement to be compatible with the training pipeline.
    """
    
    def __init__(self, config):
        self.config = config
        self.model = None
    
    @abstractmethod
    def setup(self, env):
        """Setup the agent with the environment."""
        pass
    
    @abstractmethod
    def predict(self, observation, deterministic=False):
        """Predict action for given observation."""
        pass
    
    @abstractmethod
    def train(self, env, total_timesteps, callback=None):
        """Train the agent."""
        pass
    
    @abstractmethod
    def save(self, path):
        """Save the trained model."""
        pass
    
    @abstractmethod
    def load(self, path):
        """Load a trained model."""
        pass
    
    @abstractmethod
    def evaluate(self, env, n_eval_episodes=10):
        """Evaluate the agent."""
        pass 