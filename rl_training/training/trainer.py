"""
Main trainer for Ardupilot RL training.
"""

import time
import numpy as np
from typing import Dict, Any, List, Optional

from ..agents import BaseAgent
from ..environments import ArdupilotEnv
from .callbacks import TrainingCallback


class Trainer:
    """
    Main trainer class for Ardupilot RL training.
    
    This class handles the training loop, callbacks, and logging.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize trainer.
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.agent = None
        self.env = None
        self.callbacks = []
        self.training_history = []
    
    def setup(self, agent: BaseAgent, env: ArdupilotEnv):
        """
        Setup trainer with agent and environment.
        
        Args:
            agent: RL agent to train
            env: Environment to train on
        """
        self.agent = agent
        self.env = env
        
        # Setup callbacks
        self._setup_callbacks()
    
    def train(self, total_timesteps: int, eval_freq: int = 10000):
        """
        Train the agent.
        
        Args:
            total_timesteps: Total number of timesteps to train for
            eval_freq: Frequency of evaluation episodes
        """
        if self.agent is None or self.env is None:
            raise RuntimeError("Trainer not setup. Call setup() first.")
        
        print(f"Starting training for {total_timesteps} timesteps...")
        
        # Setup agent with environment
        self.agent.setup(self.env)
        
        # Training loop
        timesteps_so_far = 0
        episode_count = 0
        
        while timesteps_so_far < total_timesteps:
            # Reset environment
            obs, info = self.env.reset()
            episode_reward = 0.0
            episode_steps = 0
            done = False
            truncated = False
            
            # Episode loop
            while not (done or truncated):
                # Get action from agent
                action = self.agent.predict(obs, deterministic=False)
                
                # Take step in environment
                next_obs, reward, done, truncated, info = self.env.step(action)
                
                # Update episode statistics
                episode_reward += reward
                episode_steps += 1
                timesteps_so_far += 1
                
                # Update observation
                obs = next_obs
                
                # Call callbacks
                for callback in self.callbacks:
                    callback.on_step(timesteps_so_far, episode_reward, episode_steps)
            
            # Episode finished
            episode_count += 1
            
            # Log episode statistics
            episode_info = {
                'episode': episode_count,
                'timesteps': timesteps_so_far,
                'reward': episode_reward,
                'steps': episode_steps,
                'done': done,
                'truncated': truncated
            }
            self.training_history.append(episode_info)
            
            # Call episode callbacks
            for callback in self.callbacks:
                callback.on_episode_end(episode_info)
            
            # Evaluation
            if episode_count % eval_freq == 0:
                self._evaluate()
        
        print("Training completed!")
    
    def _setup_callbacks(self):
        """Setup training callbacks."""
        callback_configs = self.config.get('callbacks', [])
        
        for callback_config in callback_configs:
            callback_type = callback_config.get('type')
            if callback_type == 'logging':
                from .callbacks import LoggingCallback
                callback = LoggingCallback(callback_config)
            elif callback_type == 'evaluation':
                from .callbacks import EvaluationCallback
                callback = EvaluationCallback(callback_config)
            else:
                raise ValueError(f"Unknown callback type: {callback_type}")
            
            self.callbacks.append(callback)
    
    def _evaluate(self):
        """Run evaluation."""
        if hasattr(self.agent, 'evaluate'):
            eval_results = self.agent.evaluate(self.env)
            print(f"Evaluation results: {eval_results}")
    
    def save_checkpoint(self, path: str):
        """Save training checkpoint."""
        if self.agent is not None:
            self.agent.save(path)
    
    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        if self.agent is not None:
            self.agent.load(path)
    
    def get_training_history(self) -> List[Dict[str, Any]]:
        """Get training history."""
        return self.training_history 