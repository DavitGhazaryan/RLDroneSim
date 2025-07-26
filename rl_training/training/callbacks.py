"""
Training callbacks for Ardupilot RL training.
"""

import time
import numpy as np
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod


class TrainingCallback(ABC):
    """
    Abstract base class for training callbacks.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    @abstractmethod
    def on_step(self, timesteps: int, episode_reward: float, episode_steps: int):
        """Called after each step."""
        pass
    
    @abstractmethod
    def on_episode_end(self, episode_info: Dict[str, Any]):
        """Called at the end of each episode."""
        pass


class LoggingCallback(TrainingCallback):
    """
    Callback for logging training progress.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.log_freq = config.get('log_freq', 1000)
        self.last_log_time = time.time()
        self.episode_rewards = []
        self.episode_lengths = []
    
    def on_step(self, timesteps: int, episode_reward: float, episode_steps: int):
        """Log step information."""
        current_time = time.time()
        
        if current_time - self.last_log_time >= self.log_freq:
            print(f"Timesteps: {timesteps}, Episode Reward: {episode_reward:.2f}, Steps: {episode_steps}")
            self.last_log_time = current_time
    
    def on_episode_end(self, episode_info: Dict[str, Any]):
        """Log episode information."""
        self.episode_rewards.append(episode_info['reward'])
        self.episode_lengths.append(episode_info['steps'])
        
        # Log episode statistics
        if len(self.episode_rewards) % 10 == 0:
            mean_reward = np.mean(self.episode_rewards[-10:])
            mean_length = np.mean(self.episode_lengths[-10:])
            print(f"Episode {episode_info['episode']}: Mean Reward: {mean_reward:.2f}, Mean Length: {mean_length:.1f}")


class EvaluationCallback(TrainingCallback):
    """
    Callback for periodic evaluation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.eval_freq = config.get('eval_freq', 10000)
        self.n_eval_episodes = config.get('n_eval_episodes', 10)
        self.last_eval_timesteps = 0
    
    def on_step(self, timesteps: int, episode_reward: float, episode_steps: int):
        """Check if evaluation is needed."""
        if timesteps - self.last_eval_timesteps >= self.eval_freq:
            self._run_evaluation(timesteps)
            self.last_eval_timesteps = timesteps
    
    def on_episode_end(self, episode_info: Dict[str, Any]):
        """Handle episode end."""
        pass
    
    def _run_evaluation(self, timesteps: int):
        """Run evaluation episodes."""
        print(f"Running evaluation at {timesteps} timesteps...")
        # This would typically run evaluation episodes
        # and log the results


class TensorboardCallback(TrainingCallback):
    """
    Callback for TensorBoard logging.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.log_dir = config.get('log_dir', './logs')
        self.tensorboard_writer = None
        self._setup_tensorboard()
    
    def _setup_tensorboard(self):
        """Setup TensorBoard writer."""
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.tensorboard_writer = SummaryWriter(self.log_dir)
        except ImportError:
            print("TensorBoard not available. Install torch to use TensorBoard logging.")
    
    def on_step(self, timesteps: int, episode_reward: float, episode_steps: int):
        """Log step information to TensorBoard."""
        if self.tensorboard_writer is not None:
            self.tensorboard_writer.add_scalar('Training/EpisodeReward', episode_reward, timesteps)
            self.tensorboard_writer.add_scalar('Training/EpisodeSteps', episode_steps, timesteps)
    
    def on_episode_end(self, episode_info: Dict[str, Any]):
        """Log episode information to TensorBoard."""
        if self.tensorboard_writer is not None:
            self.tensorboard_writer.add_scalar('Episode/Reward', episode_info['reward'], episode_info['episode'])
            self.tensorboard_writer.add_scalar('Episode/Steps', episode_info['steps'], episode_info['episode'])
            self.tensorboard_writer.add_scalar('Episode/Timesteps', episode_info['timesteps'], episode_info['episode'])


class CheckpointCallback(TrainingCallback):
    """
    Callback for saving checkpoints.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.save_freq = config.get('save_freq', 50000)
        self.save_path = config.get('save_path', './checkpoints')
        self.last_save_timesteps = 0
    
    def on_step(self, timesteps: int, episode_reward: float, episode_steps: int):
        """Check if checkpoint should be saved."""
        if timesteps - self.last_save_timesteps >= self.save_freq:
            self._save_checkpoint(timesteps)
            self.last_save_timesteps = timesteps
    
    def on_episode_end(self, episode_info: Dict[str, Any]):
        """Handle episode end."""
        pass
    
    def _save_checkpoint(self, timesteps: int):
        """Save training checkpoint."""
        print(f"Saving checkpoint at {timesteps} timesteps...")
        # This would typically save the model and training state 