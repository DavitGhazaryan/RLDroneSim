"""
Reward function implementations for Ardupilot RL training.
TO DO
"""

import numpy as np
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod


class RewardFunction(ABC):
    """
    Abstract base class for reward functions.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    @abstractmethod
    def compute_reward(self, 
                      current_state: Dict[str, Any], 
                      action: np.ndarray, 
                      next_state: Dict[str, Any],
                      done: bool) -> float:
        """Compute reward for the given transition."""
        pass


class HoverReward(RewardFunction):
    """
    Reward function for hover control task.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.target_position = config.get('target_position', [0.0, 0.0, 10.0])
        self.position_weight = config.get('position_weight', 1.0)
        self.velocity_weight = config.get('velocity_weight', 0.1)
        self.attitude_weight = config.get('attitude_weight', 0.5)
        self.action_weight = config.get('action_weight', 0.01)
        self.crash_penalty = config.get('crash_penalty', -100.0)
    
    def compute_reward(self, 
                      current_state: Dict[str, Any], 
                      action: np.ndarray, 
                      next_state: Dict[str, Any],
                      done: bool) -> float:
        """Compute hover reward."""
        if done:
            return self.crash_penalty
        
        # Position error
        current_pos = current_state.get('position', [0.0, 0.0, 0.0])
        pos_error = np.linalg.norm(np.array(current_pos) - np.array(self.target_position))
        position_reward = -self.position_weight * pos_error
        
        # Velocity penalty
        current_vel = current_state.get('velocity', [0.0, 0.0, 0.0])
        vel_magnitude = np.linalg.norm(current_vel)
        velocity_reward = -self.velocity_weight * vel_magnitude
        
        # Attitude penalty
        current_att = current_state.get('attitude', [0.0, 0.0, 0.0])
        att_magnitude = np.linalg.norm(current_att)
        attitude_reward = -self.attitude_weight * att_magnitude
        
        # Action penalty
        action_magnitude = np.linalg.norm(action)
        action_reward = -self.action_weight * action_magnitude
        
        return position_reward + velocity_reward + attitude_reward + action_reward


class WaypointReward(RewardFunction):
    """
    Reward function for waypoint navigation task.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.waypoints = config.get('waypoints', [])
        self.current_waypoint_idx = 0
        self.position_weight = config.get('position_weight', 1.0)
        self.velocity_weight = config.get('velocity_weight', 0.1)
        self.waypoint_reached_reward = config.get('waypoint_reached_reward', 100.0)
        self.crash_penalty = config.get('crash_penalty', -100.0)
        self.waypoint_threshold = config.get('waypoint_threshold', 5.0)
    
    def compute_reward(self, 
                      current_state: Dict[str, Any], 
                      action: np.ndarray, 
                      next_state: Dict[str, Any],
                      done: bool) -> float:
        """Compute waypoint navigation reward."""
        if done:
            return self.crash_penalty
        
        current_pos = current_state.get('position', [0.0, 0.0, 0.0])
        
        # Check if waypoint reached
        if self.current_waypoint_idx < len(self.waypoints):
            target_waypoint = self.waypoints[self.current_waypoint_idx]
            distance = np.linalg.norm(np.array(current_pos) - np.array(target_waypoint))
            
            if distance < self.waypoint_threshold:
                self.current_waypoint_idx += 1
                return self.waypoint_reached_reward
        
        # Distance to current waypoint
        if self.current_waypoint_idx < len(self.waypoints):
            target_waypoint = self.waypoints[self.current_waypoint_idx]
            distance = np.linalg.norm(np.array(current_pos) - np.array(target_waypoint))
            return -self.position_weight * distance
        
        return 0.0


class PIDTuningReward(RewardFunction):
    """
    Reward function for PID tuning task.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.setpoint = config.get('setpoint', [0.0, 0.0, 10.0, 0.0])  # x, y, z, yaw
        self.error_weight = config.get('error_weight', 1.0)
        self.overshoot_penalty = config.get('overshoot_penalty', 0.5)
        self.settling_time_weight = config.get('settling_time_weight', 0.1)
        self.crash_penalty = config.get('crash_penalty', -100.0)
        self.settling_threshold = config.get('settling_threshold', 0.1)
        self.step_count = 0
    
    def compute_reward(self, 
                      current_state: Dict[str, Any], 
                      action: np.ndarray, 
                      next_state: Dict[str, Any],
                      done: bool) -> float:
        """Compute PID tuning reward."""
        if done:
            return self.crash_penalty
        
        self.step_count += 1
        
        current_pos = current_state.get('position', [0.0, 0.0, 0.0])
        current_att = current_state.get('attitude', [0.0, 0.0, 0.0])
        
        # Error from setpoint
        error = np.array(current_pos + [current_att[2]]) - np.array(self.setpoint)
        error_magnitude = np.linalg.norm(error)
        error_reward = -self.error_weight * error_magnitude
        
        # Overshoot penalty
        overshoot = np.maximum(0, error)
        overshoot_penalty = -self.overshoot_penalty * np.sum(overshoot)
        
        # Settling time reward (reward for staying within threshold)
        if error_magnitude < self.settling_threshold:
            settling_reward = self.settling_time_weight
        else:
            settling_reward = 0.0
        
        return error_reward + overshoot_penalty + settling_reward


class CustomReward(RewardFunction):
    """
    Custom reward function with configurable components.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.components = config.get('components', {})
    
    def compute_reward(self, 
                      current_state: Dict[str, Any], 
                      action: np.ndarray, 
                      next_state: Dict[str, Any],
                      done: bool) -> float:
        """Compute custom reward based on configured components."""
        total_reward = 0.0
        
        for component_name, component_config in self.components.items():
            weight = component_config.get('weight', 1.0)
            
            if component_name == 'position_error':
                target = component_config.get('target', [0.0, 0.0, 0.0])
                current_pos = current_state.get('position', [0.0, 0.0, 0.0])
                error = np.linalg.norm(np.array(current_pos) - np.array(target))
                total_reward += -weight * error
            
            elif component_name == 'velocity_penalty':
                current_vel = current_state.get('velocity', [0.0, 0.0, 0.0])
                vel_magnitude = np.linalg.norm(current_vel)
                total_reward += -weight * vel_magnitude
            
            elif component_name == 'action_penalty':
                action_magnitude = np.linalg.norm(action)
                total_reward += -weight * action_magnitude
        
        if done:
            crash_penalty = self.config.get('crash_penalty', -100.0)
            total_reward += crash_penalty
        
        return total_reward 