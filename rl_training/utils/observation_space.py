"""
Observation space definitions for Ardupilot RL training.
TO DO
"""

import numpy as np
from gymnasium import spaces
from typing import Dict, Any, List


class ObservationSpace:
    """
    Observation space definitions for different drone states.
    """
    
    @staticmethod
    def position_velocity_attitude() -> spaces.Box:
        """
        Observation space for position, velocity, and attitude.
        
        Returns:
            Box space with 9 dimensions:
            - position (x, y, z): 3
            - velocity (vx, vy, vz): 3  
            - attitude (roll, pitch, yaw): 3
        """
        return spaces.Box(
            low=np.array([-1000, -1000, -100, -50, -50, -50, -180, -180, -180]),
            high=np.array([1000, 1000, 1000, 50, 50, 50, 180, 180, 180]),
            dtype=np.float32
        )
    
    @staticmethod
    def position_velocity_attitude_angular_velocity() -> spaces.Box:
        """
        Observation space for position, velocity, attitude, and angular velocity.
        
        Returns:
            Box space with 12 dimensions:
            - position (x, y, z): 3
            - velocity (vx, vy, vz): 3
            - attitude (roll, pitch, yaw): 3
            - angular velocity (roll_rate, pitch_rate, yaw_rate): 3
        """
        return spaces.Box(
            low=np.array([-1000, -1000, -100, -50, -50, -50, -180, -180, -180, -200, -200, -200]),
            high=np.array([1000, 1000, 1000, 50, 50, 50, 180, 180, 180, 200, 200, 200]),
            dtype=np.float32
        )
    
    @staticmethod
    def minimal_state() -> spaces.Box:
        """
        Minimal observation space with essential state information.
        
        Returns:
            Box space with 6 dimensions:
            - position (x, y, z): 3
            - attitude (roll, pitch, yaw): 3
        """
        return spaces.Box(
            low=np.array([-1000, -1000, -100, -180, -180, -180]),
            high=np.array([1000, 1000, 1000, 180, 180, 180]),
            dtype=np.float32
        )
    
    @staticmethod
    def custom_space(low: List[float], high: List[float]) -> spaces.Box:
        """
        Create custom observation space.
        
        Args:
            low: Lower bounds for each dimension
            high: Upper bounds for each dimension
            
        Returns:
            Box space with custom bounds
        """
        return spaces.Box(
            low=np.array(low, dtype=np.float32),
            high=np.array(high, dtype=np.float32),
            dtype=np.float32
        )
    
    @staticmethod
    def get_observation_vector(position: Dict[str, float], 
                             velocity: Dict[str, float], 
                             attitude: Dict[str, float]) -> np.ndarray:
        """
        Convert drone state dictionaries to observation vector.
        
        Args:
            position: Position dictionary with keys ['latitude', 'longitude', 'altitude']
            velocity: Velocity dictionary with keys ['north', 'east', 'down']
            attitude: Attitude dictionary with keys ['roll', 'pitch', 'yaw']
            
        Returns:
            Observation vector as numpy array
        """
        return np.array([
            position.get('latitude', 0.0),
            position.get('longitude', 0.0),
            position.get('altitude', 0.0),
            velocity.get('north', 0.0),
            velocity.get('east', 0.0),
            velocity.get('down', 0.0),
            attitude.get('roll', 0.0),
            attitude.get('pitch', 0.0),
            attitude.get('yaw', 0.0)
        ], dtype=np.float32) 