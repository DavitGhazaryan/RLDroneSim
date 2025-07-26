"""
Ardupilot RL Training System

A comprehensive framework for training reinforcement learning agents
on Ardupilot SITL with Gazebo simulation.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .environments import ArdupilotEnv
from .agents import TD3Agent

__all__ = [
    "ArdupilotEnv",
    "PPOAgent", 
    "SACAgent",
    "TD3Agent",
    "Config"
] 