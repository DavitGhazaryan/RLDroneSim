#!/usr/bin/env python3
"""
Basic training example for Ardupilot RL.

This example demonstrates how to set up and run basic training
with the Ardupilot environment and PPO agent.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from rl_training.environments import ArdupilotEnv
# from rl_training.agents import PPOAgent
from rl_training.utils.utils import load_config

def main():
    """Run basic training example."""
    print("Starting basic Ardupilot RL training...")
    
    # Load configuration
    config = load_config('/home/pid_rl/rl_training/configs/default_config.yaml')
    
    # Create environment
    print("Creating Ardupilot environment...")
    env = ArdupilotEnv(config)
    
    # Create agent
    print("Creating PPO agent...")
    agent = PPOAgent(config)
    
    # Create trainer
    print("Setting up trainer...")
    trainer = Trainer(config.get('training_config', {}))
    trainer.setup(agent, env)
    
    # Start training
    print("Starting training...")
    total_timesteps = config.get('training_config', {}).get('total_timesteps', 100000)
    trainer.train(total_timesteps=total_timesteps)
    
    # Save trained model
    print("Saving trained model...")
    trainer.save_checkpoint("./trained_model")
    
    print("Training completed!")


if __name__ == "__main__":
    main() 