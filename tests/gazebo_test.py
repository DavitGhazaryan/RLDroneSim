#!/usr/bin/env python3
"""
Gazebo integration test example.

This example demonstrates how to:
1. Create an ArdupilotEnv with Gazebo
2. Run a simple episode
3. Monitor Gazebo status
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import time
import numpy as np
from rl_training.utils.config import Config
from rl_training.environments import ArdupilotEnv


def main():
    """Run a simple Gazebo integration test."""
    print("🚁 Ardupilot RL - Gazebo Integration Test")
    print("=" * 50)
    
    # Create configuration
    config = Config()
    
    # Configure for testing (headless mode for automation)
    config.set('gazebo_config', {
        'sdf_file': None,        # Use default world
        'gui': False,            # Set to True if you want to see the GUI
        'headless': True,        # Run in headless mode
        'real_time_factor': 2.0, # Run simulation faster
        'timeout': 20.0,         # Allow more time for startup
        'extra_args': []
    })
    
    config.set('environment_config', {
        'max_episode_steps': 50  # Short episodes for testing
    })
    
    try:
        print("Creating ArdupilotEnv...")
        env = ArdupilotEnv(config)
        
        print("Environment created successfully!")
        print(f"Observation space: {env.observation_space}")
        print(f"Action space: {env.action_space}")
        
        # Get initial Gazebo status
        gazebo_info = env.get_gazebo_info()
        print(f"Gazebo status: {gazebo_info}")
        
        # Run a simple episode
        print("\n🎮 Running test episode...")
        print("-" * 30)
        
        obs, info = env.reset()
        print(f"Episode started - Initial obs: {obs}")
        print(f"Reset info: {info}")
        
        episode_reward = 0.0
        step_count = 0
        
        for step in range(10):  # Run 10 steps
            # Sample random action
            action = env.action_space.sample()
            
            # Take step
            obs, reward, done, truncated, info = env.step(action)
            
            episode_reward += reward
            step_count += 1
            
            print(f"Step {step + 1}: Action={action}, Reward={reward:.3f}, Done={done}")
            
            if done or truncated:
                print(f"Episode ended: Done={done}, Truncated={truncated}")
                break
            
            # Small delay to see the simulation
            time.sleep(0.1)
        
        print(f"\n📊 Episode Results:")
        print(f"Total steps: {step_count}")
        print(f"Total reward: {episode_reward:.3f}")
        print(f"Final info: {info}")
        
        # Test environment rendering
        print("\n🖥️  Testing rendering...")
        env.render()
        
        # Final Gazebo status
        final_gazebo_info = env.get_gazebo_info()
        print(f"\nFinal Gazebo status: {final_gazebo_info}")
        
        # Test reset again
        print("\n🔄 Testing environment reset...")
        obs, info = env.reset()
        print(f"Reset successful - New obs: {obs}")
        
        print("\n✅ Test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        print("\n🧹 Cleaning up...")
        if 'env' in locals():
            env.close()
        print("Cleanup complete.")


if __name__ == "__main__":
    main() 