#!/usr/bin/env python3

"""
Evaluate the Default PID gains as baseline, using the same reward function and pipeline as the other agents.
"""

import sys
sys.path.insert(0, "/home/pid_rl")

from rl_training.environments import SimGymEnv
from rl_training.utils.utils import load_config, validate_config, evaluate_agent

from stable_baselines3.common.monitor import Monitor

    

def main():
    """Main function demonstrating the complete workflow."""
    
    
    config_path = '/home/pid_rl/rl_training/configs/default_config.yaml'
    
    try:

        config = load_config(config_path)        
        print("🔧 Creating  ArdupilotEnv...")
        
        env = SimGymEnv(config)
        
        env = Monitor(env)

        results = evaluate_agent(model=None, env=env, num_episodes=25)
                
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up
        if 'env' in locals():
            env.close()
        print("\n🧹 Environment closed.")


if __name__ == "__main__":
    main() 