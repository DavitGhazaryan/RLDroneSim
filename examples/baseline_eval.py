#!/usr/bin/env python3

"""
Example script showing how to use the modified ArdupilotEnv directly with Stable Baselines DDPG.
The environment now returns flattened observations and actions, making it directly compatible.
"""

import sys
sys.path.insert(0, "/home/pid_rl")

from rl_training.environments import ArdupilotEnv
from rl_training.utils.utils import load_config, validate_config, evaluate_agent

from stable_baselines3.common.monitor import Monitor

    

def main():
    """Main function demonstrating the complete workflow."""
    
    
    config_path = '/home/pid_rl/rl_training/configs/default_config.yaml'
    
    try:
        from rl_training.utils.utils import create_run_dir, save_config_copy, save_git_info, save_metrics_json
        import subprocess
        import os

        config = load_config(config_path)
        
        print("🔧 Creating  ArdupilotEnv...")
        
        env = ArdupilotEnv(config, eval=True)
        
        env = Monitor(env)

        results = evaluate_agent(model=None, env=env, num_episodes=5)
                
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