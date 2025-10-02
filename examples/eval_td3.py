#!/usr/bin/env python3

import sys
sys.path.insert(0, "/home/pid_rl")

from rl_training.environments import BaseEnv
from rl_training.utils.utils import load_config, evaluate_agent

from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import TD3
    

def main():
    """Main function demonstrating the complete workflow."""
    
    
    config_path = '/home/pid_rl/rl_training/configs/default_config.yaml'
    model_zip = "/home/pid_rl/rl_training/runs/ddpg/hover/20250930_175629/models/td3_ardupilot_43000_steps.zip"
    

    config = load_config(config_path)
    
    print("🔧 Creating  ArdupilotEnv...")
    
    env = BaseEnv(config, hardware=False)        
    env = Monitor(env)
    model = TD3.load(model_zip, env=env, device=config.get('td3_params').get('device'))
    n_eval = 50
    results = evaluate_agent(model, env, n_eval)



if __name__ == "__main__":
    main() 