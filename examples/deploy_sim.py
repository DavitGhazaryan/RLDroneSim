#!/usr/bin/env python3

"""
File is intended to check the SEPARATED ACTOR NETWORK in the simulation environment.
Uses the same Gym Compatible Environment as the was used during the training.
"""

import sys
sys.path.insert(0, "/home/pid_rl")
import torch 
import os
import yaml

from rl_training.environments import SimGymEnv

    
def deploy(model_path):

    # 1. Load actor Network only
    model_file = os.path.join(model_path, "policy_actor_scripted.pt")
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model file not found at {model_file}")
    actor = torch.jit.load(model_file, map_location="cpu").eval()

    # 2. Load the environment configuration from cfg.yaml
    config_file = os.path.join(model_path, "cfg.yaml")
    print(config_file)
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file not found at {config_file}")
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    # 3. Initialize the deployment environment
    env = SimGymEnv(config)
    env.reset()
    # 4. Start the deployment loop
    try:
        while True:
            # 4.1 Get observation (current state of the drone)
            current_state = env._get_observation(env.ep_initial_gains)
            # 4.2 Prepare the observation for the model
            observation = torch.tensor(current_state[0], dtype=torch.float32).unsqueeze(0)  # Add batch dimension
            # 4.3 Get the action from the actor (policy)
            with torch.no_grad():
                action = actor(observation).squeeze(0).cpu().numpy()
            # 4.4 Send action to the drone, observe result
            obs, reward, terminated, truncated, info = env.step(action)

    except KeyboardInterrupt:
        print("Deployment stopped manually.")
    finally:
        env.close()


if __name__ == "__main__":
    # main() 
    deploy("/home/pid_rl/rl_training/runs/ddpg/hover/20251005_040415")