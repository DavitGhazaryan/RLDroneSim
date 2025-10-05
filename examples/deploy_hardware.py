import numpy as np
import logging
import sys
import time
import os 
import torch
import yaml

raise NotImplemented("The correct version is on Raspberry...")

from rl_training.environments.base_env import BaseEnv

sys.path.insert(0, "/home/student/Dev/pid_rl")

logger = logging.getLogger("Env")
from enum import Enum, auto

class Termination(Enum):    
    ATTITUDE_ERR = auto()   # excessive attitude error
    VEL_EXC = auto()        # velocity exceeded
    FLIP = auto()           # flip detected
    FAR = auto()            # too far from target

    
def deploy(model_path):
    """Static method to load the model, read the config, and set up the deployment loop."""
    # 1. Load model
    model_file = os.path.join(model_path, "policy_actor_scripted.pt")
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model file not found at {model_file}")
    actor = torch.jit.load(model_file, map_location="cpu").eval()
    
    # 2. Load the environment configuration from cfg.yaml
    config_file = os.path.join(model_path, "cfg.yaml")
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file not found at {config_file}")
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    
    # 3. Initialize the deployment environment
    env = BaseEnv(config)
    
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
            # 4.4 Send action to the drone
            print("step")
            env.step(action)

            # time.sleep(0.1)
    except KeyboardInterrupt:
        print("Deployment stopped manually.")
    finally:
        env.close()


if __name__ == "__main__":
    from rl_training.utils.utils import load_config
    config = load_config('/home/pid_rl/rl_training/configs/default_config.yaml')

    deploy("/home/student/Dev/pid_rl/models/ddpg_model")
