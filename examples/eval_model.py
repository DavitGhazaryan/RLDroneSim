#!/usr/bin/env python3
import sys
sys.path.insert(0, "/home/pid_rl")

from rl_training.environments import SimGymEnv
from rl_training.utils.utils import load_config, evaluate_agent

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize # pyright: ignore[reportMissingImports]
from stable_baselines3.common.monitor import Monitor # pyright: ignore[reportMissingImports]

def main():
    algo = "baseline"   # baseline, td3, ddpg
    training_number = '20251013_193506'
    checkpoint_step = "995000"
    
    # modify params
    gui = True
    speedup = 10

    if algo != "baseline":
        config_path = f'/home/pid_rl/rl_training/runs/ddpg/hover/{training_number}/cfg.yaml'
        model_zip = f"/home/pid_rl/rl_training/runs/ddpg/hover/{training_number}/models/{algo}_ardupilot_{checkpoint_step}_steps.zip"
        vecnorm_path = f"/home/pid_rl/rl_training/runs/ddpg/hover/{training_number}/models/{algo}_ardupilot_vecnormalize_{checkpoint_step}_steps.pkl"
    else:
        config_path = f'/home/pid_rl/rl_training/configs/default_config.yaml'
        model_zip = None
        vecnorm_path = None
    
    config = load_config(config_path)
    
    config["sitl_config"]["speedup"] =speedup
    config["gazebo_config"]["gui"] = gui
    

    # create and wrap environment
    def make_env():
        env = SimGymEnv(config)
        return Monitor(env)
    env = DummyVecEnv([make_env])

    # load VecNormalize statistics
    env = VecNormalize.load(vecnorm_path, env) if vecnorm_path else env
    env.training = False          # disable further updates
    env.norm_reward = False       # don't normalize rewards during eval

    # load model with attached env
    if algo == "td3":
        from stable_baselines3 import TD3 # type: ignore
        model = TD3.load(model_zip, env=env, device=config[f'{algo}_params']['device'])
        gamma = config[f"{algo}_params"]["gamma"]
    elif algo == "ddpg":
        from stable_baselines3.ddpg import DDPG # pyright: ignore[reportMissingImports]
        model = DDPG.load(model_zip, env=env, device=config.get(f'{algo}_params').get('device'))
        gamma = config[f"{algo}_params"]["gamma"]
    elif algo == "baseline":
        model = None
        gamma  = 0.99
    n_eval = 5

    header = f" Evaluating : {algo}"
    if model:
        header += f": {training_number} : step {checkpoint_step} : {n_eval} episodes"
    
    print(header)

    results = evaluate_agent(model, env, n_eval, gamma=gamma)

if __name__ == "__main__":
    main()
