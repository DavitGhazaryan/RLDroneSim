#!/usr/bin/env python3

import sys
sys.path.insert(0, "/home/pid_rl")

from rl_training.environments import ArdupilotEnv
from rl_training.utils.utils import load_config
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor


def main():
    """Main function demonstrating the complete workflow."""

    config_path = '/home/pid_rl/rl_training/configs/default_config.yaml'
    config = load_config(config_path)

    print("🔧 Creating ArdupilotEnv...")

    make_env = lambda: ArdupilotEnv(config)

    # wrap with Monitor
    env = DummyVecEnv([lambda: Monitor(make_env())])

    # normalize ONLY observations
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    obs = env.reset()  # only obs
    for episode in range(50):
        done = False
        while not done:
            action = env.action_space.sample()[None, ...]  # add batch dim
            obs, reward, done, infos = env.step(action)
            print(f"Obs {obs}, Reward {reward}, Done {done}, Info {infos}")
            if done[0]:
                print(f"Terminal obs: {infos[0]['terminal_observation']}")



if __name__ == "__main__":
    main()
