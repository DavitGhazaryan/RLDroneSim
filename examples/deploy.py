#!/usr/bin/env python3
"""
Deploy a trained DDPG agent on ArdupilotEnv.

Usage:
  python deploy.py --model_path /path/to/ddpg_ardupilot_final.zip
                   --config /home/pid_rl/rl_training/configs/default_config.yaml
                   --instance 1
                   --episodes 5
                   --deterministic
"""

import numpy as np
import sys
import argparse

# project paths
sys.path.insert(0, "/home/pid_rl")

from rl_training.environments import HardEnv
from rl_training.utils.utils import load_config, evaluate_agent
from stable_baselines3 import DDPG
from stable_baselines3.common.monitor import Monitor




def run_episodes(model: DDPG, env, n_episodes: int, deterministic: bool, max_steps: int = None):
    """
    Manual rollout loop (useful when you want step-by-step control).
    Returns list of dicts: {"episode": i, "return": R, "length": L}.
    """
    results = []
    for ep in range(n_episodes):
        obs, info = env.reset()
        done, truncated = False, False
        ep_ret, ep_len = 0.0, 0
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done, truncated, info = env.step(action)
            ep_ret += float(reward)
            ep_len += 1
            if max_steps and ep_len >= max_steps:
                break
        results.append({"episode": ep + 1, "return": ep_ret, "length": ep_len})
        print(f"[Episode {ep+1}] return={ep_ret:.3f} length={ep_len}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to saved SB3 model (.zip).")
    parser.add_argument("--config", type=str, default="/home/pid_rl/rl_training/configs/default_config.yaml")
    parser.add_argument("--instance", type=int, default=1, choices=[1, 2])
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--deterministic", action="store_true", help="Use greedy policy for deployment.")
    parser.add_argument("--eval_mode", action="store_true",
                        help="Use utils.evaluate_agent (fast) instead of manual rollouts.")
    parser.add_argument("--max_steps", type=int, default=None, help="Cap steps per episode (optional).")
    parser.add_argument("--device", type=str, default="auto", help="SB3 device: auto/cpu/cuda")
    parser.add_argument("--save_metrics", type=str, default=None, help="Optional JSON path for results.")
    args = parser.parse_args()

    # 1) Load cfg & env
    cfg = load_config(args.config)
    print("🔧 Building ArdupilotEnv...")
    env = HardEnv(cfg, eval=True, instance=args.instance)

    episode = 0
    episode_rewards = []
    episode_lengths = []
    try:
        while True:
            obs, info = env.reset()
            episode_reward = 0.0
            episode_length = 0
            
            print(f"   Episode {episode + 1}: ")
            
            while True:

                action = env.action_space.sample()  # will be 0 0 0 0 action 
                obs, reward, terminated, truncated, info = env.step(action)
                
                episode_reward += reward
                episode_length += 1
                
                if terminated or truncated:
                    break
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            print(f"Reward: {episode_reward:.2f}, Length: {episode_length}")
    finally:
        # Calculate statistics
        avg_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        avg_length = np.mean(episode_lengths)
        print("\n📊 Evaluation Results:")
        print(f"   Average reward: {avg_reward:.2f} ± {std_reward:.2f}")
        print(f"   Average episode length: {avg_length:.1f} steps")
        print(f"   Success rate: {sum(1 for r in episode_rewards if r > 0) / len(episode_rewards):.1%}")
        
        # 5) Cleanup
        env.close()
        print("🧹 Environment closed. ✅")




if __name__ == "__main__":
    main()
