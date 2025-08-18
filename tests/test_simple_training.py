#!/usr/bin/env python3

from rl_training.environments import ArdupilotEnv
from rl_training.utils.utils import load_config
from gymnasium.wrappers import RecordEpisodeStatistics
import numpy as np
import wandb


def main():
    # wandb.init(project="pid_rl", name="test_simple_training")
    config = load_config('/home/pid_rl/rl_training/configs/default_config.yaml')
    
    env = ArdupilotEnv(config)
    print(env.observation_space.sample())
    print(env.reset())
    # agent = PPOAgent(config)
    # agent.setup(env)
    total_episodes = 20
    max_steps_per_episode = 30
    env = RecordEpisodeStatistics(env, buffer_length=total_episodes)
    
    for episode in range(total_episodes):
        obs, info = env.reset()
        print(f"Observation: {obs}")
        episode_reward = 0.0
        for step in range(max_steps_per_episode):
            # action = agent.predict(obs, deterministic=False)
            action = env.action_space.sample()

            next_obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            obs = next_obs
            print(f"Observation: {obs}")
            if terminated or truncated:
                print(f"Episode Finished: Info = {info}")
                break
        
        
        # Log episode results
        print()
        print(f"Episode {episode}: Reward = {episode_reward:.2f}")

        if "episode" in info:
            episode_data = info["episode"]
            # wandb.log({
            #     "episode": episode,
            #     "reward": episode_data['r'],
            #     "length": episode_data['l'],
            #     "episode_time": episode_data['t']
            # })
    print("Evaluation Summary")
    # print(f'Episode rewards: {list(env.return_queue)}')
    # print(f'Episode lengths: {list(env.length_queue)}')

        # Calculate some useful metrics
    avg_reward = np.sum(env.return_queue)
    avg_length = np.sum(env.length_queue)
    std_reward = np.std(env.return_queue)

    print(f'\nAverage reward: {avg_reward:.2f} ± {std_reward:.2f}')
    print(f'Average episode length: {avg_length:.1f} steps')
    print(f'Success rate: {sum(1 for r in env.return_queue if r > 0) / len(env.return_queue):.1%}')
    #     # Save checkpoint every 100 episodes
    #     # if episode % 100 == 0:
    #         # agent.save(f"./checkpoint_episode_{episode}")
    
    env.close()

if __name__ == "__main__":
    main()