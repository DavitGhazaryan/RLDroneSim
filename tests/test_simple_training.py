#!/usr/bin/env python3

from rl_training.environments import ArdupilotEnv
from rl_training.utils.utils import load_config

def main():
    config = load_config('/home/pid_rl/rl_training/configs/default_config.yaml')
    
    env = ArdupilotEnv(config)

    # agent = PPOAgent(config)
    # agent.setup(env)
    
    total_episodes = 5
    max_steps_per_episode = 5
    
    for episode in range(total_episodes):
        obs, info = env.reset()
        episode_reward = 0.0
        print(f"Episode {episode}: Reward = {episode_reward:.2f}")
        for step in range(max_steps_per_episode):
            # Get action from agent
            # action = agent.predict(obs, deterministic=False)
            # action = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            action = env.action_space.sample()

            # Take step in environment
            next_obs, reward, done, truncated, info = env.step(action)
            
            # Update episode statistics
            episode_reward += reward
            obs = next_obs
            
            if done or truncated:
                print(f"Episode {episode}: Done = {done}, Truncated = {truncated}")
                break
        
        # Log episode results
        print(f"Episode {episode}: Reward = {episode_reward:.2f}")
        
        # Save checkpoint every 100 episodes
        # if episode % 100 == 0:
            # agent.save(f"./checkpoint_episode_{episode}")
    
    env.close()

if __name__ == "__main__":
    main()