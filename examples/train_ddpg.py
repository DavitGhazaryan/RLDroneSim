#!/usr/bin/env python3

"""
Example script showing how to use the modified ArdupilotEnv directly with Stable Baselines DDPG.
The environment now returns flattened observations and actions, making it directly compatible.
"""

import sys
import os
sys.path.insert(0, "/home/pid_rl")

from rl_training.environments import ArdupilotEnv
from rl_training.utils.utils import load_config
import numpy as np


def create_environment(config_path: str):

    print("🔧 Creating  ArdupilotEnv...")
    config = load_config(config_path)
    
    # Create the environment (now directly compatible)
    env = ArdupilotEnv(config)


    # Show mapping information
    obs_mapping = env.get_observation_key_mapping()
    action_mapping = env.get_action_key_mapping()
    
    print(f"\n🗺️  Observation Mapping:")
    for key, idx in obs_mapping.items():
        print(f"   obs[{idx}] = {key}")
    
    print(f"\n🎯 Action Mapping:")
    for key, idx in action_mapping.items():
        print(f"   action[{idx}] = {key}")
    
    return env, config


def train_ddpg_agent(env, config, total_timesteps=1000):
    """
    Train a DDPG agent on the modified environment.
    
    Args:
        env: Modified ArdupilotEnv
        config: Configuration dictionary
        total_timesteps: Total training timesteps
        
    Returns:
        Trained DDPG model
    """
    from stable_baselines3 import DDPG
    
    from stable_baselines3.common.noise import NormalActionNoise
    from stable_baselines3.common.callbacks import CheckpointCallback

    print(f"\n🚀 Starting DDPG training for {total_timesteps} timesteps...")
    
    # Create action noise for exploration
    action_dim = env.action_space.shape[0]
    action_noise = NormalActionNoise(
        mean=np.zeros(action_dim),
        sigma=0.1 * np.ones(action_dim)
    )
    
    # Create the DDPG model
    model = DDPG(
        "MlpPolicy",
        env,
        action_noise=action_noise,
        learning_rate=1e-3,
        buffer_size=100000,
        learning_starts=100,
        batch_size=64,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        verbose=1,
        tensorboard_log="./logs/ddpg_ardupilot/"
    )
    
    # Create checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=500,
        save_path="./models/",
        name_prefix="ddpg_ardupilot"
    )
    
    # Train the model
    print("🎯 Training started...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback,
        progress_bar=True
    )
    
    print("✅ Training completed!")
    return model

def evaluate_agent(model, env, num_episodes=5):
    """
    Evaluate the trained agent.
    
    Args:
        model: Trained DDPG model
        env: Modified ArdupilotEnv
        num_episodes: Number of evaluation episodes
        
    Returns:
        Evaluation results
    """
    print(f"\n🧪 Evaluating agent over {num_episodes} episodes...")
    
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0.0
        episode_length = 0
        
        print(f"   Episode {episode + 1}: ", end="")
        
        while True:
            # Get action from the trained model
            action, _ = model.predict(obs, deterministic=True)
            
            # Take step in environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            
            if terminated or truncated:
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        print(f"Reward: {episode_reward:.2f}, Length: {episode_length}")
    
    # Calculate statistics
    avg_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    avg_length = np.mean(episode_lengths)
    
    print(f"\n📊 Evaluation Results:")
    print(f"   Average reward: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"   Average episode length: {avg_length:.1f} steps")
    print(f"   Success rate: {sum(1 for r in episode_rewards if r > 0) / len(episode_rewards):.1%}")
    
    return {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'avg_reward': avg_reward,
        'std_reward': std_reward,
        'avg_length': avg_length
    }


def demonstrate_observation_action_format(env):
    """Demonstrate the new observation and action format."""
    
    print(f"\n🔍 Demonstrating Observation and Action Format")
    print("=" * 60)
    
    # Show sample observations and actions
    sample_obs = env.observation_space.sample()
    sample_action = env.action_space.sample()
    
    print(f"📊 Sample Observation (Array):")
    print(f"   Shape: {sample_obs.shape}")
    print(f"   Values: {sample_obs}")
    
    print(f"\n🎯 Sample Action (Array):")
    print(f"   Shape: {sample_action.shape}")
    print(f"   Values: {sample_action}")
    
    # Show what each index represents
    obs_mapping = env.get_observation_key_mapping()
    action_mapping = env.get_action_key_mapping()
    
    print(f"\n🗺️  Observation Index Meaning:")
    for key, idx in obs_mapping.items():
        print(f"   obs[{idx}] = {key} = {sample_obs[idx]:.3f}")
    
    print(f"\n🎯 Action Index Meaning:")
    for key, idx in action_mapping.items():
        print(f"   action[{idx}] = {key} adjustment = {sample_action[idx]:.3f}")
    

def main():
    """Main function demonstrating the complete workflow."""
    
    print("🚁 Modified ArdupilotEnv + Stable Baselines DDPG Example")
    print("=" * 70)
    
    # Configuration
    config_path = '/home/pid_rl/rl_training/configs/default_config.yaml'
    total_timesteps = 1000  # Adjust based on your needs
    
    try:
        # Step 1: Create environment (now directly compatible)
        env, config = create_environment(config_path)
        
        # Step 2: Demonstrate the new format
        demonstrate_observation_action_format(env)
        
        # Step 3: Train DDPG agent
        model = train_ddpg_agent(env, config, total_timesteps)
        
        if model is not None:
            # Step 4: Evaluate the trained agent
            results = evaluate_agent(model, env)
            
            # Step 5: Save the model
            model_path = "./models/ddpg_ardupilot_final"
            model.save(model_path)
            print(f"\n💾 Model saved to: {model_path}")
            
            # Step 6: Show usage examples
            print(f"\n💡 Usage Examples:")
            print(f"   # The environment is now directly compatible!")
            print(f"   from stable_baselines3 import DDPG")
            print(f"   from rl_training.environments import ArdupilotEnv")
            print(f"   ")
            print(f"   env = ArdupilotEnv(config)  # No wrapper needed!")
            print(f"   model = DDPG('MlpPolicy', env)")
            print(f"   model.learn(total_timesteps=10000)")
            
        else:
            print("❌ Training failed - Stable Baselines3 not available")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up
        if 'env' in locals():
            env.close()
        print(f"\n🧹 Environment closed.")


if __name__ == "__main__":
    main() 