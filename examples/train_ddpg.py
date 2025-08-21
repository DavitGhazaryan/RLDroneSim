#!/usr/bin/env python3

"""
Example script showing how to use the modified ArdupilotEnv directly with Stable Baselines DDPG.
The environment now returns flattened observations and actions, making it directly compatible.
"""

import sys
import os
sys.path.insert(0, "/home/pid_rl")

from rl_training.environments import ArdupilotEnv
from rl_training.utils.utils import load_config, validate_config, demonstrate_observation_action_format, evaluate_agent
import numpy as np

def create_action_noise_from_config(action_noise_config, action_dim):
    """
    Create action noise object from configuration.
    
    Args:
        action_noise_config: Action noise configuration dictionary
        action_dim: Dimension of the action space
        
    Returns:
        Action noise object
    """
    noise_type = action_noise_config.get('type', 'NormalActionNoise')
    
    if noise_type == 'NormalActionNoise':
        from stable_baselines3.common.noise import NormalActionNoise
        mean = action_noise_config.get('mean', 0.0)
        sigma = action_noise_config.get('sigma', 0.1)
        
        return NormalActionNoise(
            mean=mean * np.ones(action_dim),
            sigma=sigma * np.ones(action_dim)
        )
    else:
        # Default to no noise if type not recognized
        from stable_baselines3.common.noise import NormalActionNoise
        return NormalActionNoise(
            mean=np.zeros(action_dim),
            sigma=0.1 * np.ones(action_dim)
        )


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


def train_ddpg_agent(env, config, total_timesteps=None):
    """
    Train a DDPG agent on the modified environment.
    
    Args:
        env: Modified ArdupilotEnv
        config: Configuration dictionary
        total_timesteps: Total training timesteps (overrides config if provided)
        
    Returns:
        Trained DDPG model
    """
    from stable_baselines3 import DDPG
    
    from stable_baselines3.common.noise import NormalActionNoise
    from stable_baselines3.common.callbacks import CheckpointCallback

    # Get training parameters from config
    training_config = config.get('training_config', {})
    ddpg_config = config.get('ddpg_params', {})
    
    # Use provided total_timesteps or fall back to config
    if total_timesteps is None:
        total_timesteps = training_config.get('total_timesteps', 1000)
    
    print(f"\n🚀 Starting DDPG training for {total_timesteps} timesteps...")
    
    # Create action noise for exploration
    action_dim = env.action_space.shape[0]
    action_noise_config = ddpg_config.get('action_noise', {})
    
    action_noise = create_action_noise_from_config(action_noise_config, action_dim)
    
    # Create the DDPG model with config parameters
    model = DDPG(
        "MlpPolicy",
        env,
        action_noise=action_noise,
        learning_rate=ddpg_config.get('learning_rate', 1e-3),
        buffer_size=ddpg_config.get('buffer_size', 100000),
        learning_starts=ddpg_config.get('learning_starts', 100),
        batch_size=ddpg_config.get('batch_size', 64),
        tau=ddpg_config.get('tau', 0.005),
        gamma=ddpg_config.get('gamma', 0.99),
        train_freq=ddpg_config.get('train_freq', 1),
        gradient_steps=ddpg_config.get('gradient_steps', 1),
        verbose=ddpg_config.get('verbose', 1),
        tensorboard_log=ddpg_config.get('tensorboard_log', "./logs/ddpg_ardupilot/"),
        policy_kwargs=ddpg_config.get('policy_kwargs'),
        seed=ddpg_config.get('seed'),
        device=ddpg_config.get('device', "auto"),
        _init_setup_model=ddpg_config.get('_init_setup_model', True)
    )
    
    # Create checkpoint callback from config
    callbacks_config = config.get('callbacks', [])
    checkpoint_callback = None
    
    for callback_config in callbacks_config:
        if callback_config.get('type') == 'checkpoint':
            checkpoint_callback = CheckpointCallback(
                save_freq=callback_config.get('save_freq', 500),
                save_path=callback_config.get('save_path', "./models/"),
                name_prefix=callback_config.get('name_prefix', "ddpg_ardupilot")
            )
            break
    
    # If no checkpoint callback found in config, create default one
    if checkpoint_callback is None:
        checkpoint_callback = CheckpointCallback(
            save_freq=training_config.get('save_freq', 500),
            save_path=training_config.get('checkpoint_dir', "./models/"),
            name_prefix=training_config.get('model_name_prefix', "ddpg_ardupilot")
        )
    
    # Train the model
    print("🎯 Training started...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback,
        progress_bar=training_config.get('progress_bar', True),
        reset_num_timesteps=training_config.get('reset_num_timesteps', True)
    )
    
    print("✅ Training completed!")
    return model
    

def main():
    """Main function demonstrating the complete workflow."""
    
    print("🚁 Modified ArdupilotEnv + Stable Baselines DDPG Example")
    print("=" * 70)
    
    # Configuration
    config_path = '/home/pid_rl/rl_training/configs/default_config.yaml'
    
    try:
        # Step 1: Create environment (now directly compatible)
        env, config = create_environment(config_path)
        
        # Validate configuration
        if not validate_config(config, "ddpg"):
            print("❌ Configuration validation failed. Please check your config file.")
            return
        
        # Store config in environment for access by other functions
        env.config = config
        
        # Get training parameters from config
        training_config = config.get('training_config', {})
        total_timesteps = training_config.get('total_timesteps', 1000)
        
        # Step 2: Demonstrate the new format
        demonstrate_observation_action_format(env)
        
        # Step 3: Train DDPG agent
        model = train_ddpg_agent(env, config, total_timesteps)
        
        if model is not None:
            # Step 4: Evaluate the trained agent
            results = evaluate_agent(model, env)
            
            # Step 5: Save the model
            model_path = f"./{training_config.get('checkpoint_dir', './models')}/{training_config.get('model_name_prefix', 'ddpg_ardupilot')}_final"
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