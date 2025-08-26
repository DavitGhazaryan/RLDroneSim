#!/usr/bin/env python3

"""
Example script showing how to use the modified ArdupilotEnv directly with Stable Baselines DDPG.
The environment now returns flattened observations and actions, making it directly compatible.
"""

import sys
sys.path.insert(0, "/home/pid_rl")

from rl_training.environments import ArdupilotEnv
from rl_training.utils.utils import load_config, validate_config, demonstrate_observation_action_format, evaluate_agent

from stable_baselines3 import DDPG
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from rl_training.utils.tb_callback import TensorboardCallback

import numpy as np


def create_action_noise_from_config(action_noise_config, action_dim):

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


def train_ddpg_agent(env, config, run_dirs=None):

    # Get training parameters from config
    training_config = config.get('training_config', {})
    ddpg_config = config.get('ddpg_params', {})
    
    # Use provided total_timesteps or fall back to config
    total_timesteps = training_config.get('total_timesteps')
    
    print(f"\n🚀 Starting DDPG training for {total_timesteps} timesteps...")
    
    # Create action noise for exploration
    action_dim = env.action_space.shape[0]
    action_noise_config = ddpg_config.get('action_noise', {})
    
    action_noise = create_action_noise_from_config(action_noise_config, action_dim)
    
    # Wrap env with Monitor for episode stats
    env = Monitor(env)

    # Resolve tensorboard and models directory from run_dirs if provided
    tensorboard_log = run_dirs['tb_dir'] if (run_dirs and 'tb_dir' in run_dirs) else None
    # Create the DDPG model with config parameters

    model = DDPG(
        "MlpPolicy",
        env,
        action_noise=action_noise,
        learning_rate=ddpg_config.get('learning_rate'),
        buffer_size=ddpg_config.get('buffer_size', 100000),
        learning_starts=ddpg_config.get('learning_starts', 100),
        batch_size=ddpg_config.get('batch_size', 64),
        tau=ddpg_config.get('tau', 0.005),
        gamma=ddpg_config.get('gamma', 0.99),
        train_freq=ddpg_config.get('train_freq', 1),
        gradient_steps=ddpg_config.get('gradient_steps', 1),
        verbose=ddpg_config.get('verbose', 1),
        tensorboard_log=tensorboard_log,
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
            if run_dirs and 'models_dir' in run_dirs:
                save_path = run_dirs['models_dir']
            checkpoint_callback = CheckpointCallback(
                save_freq=callback_config.get('save_freq'),
                save_path=save_path,
                name_prefix=callback_config.get('model_name_prefix', "ddpg_ardupilot")
            )
            break
    
    # Custom TB callback for domain metrics
    gain_keys = config.get('environment_config', {}).get('action_gains', "").split('+')
    tb_callback = TensorboardCallback(log_action_stats=True, log_gain_keys=gain_keys)
    callbacks = [cb for cb in [checkpoint_callback, tb_callback] if cb is not None]

    
    # # Train the model
    print("🎯 Training started...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks if len(callbacks) > 1 else callbacks[0] if callbacks else None,
        progress_bar=training_config.get('progress_bar'),
        reset_num_timesteps=training_config.get('reset_num_timesteps'),
    )
    
    print("✅ Training completed!")
    return model
    

def main():
    """Main function demonstrating the complete workflow."""
    
    print("🚁 ArdupilotEnv + Stable Baselines DDPG Experiment")
    print("=" * 70)
    
    config_path = '/home/pid_rl/rl_training/configs/default_config.yaml'
    
    try:
        from rl_training.utils.utils import create_run_dir, save_config_copy, save_git_info, save_metrics_json
        import subprocess
        import os

        config = load_config(config_path)
        if not validate_config(config, config["training_config"]["algo"]):
            print("❌ Configuration validation failed. Please check your config file.")
            return
        
        print("🔧 Creating  ArdupilotEnv...")
        
        env = ArdupilotEnv(config)
        
        # Prepare run directory structure
        training_config = config.get('training_config', {})
        algorithm = training_config.get('algo')
        mission = training_config.get('mission')
        runs_base = training_config.get('runs_base')
        run_dirs = create_run_dir(runs_base, algorithm, mission)
        print(f"📁 Run directory: {run_dirs['run_dir']}")

        # Save config and git info snapshots
        save_config_copy(config, run_dirs['cfg_path'])
        save_git_info(run_dirs['git_path'])
        
        model = train_ddpg_agent(env, config, run_dirs)
        
        if model is not None:
            results = evaluate_agent(model, env)
            
            # Save final model in run models dir
            model_name_prefix = training_config.get('model_name_prefix', 'ddpg_ardupilot')
            model_path = os.path.join(run_dirs['models_dir'], f"{model_name_prefix}_final")
            model.save(model_path)
            print(f"\n💾 Model saved to: {model_path}")

            # Save metrics summary
            save_metrics_json(results, run_dirs['metrics_path'])

            # Print ls of run directory
            try:
                print("\n📂 Run directory contents:")
                out = subprocess.check_output(["bash", "-lc", f"ls -la '{run_dirs['run_dir']}' && echo '---' && find '{run_dirs['run_dir']}' -maxdepth 2 -type d -print"], stderr=subprocess.STDOUT)
                print(out.decode())
            except Exception as e:
                print(f"⚠️ Could not list run directory: {e}")
            
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
        print("\n🧹 Environment closed.")


if __name__ == "__main__":
    main() 