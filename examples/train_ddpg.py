#!/usr/bin/env python3

import sys
sys.path.insert(0, "/home/pid_rl")

from rl_training.environments import ArdupilotEnv
from rl_training.utils.utils import load_config, validate_config, evaluate_agent
from stable_baselines3 import DDPG
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from rl_training.utils.tb_callback import TensorboardCallback
import numpy as np
from rl_training.utils.utils import create_run_dir, save_config_copy, save_git_info, save_metrics_json
import subprocess
import os, glob, re

def create_action_noise_from_config(action_noise_config, action_dim):
    noise_type = action_noise_config.get('type', 'NormalActionNoise')
    
    if noise_type == 'NormalActionNoise':
        from stable_baselines3.common.noise import NormalActionNoise
        mean = action_noise_config.get('mean')
        sigma = action_noise_config.get('sigma')
        
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


def _latest_ckpt_path(path_like: str, name_prefix: str):
    """
    Accepts a directory or a specific .zip path.
    Returns (model_zip_path, steps_int) or (None, None) if not found.
    """
    print(path_like, name_prefix)
    if path_like is None:
        return None, None
    if os.path.isfile(path_like) and path_like.endswith(".zip"):
        m = re.search(r"_(\d+)_steps\.zip$", os.path.basename(path_like))
        steps = int(m.group(1)) if m else None
        return path_like, steps
    return None, None


def _replay_for(model_zip_path: str, steps: int, name_prefix: str):
    """
    Finds the replay buffer file that matches the given model steps.
    Expects: <dir>/<name_prefix>_replay_buffer_<steps>_steps.pkl
    """
    if model_zip_path is None or steps is None:
        return None
    directory = os.path.dirname(model_zip_path)
    rb = os.path.join(directory, f"{name_prefix}_replay_buffer_{steps}_steps.pkl")
    return rb if os.path.exists(rb) else None

def train_ddpg_agent(env, config, run_dirs=None, checkpoint: str | None = None):
    """
    checkpoint:
      - None -> fresh training (current behavior)
      - path to model .zip OR directory containing checkpoints
        (e.g., '.../models' or '.../models/ddpg_ardupilot_30_steps.zip')
    """
    training_config = config.get('training_config', {})
    ddpg_config = config.get('ddpg_params', {})
    callbacks_config = config.get('callbacks', [])

    total_timesteps = training_config.get('total_timesteps')
    print(f"\n🚀 Starting DDPG training for {total_timesteps} timesteps...")

    env = Monitor(env)

    action_dim = env.action_space.shape[0]
    action_noise_config = ddpg_config.get('action_noise', {})
    action_noise = create_action_noise_from_config(action_noise_config, action_dim)

    tensorboard_log = run_dirs['tb_dir'] if (run_dirs and 'tb_dir' in run_dirs) else None
    name_prefix = "ddpg_ardupilot"  # default prefix
    for callback_config in callbacks_config:
        if callback_config.get('type') == 'checkpoint':
            name_prefix = callback_config.get('model_name_prefix', name_prefix)
            break

    model_zip, steps = _latest_ckpt_path(checkpoint, name_prefix) if checkpoint else (None, None)
    print("Threr we go")
    print(model_zip, steps)
    if model_zip:
        print(f"📦 Resuming from checkpoint: {model_zip}")
        model = DDPG.load(model_zip, env=env, device=ddpg_config.get('device', "auto"))
        model.action_noise = action_noise

        # try to load matching replay buffer
        rb_path = _replay_for(model_zip, steps, name_prefix)
        if rb_path:
            print(f"🔄 Loading replay buffer: {rb_path}")
            model.load_replay_buffer(rb_path)
        else:
            print("⚠️ Replay buffer not found for this checkpoint; continuing without it.")
    else:
        # fresh init (current behavior)
        policy_kwargs = ddpg_config['policy_kwargs']
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
            policy_kwargs=policy_kwargs,
            seed=ddpg_config.get('seed'),
            device=ddpg_config.get('device', "auto"),
            _init_setup_model=ddpg_config.get('_init_setup_model', True)
        )

    # ====== Callbacks (unchanged) ======
    checkpoint_callback = None
    for callback_config in callbacks_config:
        if callback_config.get('type') == 'checkpoint':
            if run_dirs and 'models_dir' in run_dirs:
                save_path = run_dirs['models_dir']
            checkpoint_callback = CheckpointCallback(
                save_freq=callback_config.get('save_freq'),
                save_path=save_path,
                name_prefix=name_prefix,
                save_replay_buffer=True
            )
            break

    gain_keys = config.get('environment_config', {}).get('action_gains', "").split('+')
    tb_callback = TensorboardCallback(log_action_stats=True, log_gain_keys=gain_keys)
    callbacks = [cb for cb in [checkpoint_callback, tb_callback] if cb is not None]

    # If resuming and user didn't specify, default to not resetting timesteps
    reset_flag = training_config.get('reset_num_timesteps')
    if reset_flag is None and model_zip:
        reset_flag = False

    print("🎯 Training started...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks if len(callbacks) > 1 else callbacks[0] if callbacks else None,
        progress_bar=training_config.get('progress_bar'),
        reset_num_timesteps=reset_flag,
    )

    print("✅ Training completed!")
    return model

def main():
    """Main function demonstrating the complete workflow."""
    if len(sys.argv) == 1:
        instance = 1
    else:
        try:
            instance = int(sys.argv[1])
        except ValueError:
            print("Error: argument must be an integer (1 or 2).")
            sys.exit(1)
        if instance not in (1, 2):
            print("Error: argument must be 1 or 2.")
            sys.exit(1)
    print(f"Using value: {instance}")
    
    print("🚁 ArdupilotEnv + Stable Baselines DDPG Experiment")
    print("=" * 70)
    
    config_path = '/home/pid_rl/rl_training/configs/default_config.yaml'
    
    try:
        config = load_config(config_path)
        env = ArdupilotEnv(config, instance=instance)
        
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
        
        # model = train_ddpg_agent(env, config, run_dirs)
        model = train_ddpg_agent(env, config, run_dirs, checkpoint="/home/pid_rl/rl_training/runs/ddpg/hover/20250906_181428/models/ddpg_ardupilot_45600_steps.zip")  
        if model is not None:
            results = evaluate_agent(model, env, config.get("evaluation_config").get("n_eval_episodes"))
            
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