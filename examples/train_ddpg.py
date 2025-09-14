#!/usr/bin/env python3

import sys
sys.path.insert(0, "/home/pid_rl")

from rl_training.environments import ArdupilotEnv
from rl_training.utils.utils import load_config, evaluate_agent
from stable_baselines3 import DDPG
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from rl_training.utils.tb_callback import TensorboardCallback
import numpy as np
from rl_training.utils.utils import create_run_dir, save_config_copy, save_git_info, save_metrics_json
import subprocess
import os, re

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

def train_ddpg_agent(env, config, run_dirs, checkpoint: str | None = None):
    """
    checkpoint:
      - None -> fresh training (current behavior)
      - path to model .zip OR directory containing checkpoints
        (e.g., '.../models' or '.../models/ddpg_ardupilot_30_steps.zip')
    """
    # setup configs
    training_config = config.get('training_config', {})
    ddpg_config = config.get('ddpg_params', {})
    callbacks_config = config.get('callbacks', [])
    tensorboard_log = run_dirs['tb_dir'] 
    
    total_timesteps = training_config.get('total_timesteps')
    print(f"\n🚀 Starting DDPG training for {total_timesteps} timesteps...")


    env = Monitor(env)

    action_dim = env.action_space.shape[0]
    action_noise = create_action_noise_from_config(ddpg_config.get('action_noise'), action_dim)

    name_prefix = "ddpg_ardupilot"  
    for callback_config in callbacks_config:
        if callback_config.get('type') == 'checkpoint':
            name_prefix = callback_config.get('model_name_prefix', name_prefix)
            break

    tb_run_name = training_config.get('tb_log_name', name_prefix)

    if checkpoint:
        model_zip, steps = _latest_ckpt_path(checkpoint, name_prefix)
        existing = [d for d in os.listdir(tensorboard_log)
                if d.startswith(tb_run_name) and os.path.isdir(os.path.join(tensorboard_log, d))]
        run_idx = len(existing)
        tb_run_name = f"{tb_run_name}_{run_idx}"
    else:
        model_zip, steps = (None, None)

    if model_zip:
        print(f"📦 Resuming from checkpoint: {model_zip}")
        model = DDPG.load(model_zip, env=env, device=ddpg_config.get('device'))
        model.action_noise = action_noise

        rb_path = _replay_for(model_zip, steps, name_prefix)
        if rb_path:
            print(f"🔄 Loading replay buffer: {rb_path}")
            model.load_replay_buffer(rb_path)
        else:
            print("⚠️ Replay buffer not found for this checkpoint; continuing without it.")
        
    else:
        policy_kwargs = ddpg_config['policy_kwargs']
        model = DDPG(
            "MlpPolicy",
            env,
            action_noise=action_noise,
            learning_rate=ddpg_config.get('learning_rate'),
            buffer_size=ddpg_config.get('buffer_size'),
            learning_starts=ddpg_config.get('learning_starts'),
            batch_size=ddpg_config.get('batch_size'),
            tau=ddpg_config.get('tau'),
            gamma=ddpg_config.get('gamma'),
            train_freq=ddpg_config.get('train_freq'),
            gradient_steps=ddpg_config.get('gradient_steps'),
            verbose=ddpg_config.get('verbose', 1),
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            seed=ddpg_config.get('seed'),
            device=ddpg_config.get('device', "auto"),
            _init_setup_model=ddpg_config.get('_init_setup_model', True)
        )

    callbacks = []
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
            callbacks.append(checkpoint_callback)

        elif callback_config.get('type') == 'tensorboard':
            gain_keys = config.get('environment_config', {}).get('action_gains', "").split('+')
            tb_callback = TensorboardCallback(log_action_stats=True, log_gain_keys=gain_keys)
            callbacks.append(tb_callback)


    reset_flag = training_config.get('reset_num_timesteps')
    if not reset_flag and model_zip:
        reset_flag = False
    print(f"REset flag {reset_flag}")
    ## Main learning code
    print("🎯 Training started...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks if len(callbacks) > 1 else callbacks[0] if callbacks else None,
        log_interval=10,
        progress_bar=training_config.get('progress_bar'),
        reset_num_timesteps=reset_flag,
        tb_log_name=tb_run_name 
    )

    print("✅ Training completed!")
    return model

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("instance", nargs="?", type=int, default=1)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model .zip OR a directory/file inside an existing run to resume.")
    args = parser.parse_args()

    if args.instance not in (1, 2):
        print("Error: argument must be 1 or 2.")
        sys.exit(1)
    instance = args.instance
    checkpoint = args.checkpoint
    print(f"Using value: {instance}")

    print("🚁 ArdupilotEnv + Stable Baselines DDPG Experiment")
    print("=" * 70)

    config_path = '/home/pid_rl/rl_training/configs/default_config.yaml'

    try:
        config = load_config(config_path)
        env = ArdupilotEnv(config, instance=instance)

        # Prepare (or reuse) run directory structure
        training_config = config.get('training_config', {})
        algorithm = training_config.get('algo')
        mission = training_config.get('mission')
        runs_base = training_config.get('runs_base')

        # If resuming: reuse the original run root; else create a fresh stamped run
        if checkpoint:
            run_dirs = create_run_dir(runs_base, algorithm, mission, resume_from=checkpoint)
            print("↩️  Resuming run:")
        else:
            run_dirs = create_run_dir(runs_base, algorithm, mission)
            print("🆕 Fresh run:")

        print(f"📁 Run directory: {run_dirs['run_dir']}")

        # Save config and git info only for fresh runs (avoid overwriting on resume)
        if not checkpoint:
            save_config_copy(config, run_dirs['cfg_path'])
            save_git_info(run_dirs['git_path'])
        else:
            # If you still want snapshots on resume, write to *_resume timestamped files instead.
            pass

        # Train (fresh or resume)
        model = train_ddpg_agent(env, config, run_dirs, checkpoint=checkpoint)

        # # Evaluate & save final model
        # if model is not None:
        #     n_eval = (config.get("evaluation_config") or {}).get("n_eval_episodes", 5)
        #     results = evaluate_agent(model, env, n_eval)

        #     model_name_prefix = training_config.get('model_name_prefix', 'ddpg_ardupilot')
        #     model_path = os.path.join(run_dirs['models_dir'], f"{model_name_prefix}_final")
        #     model.save(model_path)
        #     print(f"\n💾 Model saved to: {model_path}")

        #     save_metrics_json(results, run_dirs['metrics_path'])
        #     try:
        #         print("\n📂 Run directory contents:")
        #         out = subprocess.check_output([
        #             "bash", "-lc",
        #             f"ls -la '{run_dirs['run_dir']}' && echo '---' && find '{run_dirs['run_dir']}' -maxdepth 2 -type d -print"
        #         ], stderr=subprocess.STDOUT)
        #         print(out.decode())
        #     except Exception as e:
        #         print(f"⚠️ Could not list run directory: {e}")
        # else:
        #     print("❌ Training failed - Stable Baselines3 not available")

    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'env' in locals():
            env.close()
        print("\n🧹 Environment closed.")

if __name__ == "__main__":
    main() 