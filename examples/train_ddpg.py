#!/usr/bin/env python3

import sys
sys.path.insert(0, "/home/pid_rl")

from rl_training.environments import SimGymEnv
from rl_training.utils.utils import load_config
from stable_baselines3 import DDPG
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from rl_training.utils.tb_callback import TensorboardCallback
from rl_training.utils.utils import create_action_noise_from_config, _latest_ckpt_path, _replay_for
from rl_training.utils.utils import create_run_dir, save_config_copy, save_git_info
import os


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
            learning_rate=ddpg_config.get('learning_rate'),
            buffer_size=ddpg_config.get('buffer_size'),
            learning_starts=ddpg_config.get('learning_starts'),
            batch_size=ddpg_config.get('batch_size'),
            tau=ddpg_config.get('tau'),
            gamma=ddpg_config.get('gamma'),
            train_freq=ddpg_config.get('train_freq'),
            gradient_steps=ddpg_config.get('gradient_steps'),
            optimize_memory_usage=ddpg_config.get('optimize_memory_usage', False),  # Optimization for memory usage
            action_noise=action_noise,  # Action noise for exploration (NormalActionNoise or others)
            replay_buffer_class=None,  # Optionally pass a custom replay buffer class
            replay_buffer_kwargs=None,  # Optionally pass kwargs for custom replay buffer
            n_steps=ddpg_config.get('n_steps', -1),  # Number of steps to collect per update (default -1)
            policy_kwargs=policy_kwargs,  # Policy network architecture (e.g., MLP)
            verbose=ddpg_config.get('verbose', 1),  # Verbosity level (0 = silent, 1 = progress bar)
            seed=ddpg_config.get('seed'),  # Seed for reproducibility
            device=ddpg_config.get('device', "auto"),  # Device to train on (cpu, cuda)
            _init_setup_model=ddpg_config.get('_init_setup_model', True),  # Whether to initialize model automatically
            tensorboard_log=tensorboard_log  # Path for TensorBoard logging
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
        log_interval=training_config.get('log_interval'),
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
        env = SimGymEnv(config, instance=instance)

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