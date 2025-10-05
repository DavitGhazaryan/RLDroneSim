#!/usr/bin/env python3

import sys
sys.path.insert(0, "/home/pid_rl")

from rl_training.environments import SimGymEnv
from rl_training.utils.utils import load_config
from stable_baselines3 import TD3  
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from rl_training.utils.tb_callback import TensorboardCallback
from rl_training.utils.utils import create_run_dir, save_config_copy, save_git_info, create_action_noise_from_config, _latest_ckpt_path, _replay_for
import os



def train_td3_agent(env, config, run_dirs, checkpoint: str | None = None):
    """
    checkpoint:
      - None -> fresh training (current behavior)
      - path to model .zip OR directory containing checkpoints
        (e.g., '.../models' or '.../models/td3_ardupilot_30_steps.zip')
    """

    # setup configs
    training_config = config.get('training_config', {})
    td3_config = config.get('td3_params')  # Changed ddpg_params to td3_params
    callbacks_config = config.get('callbacks', [])
    tensorboard_log = run_dirs['tb_dir'] 
    
    total_timesteps = training_config.get('total_timesteps')
    print(f"\n🚀 Starting TD3 training for {total_timesteps} timesteps...")

    env = Monitor(env)

    action_dim = env.action_space.shape[0]
    action_noise = create_action_noise_from_config(td3_config.get('action_noise'), action_dim)

    name_prefix = "td3_ardupilot"  # Changed prefix for TD3  
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

    def linear_schedule(progress_remaining):
        return td3_config.get('learning_rate') * progress_remaining


    if model_zip:
        print(f"📦 Resuming from checkpoint: {model_zip}")
        model = TD3.load(model_zip, env=env, device=td3_config.get('device'))  # Changed DDPG to TD3
        model.action_noise = action_noise

        rb_path = _replay_for(model_zip, steps, name_prefix)
        if rb_path:
            print(f"🔄 Loading replay buffer: {rb_path}")
            model.load_replay_buffer(rb_path)
        else:
            print("⚠️ Replay buffer not found for this checkpoint; continuing without it.")
        
    else:
        policy_kwargs = td3_config['policy_kwargs']
        model = TD3(
            "MlpPolicy",
            env,
            learning_rate=linear_schedule,
            buffer_size=td3_config.get('buffer_size'),
            learning_starts=td3_config.get('learning_starts'),
            batch_size=td3_config.get('batch_size'),
            tau=td3_config.get('tau'),
            gamma=td3_config.get('gamma'),
            train_freq=td3_config.get('train_freq'),
            gradient_steps=td3_config.get('gradient_steps'),
            action_noise=action_noise,  # For exploration (NormalActionNoise)
            replay_buffer_class=None,   # Optionally customize the replay buffer class
            replay_buffer_kwargs=None,  # Optionally customize the replay buffer kwargs
            n_steps=td3_config.get('n_steps'),  # Default is -1, can be set for specific training steps
            policy_delay=td3_config.get('policy_delay'),  # The number of steps to wait before updating policy
            target_policy_noise=td3_config.get('target_policy_noise'),  # Noise for target policy smoothing
            target_noise_clip=td3_config.get('target_noise_clip'),  # Clip range for target noise
            stats_window_size=td3_config.get('stats_window_size'),  # Size of the statistics window for updates
            verbose=td3_config.get('verbose'),  # Verbosity level (0 = silent, 1 = progress bar)
            tensorboard_log=tensorboard_log,  # Path to log for TensorBoard
            policy_kwargs=policy_kwargs,  # Policy network architecture
            seed=td3_config.get('seed', None),  # Random seed for reproducibility
            device=td3_config.get('device', "auto"),  # Device to run on (e.g., "cpu", "cuda")
            _init_setup_model=td3_config.get('_init_setup_model', True)  # Whether to initialize model automatically
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

    print("🚁 ArdupilotEnv + Stable Baselines TD3 Experiment")  # Changed DDPG to TD3
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
        model = train_td3_agent(env, config, run_dirs, checkpoint=checkpoint)

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