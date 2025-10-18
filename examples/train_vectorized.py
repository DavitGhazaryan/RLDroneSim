#!/usr/bin/env python3

import sys
sys.path.insert(0, "/home/pid_rl")

from rl_training.environments import SimGymEnv
from rl_training.utils.utils import load_config
from stable_baselines3.common.callbacks import CheckpointCallback       # type: ignore
from stable_baselines3.common.monitor import Monitor                    # type: ignore
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize  # type: ignore

from rl_training.utils.tb_callback import TensorboardCallback
from rl_training.utils.utils import create_run_dir, save_config_copy, save_git_info, create_action_noise_from_config, _extract_steps, _replay_for, _get_config, _vecnormalize_for
import os
import re 



def train_agent(env, config, run_dirs, checkpoint: str | None = None):
    """
    checkpoint:
      - None -> fresh training (current behavior)
      - path to model .zip OR directory containing checkpoints
        (e.g., '.../models' or '.../models/td3_ardupilot_30_steps.zip')
    """

    # setup configs
    training_config = config.get('training_config')
    algo = training_config['algo']
    algo_config = config.get(f'{algo}_params')
    callbacks_config = config.get('callbacks', [])
    tensorboard_log = run_dirs['tb_dir'] 
    
    total_timesteps = training_config.get('total_timesteps')
    if checkpoint:
        print(f"\n🚀 Resuming {algo} training for {total_timesteps} timesteps...")
    else:
        print(f"\n🚀 Starting {algo} training for {total_timesteps} timesteps...")

    action_dim = env.action_space.shape[0]
    action_noise = create_action_noise_from_config(algo_config.get('action_noise'), action_dim)

    tb_run_name = "run"
    if checkpoint:
        steps = _extract_steps(checkpoint)
        print("here")
        existing = [d for d in os.listdir(tensorboard_log)
                if d.startswith(tb_run_name) and os.path.isdir(os.path.join(tensorboard_log, d))]
        run_idx = len(existing)
        tb_run_name = f"{tb_run_name}_{run_idx}"
    else:
        steps = None

    def linear_schedule(progress_remaining):
        return algo_config.get('learning_rate') * progress_remaining

    if checkpoint:
        print(f"📦 Resuming from checkpoint: {checkpoint}")
        if algo == 'td3':
            from stable_baselines3 import TD3   # type: ignore
            model = TD3.load(checkpoint, env=env, device=algo_config.get('device'))
        elif algo == 'ddpg':
            from stable_baselines3.ddpg import DDPG   # type: ignore
            model = DDPG.load(checkpoint, env=env, device=algo_config.get('device'))

        model.action_noise = action_noise

        rb_path = _replay_for(checkpoint)
        if rb_path:
            print(f"🔄 Loading replay buffer: {rb_path}")
            model.load_replay_buffer(rb_path)
        else:
            print("⚠️ Replay buffer not found for this checkpoint; continuing without it.")     
    else:
        policy_kwargs = algo_config['policy_kwargs']
        if algo == 'td3':
            from stable_baselines3 import TD3   # type: ignore
            model = TD3(
                "MlpPolicy",
                env,
                learning_rate=linear_schedule,
                buffer_size=algo_config.get('buffer_size'),
                learning_starts=algo_config.get('learning_starts'),
                batch_size=algo_config.get('batch_size'),
                tau=algo_config.get('tau'),
                gamma=algo_config.get('gamma'),
                train_freq=algo_config.get('train_freq'),
                gradient_steps=algo_config.get('gradient_steps'),
                action_noise=action_noise,  # For exploration (NormalActionNoise)
                replay_buffer_class=None,   # Optionally customize the replay buffer class
                replay_buffer_kwargs=None,  # Optionally customize the replay buffer kwargs
                n_steps=algo_config.get('n_steps'),  # Default is -1, can be set for specific training steps
                policy_delay=algo_config.get('policy_delay'),  # The number of steps to wait before updating policy
                target_policy_noise=algo_config.get('target_policy_noise'),  # Noise for target policy smoothing
                target_noise_clip=algo_config.get('target_noise_clip'),  # Clip range for target noise
                stats_window_size=algo_config.get('stats_window_size'),  # Size of the statistics window for updates
                verbose=algo_config.get('verbose'),  # Verbosity level (0 = silent, 1 = progress bar)
                tensorboard_log=tensorboard_log,  # Path to log for TensorBoard
                policy_kwargs=policy_kwargs,  # Policy network architecture
                seed=algo_config.get('seed', None),  # Random seed for reproducibility
                device=algo_config.get('device', "auto"),  # Device to run on (e.g., "cpu", "cuda")
                _init_setup_model=algo_config.get('_init_setup_model', True)  # Whether to initialize model automatically
            )
        else:
            from stable_baselines3.ddpg import DDPG   # type: ignore
            model = DDPG(
                "MlpPolicy",
                env,
                learning_rate=algo_config.get('learning_rate'),
                buffer_size=algo_config.get('buffer_size'),
                learning_starts=algo_config.get('learning_starts'),
                batch_size=algo_config.get('batch_size'),
                tau=algo_config.get('tau'),
                gamma=algo_config.get('gamma'),
                train_freq=algo_config.get('train_freq'),
                gradient_steps=algo_config.get('gradient_steps'),
                optimize_memory_usage=algo_config.get('optimize_memory_usage', False),  # Optimization for memory usage
                action_noise=action_noise,  # Action noise for exploration (NormalActionNoise or others)
                replay_buffer_class=None,  # Optionally pass a custom replay buffer class
                replay_buffer_kwargs=None,  # Optionally pass kwargs for custom replay buffer
                n_steps=algo_config.get('n_steps', -1),  # Number of steps to collect per update (default -1)
                policy_kwargs=policy_kwargs,  # Policy network architecture (e.g., MLP)
                verbose=algo_config.get('verbose', 1),  # Verbosity level (0 = silent, 1 = progress bar)
                seed=algo_config.get('seed'),  # Seed for reproducibility
                device=algo_config.get('device', "auto"),  # Device to train on (cpu, cuda)
                _init_setup_model=algo_config.get('_init_setup_model', True),  # Whether to initialize model automatically
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
                name_prefix=algo,
                save_replay_buffer=True,
                save_vecnormalize=True
            )
            callbacks.append(checkpoint_callback)

        elif callback_config.get('type') == 'tensorboard':
            gain_keys = config.get('environment_config', {}).get('action_gains', "").split('+')
            tb_callback = TensorboardCallback(log_action_stats=True, log_gain_keys=gain_keys)
            callbacks.append(tb_callback)


    reset_flag = training_config.get('reset_num_timesteps')
    if not reset_flag and checkpoint:
        reset_flag = False
    print(f"Reset flag {reset_flag}")
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
    parser.add_argument("--newconfig", type=bool, default=False,
                        help="use the new default_config.yaml for training.")
    args = parser.parse_args()

    if args.instance not in (1, 2):
        print("Error: argument must be 1 or 2.")
        sys.exit(1)

    instance = args.instance
    checkpoint = args.checkpoint
    new_config = args.newconfig
    print()
    print(f"Using value: {instance}")

    if not checkpoint or new_config:
        config_path = '/home/pid_rl/rl_training/configs/default_config.yaml'
    else:
        config_path = _get_config(checkpoint)

    try:
        config = load_config(config_path)

        def make_env():
            return Monitor(SimGymEnv(config, instance=instance))        
        dummy_vec = DummyVecEnv([make_env])
        venv = VecNormalize(dummy_vec, norm_obs=True, norm_reward=False, clip_obs=10.0)

        training_config = config.get('training_config')
        algorithm = training_config.get('algo')      # ddpg, td3 
        mission = training_config.get('mission')     # hover
        runs_base = training_config.get('runs_base')

        # If resuming: reuse the original run root; else create a fresh stamped run
        if checkpoint:
            run_dirs = create_run_dir(runs_base, algorithm, mission, resume_from=checkpoint)
            vecnorm_path = _vecnormalize_for(checkpoint)
            print(f" Loading vecnormalize: {vecnorm_path}")

            if os.path.exists(vecnorm_path):
                venv = VecNormalize.load(vecnorm_path, dummy_vec)
                venv.training = True
                venv.clip_obs = 10
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
        model = train_agent(venv, config, run_dirs, checkpoint=checkpoint)

        venv.save(vecnorm_path)

    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'env' in locals():
            env.close() # type: ignore
        print("\n🧹 Environment closed.")

if __name__ == "__main__":
    main()