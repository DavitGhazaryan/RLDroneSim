#!/usr/bin/env python3

import sys
sys.path.insert(0, "/home/pid_rl")

from rl_training.environments import SimGymEnv
from rl_training.utils.utils import load_config
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor                    # type: ignore
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize  # type: ignore

from rl_training.utils.tb_callback import TensorboardCallback
from rl_training.utils.utils import create_run_dir, save_config_copy, save_git_info, create_action_noise_from_config, _extract_steps, _replay_for, _get_config, _vecnormalize_for
import os
import re 

import random
import numpy as np
import torch

from pathlib import Path
from datetime import datetime


class BestTrainingRewardCallback(BaseCallback):
    """
    Saves the model whenever the mean episode reward improves.

    This uses SB3's ep_info_buffer, which is populated when the environment
    is wrapped with Monitor.
    """

    def __init__(self, check_freq, save_path, window_size=5, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.window_size = window_size
        self.best_mean_reward = -np.inf

    def _init_callback(self):
        os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq != 0:
            return True

        if len(self.model.ep_info_buffer) == 0:
            return True

        recent_rewards = [
            ep_info["r"]
            for ep_info in list(self.model.ep_info_buffer)[-self.window_size:]
            if "r" in ep_info
        ]

        if len(recent_rewards) == 0:
            return True

        mean_reward = float(np.mean(recent_rewards))

        if mean_reward > self.best_mean_reward:
            self.best_mean_reward = mean_reward

            model_path = os.path.join(self.save_path, "best_model")
            self.model.save(model_path)

            vec_env = self.model.get_vec_normalize_env()
            if vec_env is not None:
                vec_env.save(os.path.join(self.save_path, "best_vecnormalize.pkl"))

            with open(os.path.join(self.save_path, "best_model_info.txt"), "w") as f:
                f.write(f"best_mean_reward: {self.best_mean_reward}\n")
                f.write(f"num_timesteps: {self.num_timesteps}\n")
                f.write(f"window_size: {self.window_size}\n")

            if self.verbose > 0:
                print(
                    f"✅ New best model saved at {self.num_timesteps} steps "
                    f"with mean reward {self.best_mean_reward:.3f}"
                )

        return True
    
class LatestReplayBufferCallback(BaseCallback):
    """
    Saves only the latest replay buffer for off-policy algorithms such as TD3 and SAC.

    The file is overwritten each time, so we do not create huge folders with
    replay buffers for every checkpoint.
    """

    def __init__(self, save_freq, save_path, algo, verbose=1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.algo = algo

    def _init_callback(self):
        os.makedirs(self.save_path, exist_ok=True)

    def _save_replay_buffer(self):
        if not hasattr(self.model, "save_replay_buffer"):
            return

        replay_buffer_path = os.path.join(
            self.save_path,
            f"{self.algo}_latest_replay_buffer.pkl"
        )

        self.model.save_replay_buffer(replay_buffer_path)

        with open(os.path.join(self.save_path, f"{self.algo}_latest_replay_buffer_info.txt"), "w") as f:
            f.write(f"num_timesteps: {self.num_timesteps}\n")
            f.write(f"replay_buffer_path: {replay_buffer_path}\n")

        if self.verbose > 0:
            print(
                f"💾 Latest replay buffer saved at {self.num_timesteps} steps: "
                f"{replay_buffer_path}"
            )

    def _on_step(self):
        if self.n_calls % self.save_freq == 0:
            self._save_replay_buffer()

        return True

    def _on_training_end(self):
        self._save_replay_buffer()

def save_training_parameters(config, output_path):
    """
    Save the main training-related configuration sections
    in a readable text file.
    """
    sections = [
        "environment_config",
        "training_config",
        "ppo_params",
        "td3_params",
        "ddpg_params",
        "sac_params",
        "callbacks",
        "reward_config",
    ]

    with open(output_path, "w", encoding="utf-8") as file:
        for section in sections:
            file.write("=" * 80 + "\n")
            file.write(f"{section}\n")
            file.write("=" * 80 + "\n")

            section_config = config.get(section)

            if section_config is None:
                file.write("Section not found in configuration.\n\n")
            else:
                import yaml
                file.write(
                    yaml.safe_dump(
                        section_config,
                        sort_keys=False,
                        default_flow_style=False,
                    )
                )
                file.write("\n")

    print(f"✅ Training parameters saved to: {output_path}")

def create_grouped_experiment_run_dir(
    runs_base,
    algorithm,
    mission,
    config_path=None,
    seed=None,
    experiment_id=None,
):
    """
    Create grouped experiment directory:

    runs/hover/<config_name>_seed<seed>/
        td3/
            models/
            tb/
            logs/
            config.yaml
            git_info.txt
        ppo/
            ...
        sac/
            ...
    """

    config_name = Path(config_path).stem if config_path else "default_config"

    if experiment_id is None:
        if seed is not None:
            experiment_id = f"{config_name}_seed{seed}"
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_id = f"{config_name}_{timestamp}"

    experiment_dir = Path(runs_base) / mission / experiment_id
    run_dir = experiment_dir / algorithm

    # Avoid accidentally overwriting an existing run of the same algorithm.
    if run_dir.exists() and any(run_dir.iterdir()):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = experiment_dir / f"{algorithm}_{timestamp}"

    models_dir = run_dir / "models"
    tb_dir = run_dir / "tb"
    logs_dir = run_dir / "logs"

    models_dir.mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    return {
        "experiment_dir": str(experiment_dir),
        "run_dir": str(run_dir),
        "models_dir": str(models_dir),
        "tb_dir": str(tb_dir),
        "logs_dir": str(logs_dir),
        "cfg_path": str(run_dir / "config.yaml"),
        "git_path": str(run_dir / "git_info.txt"),
    }

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
    off_policy_algos = ['td3', 'ddpg', 'sac']
    callbacks_config = config.get('callbacks', [])
    tensorboard_log = run_dirs['tb_dir'] 
    
    total_timesteps = training_config.get('total_timesteps')
    if checkpoint:
        print(f"\n🚀 Resuming {algo} training for {total_timesteps} timesteps...")
    else:
        print(f"\n🚀 Starting {algo} training for {total_timesteps} timesteps...")

    action_dim = env.action_space.shape[0]

    # TD3/DDPG use external action noise.
    # PPO uses its stochastic policy for exploration, so no action noise is needed.
    if algo in ['td3', 'ddpg']:
        action_noise = create_action_noise_from_config(algo_config.get('action_noise'), action_dim)
    else:
        action_noise = None

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
            model.action_noise = action_noise

        elif algo == 'ddpg':
            from stable_baselines3.ddpg import DDPG   # type: ignore
            model = DDPG.load(checkpoint, env=env, device=algo_config.get('device'))
            model.action_noise = action_noise

        elif algo == 'ppo':
            from stable_baselines3 import PPO   # type: ignore
            model = PPO.load(checkpoint, env=env, device=algo_config.get('device', 'auto'))

        elif algo == 'sac':
            from stable_baselines3 import SAC   # type: ignore
            model = SAC.load(checkpoint, env=env, device=algo_config.get('device', 'auto'))
        
        else:
            raise ValueError(f"Unsupported algorithm for resume: {algo}")

        # Replay buffers are used only by off-policy algorithms.
        if algo in off_policy_algos:
            replay_buffer_path = checkpoint.replace(".zip", "_replay_buffer.pkl")

            latest_replay_buffer_path = os.path.join(
                run_dirs["models_dir"],
                f"{algo}_latest_replay_buffer.pkl"
            )

            if os.path.exists(replay_buffer_path):
                print(f"Loading replay buffer: {replay_buffer_path}")
                model.load_replay_buffer(replay_buffer_path)

            elif os.path.exists(latest_replay_buffer_path):
                print(f"Loading latest replay buffer: {latest_replay_buffer_path}")
                model.load_replay_buffer(latest_replay_buffer_path)

            else:
                print("⚠️ Replay buffer not found for this checkpoint; continuing without it.")
        # if algo in off_policy_algos:
        #     rb_path = _replay_for(checkpoint)
        #     if rb_path:
        #         print(f"🔄 Loading replay buffer: {rb_path}")
        #         model.load_replay_buffer(rb_path)
        #     else:
        #         print("⚠️ Replay buffer not found for this checkpoint; continuing without it.")     
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
        elif algo == 'ddpg':
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
                optimize_memory_usage=algo_config.get('optimize_memory_usage', False),
                action_noise=action_noise,
                replay_buffer_class=None,
                replay_buffer_kwargs=None,
                n_steps=algo_config.get('n_steps', -1),
                policy_kwargs=policy_kwargs,
                verbose=algo_config.get('verbose', 1),
                seed=algo_config.get('seed'),
                device=algo_config.get('device', "auto"),
                _init_setup_model=algo_config.get('_init_setup_model', True),
                tensorboard_log=tensorboard_log
            )
        elif algo == 'sac':
            from stable_baselines3 import SAC   # type: ignore
            model = SAC(
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
                action_noise=None,
                replay_buffer_class=None,
                replay_buffer_kwargs=None,
                optimize_memory_usage=algo_config.get('optimize_memory_usage', False),
                ent_coef=algo_config.get('ent_coef', 'auto'),
                target_update_interval=algo_config.get('target_update_interval', 1),
                target_entropy=algo_config.get('target_entropy', 'auto'),
                use_sde=algo_config.get('use_sde', False),
                sde_sample_freq=algo_config.get('sde_sample_freq', -1),
                use_sde_at_warmup=algo_config.get('use_sde_at_warmup', False),
                stats_window_size=algo_config.get('stats_window_size', 100),
                tensorboard_log=tensorboard_log,
                policy_kwargs=policy_kwargs,
                verbose=algo_config.get('verbose', 1),
                seed=algo_config.get('seed', None),
                device=algo_config.get('device', "auto"),
                _init_setup_model=algo_config.get('_init_setup_model', True)
            )            
        elif algo == 'ppo':
            from stable_baselines3 import PPO   # type: ignore
            model = PPO(
                "MlpPolicy",
                env,
                learning_rate=linear_schedule,
                n_steps=algo_config.get('n_steps', 2048),
                batch_size=algo_config.get('batch_size', 64),
                n_epochs=algo_config.get('n_epochs', 10),
                gamma=algo_config.get('gamma', 0.99),
                gae_lambda=algo_config.get('gae_lambda', 0.95),
                clip_range=algo_config.get('clip_range', 0.2),
                clip_range_vf=algo_config.get('clip_range_vf', None),
                normalize_advantage=algo_config.get('normalize_advantage', True),
                ent_coef=algo_config.get('ent_coef', 0.001),
                vf_coef=algo_config.get('vf_coef', 0.5),
                max_grad_norm=algo_config.get('max_grad_norm', 0.5),
                use_sde=algo_config.get('use_sde', False),
                sde_sample_freq=algo_config.get('sde_sample_freq', -1),
                target_kl=algo_config.get('target_kl', None),
                stats_window_size=algo_config.get('stats_window_size', 100),
                verbose=algo_config.get('verbose', 1),
                tensorboard_log=tensorboard_log,
                policy_kwargs=policy_kwargs,
                seed=algo_config.get('seed', None),
                device=algo_config.get('device', "auto"),
                _init_setup_model=algo_config.get('_init_setup_model', True)
            )

        else:
            raise ValueError(f"Unsupported algorithm: {algo}")

    
    callbacks = []

    for callback_config in callbacks_config:
        if callback_config.get('type') == 'checkpoint':
            if run_dirs and 'models_dir' in run_dirs:
                save_path = run_dirs['models_dir']

            checkpoint_callback = CheckpointCallback(
                save_freq=callback_config.get('save_freq'),
                save_path=save_path,
                name_prefix=algo,
                save_replay_buffer=callback_config.get('save_replay_buffer', False),
                save_vecnormalize=True
            )

            callbacks.append(checkpoint_callback)

        elif callback_config.get('type') == 'best_reward':
            if run_dirs and 'models_dir' in run_dirs:
                save_path = run_dirs['models_dir']

            best_reward_callback = BestTrainingRewardCallback(
                check_freq=callback_config.get('check_freq', 2000),
                save_path=save_path,
                window_size=callback_config.get('window_size', 5),
                verbose=1
            )

            callbacks.append(best_reward_callback)

        elif callback_config.get('type') == 'latest_replay_buffer':
            if algo in off_policy_algos:
                if run_dirs and 'models_dir' in run_dirs:
                    save_path = run_dirs['models_dir']

                latest_replay_callback = LatestReplayBufferCallback(
                    save_freq=callback_config.get('save_freq', 10000),
                    save_path=save_path,
                    algo=algo,
                    verbose=1
                )

                callbacks.append(latest_replay_callback)
        elif callback_config.get('type') == 'tensorboard':
            gain_keys = config.get('environment_config', {}).get('action_gains', "").split('+')
            tb_callback = TensorboardCallback(
                log_action_stats=True,
                log_gain_keys=gain_keys
            )
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

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model .zip OR a directory/file inside an existing run to resume."
    )

    parser.add_argument(
        "--newconfig",
        action="store_true",
        help="Use the current default_config.yaml for training even when resuming."
    )

    parser.add_argument(
        "--algo",
        type=str,
        choices=["td3", "sac", "ppo", "ddpg"],
        default=None,
        help="Override algorithm from config."
    )

    parser.add_argument(
        "--total_timesteps",
        type=int,
        default=None,
        help="Override total_timesteps from config."
    )

    parser.add_argument(
    "--seed",
    type=int,
    default=None,
    help="Random seed for reproducible independent runs."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config YAML file."
    )
    parser.add_argument(
        "--experiment_id",
        type=str,
        default=None,
        help="Shared experiment folder name, e.g. exp_R1_baseline_seed303_20260915_1515."
)

    args = parser.parse_args()

    # if args.instance not in (1, 2):
    #     print("Error: argument must be 1 or 2.")
    #     sys.exit(1)
    if args.instance < 1:
        print("Error: instance must be >= 1.")
        sys.exit(1)

    instance = args.instance
    checkpoint = args.checkpoint
    new_config = args.newconfig
    print()
    print(f"Using value: {instance}")

    # if not checkpoint or new_config:
    #     config_path = '/home/pid_rl/rl_training/configs/default_config.yaml'
    # else:
    #     config_path = _get_config(checkpoint)
    if args.config:
        config_path = args.config
    elif not checkpoint or new_config:
        config_path = '/home/pid_rl/rl_training/configs/default_config.yaml'
    else:
        config_path = _get_config(checkpoint)

    try:
        config = load_config(config_path)
        if args.algo is not None:
            config["training_config"]["algo"] = args.algo

        if args.total_timesteps is not None:
            config["training_config"]["total_timesteps"] = args.total_timesteps

        if args.seed is not None:
            algo = config["training_config"]["algo"]
            algo_params_key = f"{algo}_params"

            if algo_params_key not in config:
                raise KeyError(f"Could not find {algo_params_key} in config")

            config[algo_params_key]["seed"] = args.seed
            config["training_config"]["seed"] = args.seed

            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)

            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.seed)

            print(f"Using seed: {args.seed}")

        def make_env():
            return Monitor(SimGymEnv(config, instance=instance))        
        dummy_vec = DummyVecEnv([make_env])
        venv = VecNormalize(dummy_vec, norm_obs=True, norm_reward=False, clip_obs=10.0)

        training_config = config.get('training_config')
        algorithm = training_config.get('algo')      # ddpg, td3 
        mission = training_config.get('mission')     # hover
        runs_base = training_config.get('runs_base')

        
        if checkpoint:
            run_dirs = create_run_dir(runs_base, algorithm, mission, resume_from=checkpoint)
            vecnorm_path = _vecnormalize_for(checkpoint)
            print(f" Loading vecnormalize: {vecnorm_path}")

            if os.path.exists(vecnorm_path):
                venv = VecNormalize.load(vecnorm_path, dummy_vec)
                venv.training = True
                venv.clip_obs = 10
        else:
            run_dirs = create_grouped_experiment_run_dir(
                runs_base=runs_base,
                algorithm=algorithm,
                mission=mission,
                config_path=config_path,
                seed=args.seed,
                experiment_id=args.experiment_id,
            )
            print("🆕 Fresh grouped experiment run:")

        if "experiment_dir" in run_dirs:
            print(f"📁 Experiment directory: {run_dirs['experiment_dir']}")

        print(f"📁 Run directory: {run_dirs['run_dir']}")
        
        # Save config and git info only for fresh runs (avoid overwriting on resume)
        if not checkpoint:
            save_config_copy(config, run_dirs['cfg_path'])
            save_git_info(run_dirs['git_path'])

            

        else:
            # If you still want snapshots on resume, write to *_resume timestamped files instead.
            pass

        # Train (fresh or resume)
        # model = train_agent(venv, config, run_dirs, checkpoint=checkpoint)

        # venv.save(vecnorm_path)
        model = train_agent(venv, config, run_dirs, checkpoint=checkpoint)

        vecnorm_path = os.path.join(run_dirs["models_dir"], "vecnormalize.pkl")
        venv.save(vecnorm_path)
        print(f"✅ VecNormalize stats saved to: {vecnorm_path}")
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