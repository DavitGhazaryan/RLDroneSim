import os
import yaml
from pathlib import Path
import numpy as np
# from email.mime.text import MIMEText
# import smtplib
import re

ARDUPILOT_DIR = '/home/student/Dev/ardupilot'

def load_config(config_path):
    """Load configuration from YAML file."""
    if not config_path or not Path(config_path).exists():
        print(f"⚠️  Config file not found: {config_path}")
        print("Using default configuration...")
        return get_default_config()
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"❌ Error loading config file: {e}")
        print("Using default configuration...")
        return get_default_config()

def get_default_config():
    """Return default Gazebo configuration."""
    return {
        'ardupilot_config': {
            'ardupilot_path': ARDUPILOT_DIR,
            'vehicle': 'ArduCopter',
            'frame': 'gazebo-iris',
            'model': 'JSON',
            'timeout': 60.0
        },
        'gazebo_config': {
            'sdf_file': '/home/student/Dev/pid_rl/ardupilot_gazebo/worlds/simple_world.sdf',
            'gui': 'DISPLAY' in os.environ,
            'verbose': True,
            'timeout': 15.0
        }
    }

def euler_to_quaternion(euler):
    """
    Convert Euler angles (yaw, pitch, roll) to a quaternion.
    Angles are expected in radians.
    """
    # roll, pitch, yaw = euler['roll_deg'], euler['pitch_deg'], euler['yaw_deg']
    roll, pitch, yaw = 0.0, 0.0, 90.0
    roll, pitch, yaw = np.deg2rad([roll, pitch, yaw])
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    return [qx, qy, qz, qw]

def demonstrate_observation_action_format(env):
    
    print("\n🔍 Demonstrating Observation and Action Format")
    print("=" * 60)
    
    # Show sample observations and actions
    sample_obs = env.observation_space.sample()
    sample_action = env.action_space.sample()
    
    print("📊 Sample Observation (Array):")
    print(f"   Shape: {sample_obs.shape}")
    print(f"   Values: {sample_obs}")
    
    print("\n🎯 Sample Action (Array):")
    print(f"   Shape: {sample_action.shape}")
    print(f"   Values: {sample_action}")
    
    # Show what each index represents
    obs_mapping = env.get_observation_key_mapping()
    action_mapping = env.get_action_key_mapping()
    
    print("\n🗺️  Observation Index Meaning:")
    for key, idx in obs_mapping.items():
        print(f"   obs[{idx}] = {key} = {sample_obs[idx]:.3f}")
    
    print("\n🎯 Action Index Meaning:")
    for key, idx in action_mapping.items():
        print(f"   action[{idx}] = {key} adjustment = {sample_action[idx]:.3f}")

# def evaluate_agent(model, env, num_episodes, gamma=0.99, verbose=False):
#     """
#     Evaluate the trained agent.
    
#     Args:
#         model: Trained model, if none then evaluate the baseline fixed PID
#         env: Modified ArdupilotEnv
#         num_episodes: Number of evaluation episodes
#         gamma: Discount factor
        
#     Returns:
#         Evaluation results
#     """

#     episode_rewards = []
#     episode_lengths = []

#     episode_z_errors = []
#     episode_x_errors = []
#     episode_y_errors = []
    
#     obs = env.reset()
    
#     for episode in range(num_episodes):       

#         episode_return = 0.0
#         episode_discounted_return = 0.0
#         episode_length = 0

#         sum_z_error = 0.0
#         sum_x_error = 0.0
#         sum_y_error = 0.0

#         print()
#         print(f"Episode {episode + 1}:")

#         while True:
#             if model:
#                 action, _ = model.predict(obs, deterministic=True)
#             else:
#                 action = env.action_space.sample()  
#                 action = action * 0
#                 action = [action]

#             obs, reward, done, info = env.step(action)

#             # Errors are already calculated in BaseEnv.step()
#             # and passed through the info dictionary.
#             alt_err = info[0].get("alt_err", 0.0)
#             x_err = info[0].get("x_err", 0.0)
#             y_err = info[0].get("y_err", 0.0)

#             sum_z_error += abs(alt_err)
#             sum_x_error += abs(x_err)
#             sum_y_error += abs(y_err)
            
#             episode_length += 1            
#             episode_return += reward
#             episode_discounted_return += (gamma ** episode_length) * reward
            
#             if done and info[0]['reason']:
#                 print(f"    {info[0]['reason']}")
#                 break
        
#         episode_rewards.append(episode_return)
#         episode_lengths.append(episode_length)

#         episode_z_errors.append(sum_z_error)
#         episode_x_errors.append(sum_x_error)
#         episode_y_errors.append(sum_y_error)

#         print()
#         print(
#             f"    Return: {float(episode_return):.2f}, "
#             f"Discounted: {float(episode_discounted_return):.2f}, "
#             f"Length: {int(episode_length)}"
#         )
#         print(f"    Sum Z error: {sum_z_error:.4f}")
#         print(f"    Sum X error: {sum_x_error:.4f}")
#         print(f"    Sum Y error: {sum_y_error:.4f}")

#     avg_reward = np.mean(episode_rewards)
#     std_reward = np.std(episode_rewards)
#     avg_length = np.mean(episode_lengths)

#     avg_z_error = np.mean(episode_z_errors)
#     avg_x_error = np.mean(episode_x_errors)
#     avg_y_error = np.mean(episode_y_errors)
    
#     print("\n📊 Evaluation Results:")
#     print(f"   Average reward: {avg_reward:.2f} ± {std_reward:.2f}")
#     print(f"   Average episode length: {avg_length:.1f} steps")
#     print(f"   Success rate: {sum(1 for r in episode_rewards if r > 0) / len(episode_rewards):.1%}")

#     print("\n📍 Error Summary:")
#     print(f"   Average Z error: {avg_z_error:.4f}")
#     print(f"   Average X error: {avg_x_error:.4f}")
#     print(f"   Average Y error: {avg_y_error:.4f}")

def evaluate_agent(
    model,
    env,
    num_episodes,
    gamma=0.99,
    verbose=False,
    log_pid_gains=False,
    gain_keys=None,
    save_dir=None,
):
    """
    Evaluate the trained agent.

    Args:
        model: Trained model, if None then evaluate the baseline fixed PID.
        env: Modified ArdupilotEnv / VecEnv.
        num_episodes: Number of evaluation episodes.
        gamma: Discount factor.
        verbose: Whether to print extra details.
        log_pid_gains: If True, save PID gain/action logs and plots during evaluation.
        gain_keys: List of PID gain names to log.
        save_dir: Directory where evaluation logs/plots will be saved.

    Returns:
        Evaluation results.
    """
    import os
    import numpy as np

    if log_pid_gains:
        import pandas as pd
        import matplotlib.pyplot as plt

        if gain_keys is None:
            gain_keys = []

        if save_dir is None:
            save_dir = "evaluation_logs"

        os.makedirs(save_dir, exist_ok=True)

    episode_rewards = []
    episode_lengths = []

    episode_z_errors = []
    episode_x_errors = []
    episode_y_errors = []

    # New: store per-step evaluation logs
    eval_logs = []

    obs = env.reset()

    for episode in range(num_episodes):

        episode_return = 0.0
        episode_discounted_return = 0.0
        episode_length = 0

        sum_z_error = 0.0
        sum_x_error = 0.0
        sum_y_error = 0.0

        print()
        print(f"Episode {episode + 1}:")

        while True:
            if model:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = env.action_space.sample()
                action = action * 0
                action = [action]

            obs, reward, done, info = env.step(action)

            # VecEnv returns lists/arrays
            info0 = info[0]

            # Errors are already calculated in BaseEnv.step()
            # and passed through the info dictionary.
            alt_err = info0.get("alt_err", 0.0)
            x_err = info0.get("x_err", 0.0)
            y_err = info0.get("y_err", 0.0)

            sum_z_error += abs(alt_err)
            sum_x_error += abs(x_err)
            sum_y_error += abs(y_err)

            episode_length += 1
            episode_return += reward
            episode_discounted_return += (gamma ** episode_length) * reward

            # --------------------------------------------------------
            # New: log PID gains and actions during evaluation
            # --------------------------------------------------------
            if log_pid_gains:
                reward_value = float(np.asarray(reward).reshape(-1)[0])
                done_value = bool(np.asarray(done).reshape(-1)[0])

                row = {
                    "episode": episode,
                    "step": episode_length,
                    "reward": reward_value,
                    "done": done_value,
                    "reason": str(info0.get("reason")),
                    "alt_err": alt_err,
                    "x_err": x_err,
                    "y_err": y_err,
                }

                # Current PID gain values from info dictionary
                for gain in gain_keys:
                    row[gain] = info0.get(gain)

                # Policy action values
                action_flat = np.asarray(action).reshape(-1)
                for i, gain in enumerate(gain_keys):
                    if i < len(action_flat):
                        row[f"action_{gain}"] = float(action_flat[i])

                eval_logs.append(row)

            if done and info0["reason"]:
                print(f"    {info0['reason']}")
                break

        episode_rewards.append(episode_return)
        episode_lengths.append(episode_length)

        episode_z_errors.append(sum_z_error)
        episode_x_errors.append(sum_x_error)
        episode_y_errors.append(sum_y_error)

        print()
        print(
            f"    Return: {float(episode_return):.2f}, "
            f"Discounted: {float(episode_discounted_return):.2f}, "
            f"Length: {int(episode_length)}"
        )
        print(f"    Sum Z error: {sum_z_error:.4f}")
        print(f"    Sum X error: {sum_x_error:.4f}")
        print(f"    Sum Y error: {sum_y_error:.4f}")

    avg_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    avg_length = np.mean(episode_lengths)

    avg_z_error = np.mean(episode_z_errors)
    avg_x_error = np.mean(episode_x_errors)
    avg_y_error = np.mean(episode_y_errors)

    print("\n📊 Evaluation Results:")
    print(f"   Average reward: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"   Average episode length: {avg_length:.1f} steps")
    print(f"   Success rate: {sum(1 for r in episode_rewards if r > 0) / len(episode_rewards):.1%}")

    print("\n📍 Error Summary:")
    print(f"   Average Z error: {avg_z_error:.4f}")
    print(f"   Average X error: {avg_x_error:.4f}")
    print(f"   Average Y error: {avg_y_error:.4f}")

    # ------------------------------------------------------------
    # New: save PID gain/action logs and plots
    # ------------------------------------------------------------
    if log_pid_gains:
        df = pd.DataFrame(eval_logs)

        csv_path = os.path.join(save_dir, "eval_pid_gains.csv")
        df.to_csv(csv_path, index=False)
        print(f"\nSaved evaluation log to: {csv_path}")

        if len(df) > 0:
            df["global_step"] = range(len(df))

            # Plot all PID gains together
            plt.figure(figsize=(12, 6))

            for gain in gain_keys:
                if gain in df.columns:
                    plt.plot(df["global_step"], df[gain], label=gain)

            plt.xlabel("Evaluation step")
            plt.ylabel("PID gain value")
            plt.title("PID gain evolution during evaluation")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()

            gains_plot_path = os.path.join(save_dir, "eval_pid_gains.png")
            plt.savefig(gains_plot_path, dpi=300)
            plt.close()

            print(f"Saved PID gain plot to: {gains_plot_path}")

            # Plot all policy actions together
            plt.figure(figsize=(12, 6))

            for gain in gain_keys:
                action_col = f"action_{gain}"
                if action_col in df.columns:
                    plt.plot(df["global_step"], df[action_col], label=action_col)

            plt.xlabel("Evaluation step")
            plt.ylabel("Action value")
            plt.title("Policy action outputs during evaluation")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()

            actions_plot_path = os.path.join(save_dir, "eval_actions.png")
            plt.savefig(actions_plot_path, dpi=300)
            plt.close()

            print(f"Saved action plot to: {actions_plot_path}")

            # Plot each PID gain separately
            separate_dir = os.path.join(save_dir, "gain_plots")
            os.makedirs(separate_dir, exist_ok=True)

            for gain in gain_keys:
                if gain not in df.columns:
                    continue

                plt.figure(figsize=(10, 4))

                for ep in sorted(df["episode"].unique()):
                    ep_df = df[df["episode"] == ep]
                    plt.plot(ep_df["step"], ep_df[gain], label=f"Episode {ep + 1}")

                plt.xlabel("Evaluation step")
                plt.ylabel(gain)
                plt.title(f"{gain} during evaluation")
                plt.legend()
                plt.grid(True)
                plt.tight_layout()

                single_gain_path = os.path.join(separate_dir, f"{gain}.png")
                plt.savefig(single_gain_path, dpi=300)
                plt.close()

            print(f"Saved separate gain plots to: {separate_dir}")

            # ------------------------------------------------------------
            # Plot Z, X, Y errors together
            # ------------------------------------------------------------
            plt.figure(figsize=(12, 6))

            if "alt_err" in df.columns:
                plt.plot(df["global_step"], df["alt_err"].abs(), label="Z error")

            if "x_err" in df.columns:
                plt.plot(df["global_step"], df["x_err"].abs(), label="X error")

            if "y_err" in df.columns:
                plt.plot(df["global_step"], df["y_err"].abs(), label="Y error")

            plt.xlabel("Evaluation step")
            plt.ylabel("Absolute error")
            plt.title("Z, X, Y errors during evaluation")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()

            errors_plot_path = os.path.join(save_dir, "eval_errors_xyz.png")
            plt.savefig(errors_plot_path, dpi=300)
            plt.close()

            print(f"Saved error plot to: {errors_plot_path}")

            # ------------------------------------------------------------
            # Plot Z, X, Y errors separately per episode
            # ------------------------------------------------------------
            error_dir = os.path.join(save_dir, "error_plots")
            os.makedirs(error_dir, exist_ok=True)

            for err_key, label in [
                ("alt_err", "Z error"),
                ("x_err", "X error"),
                ("y_err", "Y error"),
            ]:
                if err_key not in df.columns:
                    continue

                plt.figure(figsize=(10, 4))

                for ep in sorted(df["episode"].unique()):
                    ep_df = df[df["episode"] == ep]
                    plt.plot(ep_df["step"], ep_df[err_key].abs(), label=f"Episode {ep + 1}")

                plt.xlabel("Evaluation step")
                plt.ylabel(f"Absolute {label}")
                plt.title(f"{label} during evaluation")
                plt.legend()
                plt.grid(True)
                plt.tight_layout()

                err_plot_path = os.path.join(error_dir, f"{err_key}.png")
                plt.savefig(err_plot_path, dpi=300)
                plt.close()

            print(f"Saved separate error plots to: {error_dir}")

    return {
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "episode_z_errors": episode_z_errors,
        "episode_x_errors": episode_x_errors,
        "episode_y_errors": episode_y_errors,
        "avg_reward": avg_reward,
        "std_reward": std_reward,
        "avg_length": avg_length,
        "avg_z_error": avg_z_error,
        "avg_x_error": avg_x_error,
        "avg_y_error": avg_y_error,
    }
import datetime
import json
import subprocess


def validate_config(config, model):
    """
    Validate that the configuration contains all required parameters.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        bool: True if config is valid, False otherwise
    """
    if model != "ddpg":
        raise NotImplementedError("Only ddpg validation is implemented")

    required_sections = ['environment_config', 'ardupilot_config', 'gazebo_config', 'ddpg_params', 'training_config']
    
    for section in required_sections:
        if section not in config:
            print(f"❌ Missing required configuration section: {section}")
            return False
    
    # Check DDPG parameters
    ddpg_params = config['ddpg_params']
    required_ddpg_params = ['learning_rate', 'buffer_size', 'batch_size', 'tau', 'gamma']
    
    for param in required_ddpg_params:
        if param not in ddpg_params:
            print(f"❌ Missing required DDPG parameter: {param}")
            return False
    
    # Check training parameters
    training_params = config['training_config']
    required_training_params = ['total_timesteps', 'save_freq']
    
    for param in required_training_params:
        if param not in training_params:
            print(f"❌ Missing required training parameter: {param}")
            return False
    
    print("✅ Configuration validation passed!")
    return True


def _safe_mkdir(path: str, mode: int = 0o777) -> str:
    os.makedirs(path, exist_ok=True, mode=mode)
    return path

def _find_run_root(p: Path) -> Path | None:
    """
    Walk up until we find a folder containing 'tb' and/or 'models'.
    Accepts: run root, 'tb', 'models', a model .zip, a replay .pkl, or an events file.
    """
    p = p if p.is_dir() else p.parent
    for parent in [p, *p.parents]:
        tb = parent / "tb"
        models = parent / "models"
        if tb.exists() or models.exists():
            return parent
    return None

def create_run_dir(base_dir: str, algo: str, mission: str,
                   exp_name: str | None = None,
                   resume_from: str | None = None) -> dict:
    """
    Create or REUSE a structured run directory tree:
      runs/<mission>/<algo>/<YYYYMMDD_HHMMSS>_{exp_name}/ with subdirs tb/ and models/

    If `resume_from` is provided (can be a run folder, tb/, models/, a .zip, .pkl,
    or a TensorBoard event file), we *reuse* that run (no new timestamp).
    """
    if resume_from:
        cand = Path(resume_from).resolve()
        run_root = _find_run_root(cand)
        if run_root is None:
            raise FileNotFoundError(f"Could not locate run root for: {resume_from}")
        # Ensure expected subfolders exist
        tb_dir = (run_root / "tb")
        models_dir = (run_root / "models")
        _safe_mkdir(tb_dir.as_posix(), mode=0o777)
        _safe_mkdir(models_dir.as_posix(), mode=0o777)
        return {
            'run_dir': run_root.as_posix(),
            'tb_dir': tb_dir.as_posix(),
            'models_dir': models_dir.as_posix(),
            'cfg_path': (run_root / 'cfg.yaml').as_posix(),
            'git_path': (run_root / 'git.txt').as_posix(),
            'metrics_path': (run_root / 'metrics.json').as_posix(),
        }

    # fresh run
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{exp_name}" if exp_name else ""
    run_dir = os.path.join(base_dir, mission, algo, f"{stamp}{suffix}")
    tb_dir = os.path.join(run_dir, "tb")
    models_dir = os.path.join(run_dir, "models")
    _safe_mkdir(tb_dir, mode=0o777)
    _safe_mkdir(models_dir, mode=0o777)
    return {
        'run_dir': run_dir,
        'tb_dir': tb_dir,
        'models_dir': models_dir,
        'cfg_path': os.path.join(run_dir, 'cfg.yaml'),
        'git_path': os.path.join(run_dir, 'git.txt'),
        'metrics_path': os.path.join(run_dir, 'metrics.json'),
    }
essential_config_keys = [
    'environment_config', 'ardupilot_config', 'gazebo_config', 'ddpg_params', 'training_config',
    'evaluation_config', 'callbacks',
]

def save_config_copy(config: dict, cfg_path: str) -> None:
    """Save a trimmed copy of the config to cfg.yaml in the run dir."""
    try:
        # Preserve only essential top-level keys if present
        trimmed = {k: config.get(k) for k in config}
        with open(cfg_path, 'w') as f:
            yaml.safe_dump(trimmed if trimmed else config, f, sort_keys=False)
        os.chmod(cfg_path, 0o777)
    except Exception as exc:
        print(f"⚠️ Could not write config copy to {cfg_path}: {exc}")

def save_git_info(git_path: str) -> None:
    """Write current git commit, branch, and dirty flag to git.txt if inside a git repo."""
    try:
        # Determine if repo
        subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).decode().strip()
        status = subprocess.check_output(["git", "status", "--porcelain"]).decode()
        dirty = "dirty" if status.strip() else "clean"
        with open(git_path, 'w') as f:
            f.write(f"commit: {commit}\n")
            f.write(f"branch: {branch}\n")
            f.write(f"state: {dirty}\n")
        os.chmod(git_path, 0o777)
    except Exception as exc:
        try:
            with open(git_path, 'w') as f:
                f.write(f"git: unavailable ({exc})\n")
        except Exception as exc2:
            print(f"⚠️ Could not write git info to {git_path}: {exc2}")

def save_metrics_json(metrics: dict, metrics_path: str) -> None:
    """Save final evaluation summary as JSON."""
    try:
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        os.chmod(metrics_path, 0o777)
    except Exception as exc:
        print(f"⚠️ Could not write metrics to {metrics_path}: {exc}")

def huber(e, delta):
    a = abs(e)
    return 0.5*(a**2) if a <= delta else delta*(a - 0.5*delta)
            

def nrm(e, tau):
    return min(abs(e)/tau, 10.0)


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
def _extract_steps(path_like: str):
    """
    Accepts a specific .zip path.
    Returns integer steps or None if not found.
    Example: td3_1000_steps.zip -> 1000
    """
    if not path_like or not os.path.isfile(path_like) or not path_like.endswith(".zip"):
        return None
    m = re.search(r"_(\d+)_steps\.zip$", os.path.basename(path_like))
    return int(m.group(1)) if m else None


def _extract_prefix(path_like: str):
    """
    Accepts a specific .zip path.
    Returns the prefix before the first underscore.
    Example: td3_1000_steps.zip -> "td3"
    """
    if not path_like:
        return None
    base = os.path.basename(path_like)
    m = re.match(r"^([^_]+)_", base)
    return m.group(1) if m else None

def _replay_for(model_zip_path: str):
    """
    Finds the replay buffer file that matches the given model steps.
    Expects: <dir>/<name_prefix>_replay_buffer_<steps>_steps.pkl
    """
    steps = _extract_steps(model_zip_path)
    prefix = _extract_prefix(model_zip_path)
    if model_zip_path is None or steps is None or prefix is None:
        return None
    directory = os.path.dirname(model_zip_path)
    rb = os.path.join(directory, f"{prefix}_replay_buffer_{steps}_steps.pkl")
    return rb if os.path.exists(rb) else None

def _vecnormalize_for(model_zip_path: str):
    """
    Finds the replay buffer file that matches the given model steps.
    Expects: <dir>/<name_prefix>_replay_buffer_<steps>_steps.pkl
    """
    steps = _extract_steps(model_zip_path)
    prefix = _extract_prefix(model_zip_path)
    if model_zip_path is None or steps is None or prefix is None:
        return None
    directory = os.path.dirname(model_zip_path)
    rb = os.path.join(directory, f"{prefix}_vecnormalize_{steps}_steps.pkl")
    return rb if os.path.exists(rb) else None


def _get_config(path_like: str):
    """
    Given a model path like:
    /home/pid_rl/rl_training/runs/hover/td3/20251018_072859/models/td3_1000_steps.zip

    Returns the corresponding config path:
    /home/pid_rl/rl_training/runs/hover/td3/20251018_072859/cfg.yaml
    """
    if not path_like:
        return None
    base_dir = os.path.dirname(path_like)  # .../models
    parent_dir = os.path.dirname(base_dir) # .../<timestamp>
    cfg_path = os.path.join(parent_dir, "cfg.yaml")
    return cfg_path

