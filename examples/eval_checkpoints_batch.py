#!/usr/bin/env python3

import sys
sys.path.insert(0, "/home/pid_rl")

import argparse
import csv
import gc
import json
import re
import threading
import time
from datetime import datetime
from pathlib import Path

import subprocess

from rl_training.environments import SimGymEnv
from rl_training.utils.utils import (
    load_config,
    evaluate_agent_screening,
)

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

CHECKPOINT_STEPS = list(range(400000, 1000001, 10000))
# CHECKPOINT_STEPS = [
#     600000,
#     650000,
#     700000,
#     750000,
#     800000,
#     850000,
#     900000,
#     950000,
#     1000000,
# ]


def find_config(run_dir, experiment_dir):
    candidates = [
        run_dir / "config.yaml",
        run_dir / "cfg.yaml",
        experiment_dir / "config.yaml",
        experiment_dir / "cfg.yaml",
    ]

    for path in candidates:
        if path.exists():
            return path

    return None


def get_vecnorm_path(models_dir, algo, checkpoint):
    if checkpoint == "best":
        path = models_dir / "best_vecnormalize.pkl"
    else:
        path = (
            models_dir /
            f"{algo}_vecnormalize_{checkpoint}_steps.pkl"
        )

    return path if path.exists() else None


def get_model_path(models_dir, algo, checkpoint):
    if checkpoint == "best":
        path = models_dir / "best_model.zip"
    else:
        path = (
            models_dir /
            f"{algo}_{checkpoint}_steps.zip"
        )

    return path if path.exists() else None


def load_model(algo, model_path, env, config):
    params = config[f"{algo}_params"]
    device = params.get("device", "auto")

    if algo == "td3":
        from stable_baselines3 import TD3
        return TD3.load(
            str(model_path),
            env=env,
            device=device,
        )

    if algo == "ppo":
        from stable_baselines3 import PPO
        return PPO.load(
            str(model_path),
            env=env,
            device=device,
        )

    raise ValueError(f"Unsupported algorithm: {algo}")


def append_csv(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    write_header = not path.exists()

    with path.open("a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(row.keys()),
        )

        if write_header:
            writer.writeheader()

        writer.writerow(row)


def load_completed(summary_csv):
    summary_csv = Path(summary_csv)

    completed = set()

    if not summary_csv.exists():
        return completed

    with summary_csv.open() as f:
        reader = csv.DictReader(f)

        for row in reader:
            if row.get("status") != "ok":
                continue

            key = (
                row["experiment"],
                row["algo"],
                row["seed"],
                row["checkpoint"],
            )

            completed.add(key)

    return completed


def start_heartbeat(label, interval=30):
    """
    Print a periodic heartbeat while an evaluation is running.

    This helps distinguish a slow SITL/Gazebo startup or long episode
    from a genuinely hung process.
    """
    stop_event = threading.Event()

    def heartbeat():
        start_time = time.time()

        while not stop_event.wait(interval):
            elapsed = time.time() - start_time

            print(
                f"⏳ {datetime.now().strftime('%H:%M:%S')} | "
                f"Still working on {label} | "
                f"elapsed: {elapsed / 60:.1f} min",
                flush=True,
            )

    thread = threading.Thread(
        target=heartbeat,
        daemon=True,
    )
    thread.start()

    return stop_event


def discover_runs(runs_root, rewards=None, algos=None, seeds=None):

    runs_root = Path(runs_root)

    discovered = []

    for experiment_dir in sorted(runs_root.glob("exp_R*_seed*")):

        match = re.search(
            r"exp_(R[1-5])_.*_seed(\d+)",
            experiment_dir.name,
        )

        if not match:
            continue

        reward = match.group(1)
        seed = match.group(2)

        if rewards and reward not in rewards:
            continue

        if seeds and seed not in seeds:
            continue

        # Find every models directory below the experiment.
        # This also supports folders such as td3_2026xxxx.
        for models_dir in experiment_dir.rglob("models"):

            run_dir = models_dir.parent

            config_path = find_config(
                run_dir,
                experiment_dir,
            )

            if config_path is None:
                print(
                    f"⚠️ Config not found for {run_dir}"
                )
                continue

            config = load_config(str(config_path))

            algo = (
                config.get("training_config", {})
                .get("algo")
            )

            # Fallback to directory/model names.
            if algo not in ("td3", "ppo"):
                if list(models_dir.glob("td3_*_steps.zip")):
                    algo = "td3"
                elif list(models_dir.glob("ppo_*_steps.zip")):
                    algo = "ppo"
                elif run_dir.name.startswith("td3"):
                    algo = "td3"
                elif run_dir.name.startswith("ppo"):
                    algo = "ppo"
                else:
                    continue

            if algos and algo not in algos:
                continue

            discovered.append(
                {
                    "experiment": experiment_dir.name,
                    "reward": reward,
                    "seed": seed,
                    "algo": algo,
                    "experiment_dir": experiment_dir,
                    "run_dir": run_dir,
                    "models_dir": models_dir,
                    "config_path": config_path,
                }
            )

    return discovered


def evaluate_checkpoint(
    run,
    checkpoint,
    num_episodes,
    speedup,
    gui,
    eval_seeds,
):

    algo = run["algo"]
    models_dir = run["models_dir"]

    model_path = get_model_path(
        models_dir,
        algo,
        checkpoint,
    )

    if model_path is None:
        return None, "missing_model"

    vecnorm_path = get_vecnorm_path(
        models_dir,
        algo,
        checkpoint,
    )

    # We deliberately do not use final VecNormalize
    # statistics for an intermediate checkpoint.
    if vecnorm_path is None:
        return None, "missing_vecnormalize"

    config = load_config(
        str(run["config_path"])
    )

    config["sitl_config"]["speedup"] = speedup
    config["gazebo_config"]["gui"] = gui

    def make_env():
        env = SimGymEnv(config)
        return Monitor(env)

    dummy_env = DummyVecEnv([make_env])

    env = None
    model = None

    try:
        env = VecNormalize.load(
            str(vecnorm_path),
            dummy_env,
        )

        env.training = False
        env.norm_reward = False

        model = load_model(
            algo,
            model_path,
            env,
            config,
        )

        gamma = config[f"{algo}_params"].get(
            "gamma",
            0.99,
        )

        results = evaluate_agent_screening(
            model,
            env,
            num_episodes,
            gamma=gamma,
            verbose=True,
            episode_seeds=eval_seeds,
            progress_every=25,
        )

        return results, "ok"

    finally:
        if env is not None:
            env.close()
        else:
            dummy_env.close()

        del model
        del env
        gc.collect()


def cleanup_simulator_processes():
    """
    Clean leftover ArduPilot SITL and Gazebo processes between
    checkpoint evaluations.
    """

    print(
        "🧹 Cleaning simulator processes...",
        flush=True,
    )

    patterns = [
        "/Tools/autotest/sim_vehicle.py",
        "/build/sitl/bin/arducopter",
        "^gz sim ",
    ]

    # First try graceful termination
    for pattern in patterns:
        subprocess.run(
            ["pkill", "-TERM", "-f", pattern],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )

    time.sleep(3)

    # Kill anything that did not terminate
    for pattern in patterns:
        subprocess.run(
            ["pkill", "-KILL", "-f", pattern],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )

    time.sleep(2)

    print(
        "✅ Simulator cleanup completed.",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--runs_root",
        default="/home/pid_rl/rl_training/runs/hover",
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--speedup",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--gui",
        action="store_true",
    )

    parser.add_argument(
        "--rewards",
        nargs="*",
        default=None,
        help="Example: --rewards R1 R2",
    )

    parser.add_argument(
        "--algos",
        nargs="*",
        default=None,
        choices=["td3", "ppo"],
    )

    parser.add_argument(
        "--seeds",
        nargs="*",
        default=None,
        help="Training seeds, e.g. 101 202",
    )

    parser.add_argument(
        "--output_dir",
        default="/home/pid_rl/rl_training/evaluation_results",
    )

    parser.add_argument(
        "--cooldown",
        type=int,
        default=5,
        help="Seconds between checkpoint evaluations.",
    )

    parser.add_argument(
        "--heartbeat_interval",
        type=int,
        default=30,
        help="Seconds between progress heartbeat messages.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_csv = output_dir / "checkpoint_summary.csv"
    episodes_csv = output_dir / "checkpoint_episodes.csv"

    completed = (
        set()
        if args.overwrite
        else load_completed(summary_csv)
    )

    runs = discover_runs(
        args.runs_root,
        rewards=args.rewards,
        algos=args.algos,
        seeds=args.seeds,
    )

    print()
    print(f"Discovered {len(runs)} training runs.")

    # Same evaluation seeds for every checkpoint.
    eval_seeds = [
        1001,
        1002,
        1003,
        1004,
        1005,
    ]

    if args.episodes > len(eval_seeds):
        eval_seeds = list(
            range(
                1001,
                1001 + args.episodes,
            )
        )
    else:
        eval_seeds = eval_seeds[:args.episodes]

    checkpoints = CHECKPOINT_STEPS + ["best"]

    total_jobs = len(runs) * len(checkpoints)
    current_job = 0
    batch_start = time.time()

    print()
    print("=" * 80)
    print("CHECKPOINT EVALUATION PLAN")
    print("=" * 80)
    print(f"Training runs discovered:       {len(runs)}")
    print(f"Checkpoints per run:            {len(checkpoints)}")
    print(f"Total candidate evaluations:    {total_jobs}")
    print(f"Evaluation episodes/checkpoint: {args.episodes}")
    print(f"Evaluation seeds:               {eval_seeds}")
    print(f"Heartbeat interval:             {args.heartbeat_interval} sec")
    print("=" * 80)
    print()

    for run in runs:

        print()
        print("=" * 80)
        print(
            f"{run['experiment']} | "
            f"{run['algo'].upper()} | "
            f"seed {run['seed']}"
        )
        print("=" * 80)

        for checkpoint in checkpoints:

            current_job += 1
            checkpoint_label = str(checkpoint)

            key = (
                run["experiment"],
                run["algo"],
                str(run["seed"]),
                checkpoint_label,
            )

            if key in completed:
                print(
                    f"⏭️ [{current_job}/{total_jobs}] "
                    f"Skipping already evaluated: "
                    f"{run['reward']} | "
                    f"{run['algo'].upper()} | "
                    f"seed {run['seed']} | "
                    f"checkpoint {checkpoint_label}"
                )
                continue

            print()
            print("-" * 80)
            print(
                f"▶ [{current_job}/{total_jobs}] "
                f"{run['reward']} | "
                f"{run['algo'].upper()} | "
                f"training seed {run['seed']} | "
                f"checkpoint {checkpoint_label}"
            )
            print(
                f"Started at: "
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            print("-" * 80)

            heartbeat_label = (
                f"{run['reward']} / "
                f"{run['algo'].upper()} / "
                f"seed {run['seed']} / "
                f"checkpoint {checkpoint_label}"
            )

            heartbeat_stop = start_heartbeat(
                heartbeat_label,
                interval=args.heartbeat_interval,
            )

            job_start = time.time()

            try:
                # Make sure this checkpoint starts with a clean simulator.
                cleanup_simulator_processes()

                results, status = evaluate_checkpoint(
                    run=run,
                    checkpoint=checkpoint,
                    num_episodes=args.episodes,
                    speedup=args.speedup,
                    gui=args.gui,
                    eval_seeds=eval_seeds,
                )

                elapsed = time.time() - job_start

                if status != "ok":
                    print(
                        f"⚠️ [{current_job}/{total_jobs}] "
                        f"{checkpoint_label}: {status}"
                    )
                    print(
                        f"⏱️ Evaluation time: "
                        f"{elapsed / 60:.2f} min"
                    )
                    continue

                summary = results["summary"]

                summary_row = {
                    "experiment": run["experiment"],
                    "reward": run["reward"],
                    "algo": run["algo"],
                    "seed": run["seed"],
                    "checkpoint": checkpoint_label,

                    "episodes": summary["num_episodes"],

                    "avg_reward": summary["avg_reward"],
                    "std_reward": summary["std_reward"],

                    "avg_length": summary["avg_length"],
                    "std_length": summary["std_length"],

                    "z_mae": summary["z_mae"],
                    "x_mae": summary["x_mae"],
                    "y_mae": summary["y_mae"],
                    "xy_mae": summary["xy_mae"],

                    "z_rmse": summary["z_rmse"],
                    "x_rmse": summary["x_rmse"],
                    "y_rmse": summary["y_rmse"],
                    "xy_rmse": summary["xy_rmse"],

                    "max_z_error": summary["max_z_error"],
                    "max_x_error": summary["max_x_error"],
                    "max_y_error": summary["max_y_error"],
                    "max_xy_error": summary["max_xy_error"],

                    "termination_reasons": json.dumps(
                        summary["termination_reasons"]
                    ),

                    "status": "ok",
                }

                append_csv(
                    summary_csv,
                    summary_row,
                )

                for episode_result in results["episodes"]:

                    episode_row = {
                        "experiment": run["experiment"],
                        "reward": run["reward"],
                        "algo": run["algo"],
                        "seed": run["seed"],
                        "checkpoint": checkpoint_label,

                        **episode_result,
                    }

                    append_csv(
                        episodes_csv,
                        episode_row,
                    )

                completed.add(key)

                print(
                    f"✅ [{current_job}/{total_jobs}] "
                    f"Z MAE={summary['z_mae']:.4f} | "
                    f"XY MAE={summary['xy_mae']:.4f} | "
                    f"Reward={summary['avg_reward']:.2f}"
                )

                print(
                    f"⏱️ Checkpoint evaluation time: "
                    f"{elapsed / 60:.2f} min"
                )

                overall_elapsed = time.time() - batch_start
                print(
                    f"🕒 Total batch elapsed: "
                    f"{overall_elapsed / 60:.1f} min"
                )

            except Exception as exc:

                elapsed = time.time() - job_start

                print(
                    f"❌ [{current_job}/{total_jobs}] "
                    f"Evaluation failed: {exc}"
                )

                print(
                    f"⏱️ Failed after: "
                    f"{elapsed / 60:.2f} min"
                )

            finally:
                heartbeat_stop.set()

                # Ensure SITL/Gazebo from this checkpoint cannot
                # interfere with the next checkpoint.
                cleanup_simulator_processes()

            time.sleep(args.cooldown)

    batch_elapsed = time.time() - batch_start

    print()
    print("=" * 80)
    print("EVALUATION COMPLETED")
    print("=" * 80)
    print(
        f"Total elapsed time: "
        f"{batch_elapsed / 60:.1f} min "
        f"({batch_elapsed / 3600:.2f} h)"
    )
    print(f"Summary:  {summary_csv}")
    print(f"Episodes: {episodes_csv}")
    print("=" * 80)


if __name__ == "__main__":
    main()
