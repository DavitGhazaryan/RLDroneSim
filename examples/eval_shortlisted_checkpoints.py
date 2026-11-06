import json
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from stable_baselines3 import PPO, TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from rl_training.environments import SimGymEnv
from rl_training.utils.utils import (
    load_config,
    evaluate_agent_screening,
)


# ============================================================
# Configuration
# ============================================================

SHORTLIST_PATH = Path(
    "/home/pid_rl/rl_training/evaluation_results/"
    "all_experiments/analysis/checkpoint_shortlist.csv"
)

RUNS_ROOT = Path(
    "/home/pid_rl/rl_training/runs/hover"
)

OUTPUT_DIR = Path(
    "/home/pid_rl/rl_training/evaluation_results/final_50ep"
)

SUMMARY_PATH = OUTPUT_DIR / "final_50ep_summary.csv"
EPISODES_PATH = OUTPUT_DIR / "final_50ep_episodes.csv"

NUM_EPISODES = 50

EVAL_SEEDS = list(range(1001, 1051))

SPEEDUP = 20
GUI = False


MODEL_CLASSES = {
    "ppo": PPO,
    "td3": TD3,
}


# ============================================================
# Helpers
# ============================================================

def normalize_checkpoint(value):

    value = str(value).strip()

    if value.lower() == "best":
        return "best"

    return str(int(float(value)))


def find_config(run_dir):

    candidates = [
        run_dir / "cfg.yaml",
        run_dir / "config.yaml",
        run_dir.parent / "cfg.yaml",
        run_dir.parent / "config.yaml",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        f"No config found for {run_dir}"
    )


def get_model_path(run_dir, algo, checkpoint):

    models_dir = run_dir / "models"

    if checkpoint == "best":
        return models_dir / "best_model.zip"

    return (
        models_dir
        / f"{algo}_{checkpoint}_steps.zip"
    )


def get_vecnorm_path(run_dir, algo, checkpoint):

    models_dir = run_dir / "models"

    if checkpoint == "best":
        return models_dir / "best_vecnormalize.pkl"

    return (
        models_dir
        / f"{algo}_vecnormalize_"
          f"{checkpoint}_steps.pkl"
    )


def cleanup_simulator_processes():

    patterns = [
        "/Tools/autotest/sim_vehicle.py",
        "/build/sitl/bin/arducopter",
        "^gz sim ",
    ]

    for pattern in patterns:

        subprocess.run(
            [
                "pkill",
                "-TERM",
                "-f",
                pattern,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )

    time.sleep(2)

    for pattern in patterns:

        subprocess.run(
            [
                "pkill",
                "-KILL",
                "-f",
                pattern,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )

    time.sleep(2)


def start_heartbeat(
    stop_event,
    description,
    start_time,
):

    def heartbeat():

        while not stop_event.wait(60):

            elapsed = (
                time.time() - start_time
            ) / 60.0

            now = datetime.now().strftime(
                "%H:%M:%S"
            )

            print(
                f"⏳ {now} | "
                f"Still evaluating {description} | "
                f"elapsed: {elapsed:.1f} min",
                flush=True,
            )

    thread = threading.Thread(
        target=heartbeat,
        daemon=True,
    )

    thread.start()

    return thread


def load_completed():

    if not SUMMARY_PATH.exists():
        return set()

    df = pd.read_csv(SUMMARY_PATH)

    completed = df[
        df["status"] == "ok"
    ].copy()

    result = set()

    for _, row in completed.iterrows():

        result.add(
            (
                row["experiment"],
                row["algo"],
                int(row["seed"]),
                normalize_checkpoint(
                    row["checkpoint"]
                ),
            )
        )

    return result


def append_dataframe(df, path):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_header = not path.exists()

    df.to_csv(
        path,
        mode="a",
        header=write_header,
        index=False,
    )


# ============================================================
# Evaluate one candidate
# ============================================================

def evaluate_candidate(row):

    experiment = row["experiment"]
    reward = row["reward"]
    algo = row["algo"].lower()
    training_seed = int(row["seed"])

    checkpoint = normalize_checkpoint(
        row["checkpoint"]
    )

    selection_reason = row[
        "selection_reason"
    ]

    run_dir = (
        RUNS_ROOT
        / experiment
        / algo
    )

    config_path = find_config(run_dir)

    model_path = get_model_path(
        run_dir,
        algo,
        checkpoint,
    )

    vecnorm_path = get_vecnorm_path(
        run_dir,
        algo,
        checkpoint,
    )

    if not model_path.exists():

        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    if not vecnorm_path.exists():

        raise FileNotFoundError(
            f"VecNormalize not found: "
            f"{vecnorm_path}"
        )

    config = load_config(
        str(config_path)
    )

    config["sitl_config"]["speedup"] = (
        SPEEDUP
    )

    config["gazebo_config"]["gui"] = (
        GUI
    )

    def make_env():

        return Monitor(
            SimGymEnv(config)
        )

    env = DummyVecEnv(
        [make_env]
    )

    try:

        env = VecNormalize.load(
            str(vecnorm_path),
            env,
        )

        env.training = False
        env.norm_reward = False

        model_class = MODEL_CLASSES[algo]

        model = model_class.load(
            str(model_path),
            env=env,
        )

        results = evaluate_agent_screening(
            model=model,
            env=env,
            num_episodes=NUM_EPISODES,
            verbose=True,
            episode_seeds=EVAL_SEEDS,
            progress_every=50,
        )

        return results

    finally:

        try:
            env.close()
        except Exception:
            pass


# ============================================================
# Main
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    shortlist = pd.read_csv(
        SHORTLIST_PATH
    )

    print(
        f"Loaded {len(shortlist)} "
        f"shortlisted candidates."
    )

    print(
        f"Evaluation episodes per candidate: "
        f"{NUM_EPISODES}"
    )

    print(
        f"Total planned episodes: "
        f"{len(shortlist) * NUM_EPISODES}"
    )

    completed = load_completed()

    print(
        f"Already completed candidates: "
        f"{len(completed)}"
    )

    total_candidates = len(shortlist)

    batch_start = time.time()


    for index, row in shortlist.iterrows():

        experiment = row["experiment"]
        reward = row["reward"]
        algo = row["algo"].lower()
        training_seed = int(row["seed"])

        checkpoint = normalize_checkpoint(
            row["checkpoint"]
        )

        selection_reason = row[
            "selection_reason"
        ]

        key = (
            experiment,
            algo,
            training_seed,
            checkpoint,
        )

        current = index + 1


        # ----------------------------------------------------
        # Resume support
        # ----------------------------------------------------

        if key in completed:

            print(
                f"⏭️ [{current}/{total_candidates}] "
                f"Skipping completed: "
                f"{reward} | {algo.upper()} | "
                f"seed {training_seed} | "
                f"{checkpoint}",
                flush=True,
            )

            continue


        description = (
            f"{reward} / {algo.upper()} / "
            f"seed {training_seed} / "
            f"checkpoint {checkpoint}"
        )


        print()
        print("=" * 80)
        print(
            f"▶ [{current}/{total_candidates}] "
            f"{description}"
        )
        print(
            f"Selection: {selection_reason}"
        )
        print(
            "Started:",
            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
        )
        print("=" * 80)


        cleanup_simulator_processes()

        candidate_start = time.time()

        heartbeat_stop = threading.Event()

        heartbeat_thread = start_heartbeat(
            heartbeat_stop,
            description,
            candidate_start,
        )


        try:

            results = evaluate_candidate(
                row
            )

            episodes = pd.DataFrame(
                results["episodes"]
            )

            # -----------------------------------------------
            # Add identification columns to every episode
            # -----------------------------------------------

            episodes.insert(
                0,
                "experiment",
                experiment,
            )

            episodes.insert(
                1,
                "reward_config",
                reward,
            )

            episodes.insert(
                2,
                "algo",
                algo,
            )

            episodes.insert(
                3,
                "training_seed",
                training_seed,
            )

            episodes.insert(
                4,
                "checkpoint",
                checkpoint,
            )

            episodes.insert(
                5,
                "selection_reason",
                selection_reason,
            )

            episodes.insert(
                6,
                "eval_seed",
                EVAL_SEEDS[
                    :len(episodes)
                ],
            )


            # -----------------------------------------------
            # Create final summary
            # -----------------------------------------------

            summary_row = {
                "experiment": experiment,
                "reward": reward,
                "algo": algo,
                "seed": training_seed,
                "checkpoint": checkpoint,
                "selection_reason":
                    selection_reason,
                "episodes":
                    len(episodes),
                "status": "ok",
            }


            # -----------------------------------------------
            # Mean and standard deviation of every numeric
            # episode-level metric
            # -----------------------------------------------

            excluded_numeric = {
                "episode",
                "eval_seed",
                "training_seed",
            }

            numeric_columns = (
                episodes
                .select_dtypes(
                    include=[np.number]
                )
                .columns
            )

            for column in numeric_columns:

                if column in excluded_numeric:
                    continue

                summary_row[
                    f"{column}_mean"
                ] = float(
                    episodes[column].mean()
                )

                summary_row[
                    f"{column}_std"
                ] = float(
                    episodes[column].std(
                        ddof=1
                    )
                )


            # -----------------------------------------------
            # Preserve summary produced by evaluator too
            # -----------------------------------------------

            for key2, value in (
                results["summary"].items()
            ):

                if isinstance(
                    value,
                    (
                        int,
                        float,
                        np.integer,
                        np.floating,
                    ),
                ):

                    summary_row[
                        f"screen_{key2}"
                    ] = float(value)

                elif key2 == (
                    "termination_reasons"
                ):

                    summary_row[
                        "termination_reasons"
                    ] = json.dumps(value)


            # -----------------------------------------------
            # Termination counts from individual episodes
            # -----------------------------------------------

            reason_column = None

            for candidate_name in [
                "termination_reason",
                "reason",
            ]:

                if (
                    candidate_name
                    in episodes.columns
                ):

                    reason_column = (
                        candidate_name
                    )

                    break

            if reason_column is not None:

                counts = (
                    episodes[
                        reason_column
                    ]
                    .value_counts(
                        dropna=False
                    )
                    .to_dict()
                )

                summary_row[
                    "termination_counts"
                ] = json.dumps(counts)


            # -----------------------------------------------
            # Save immediately
            # -----------------------------------------------

            append_dataframe(
                episodes,
                EPISODES_PATH,
            )

            append_dataframe(
                pd.DataFrame(
                    [summary_row]
                ),
                SUMMARY_PATH,
            )


            elapsed = (
                time.time()
                - candidate_start
            ) / 60.0

            print(
                f"✅ Completed "
                f"{description}",
                flush=True,
            )

            print(
                f"   Episodes: "
                f"{len(episodes)}",
                flush=True,
            )

            print(
                f"   Evaluation time: "
                f"{elapsed:.2f} min",
                flush=True,
            )


        except Exception as exc:

            print(
                f"❌ Failed: "
                f"{description}",
                flush=True,
            )

            print(
                f"   {type(exc).__name__}: "
                f"{exc}",
                flush=True,
            )

            error_row = {
                "experiment": experiment,
                "reward": reward,
                "algo": algo,
                "seed": training_seed,
                "checkpoint": checkpoint,
                "selection_reason":
                    selection_reason,
                "episodes": 0,
                "status": "error",
                "error": (
                    f"{type(exc).__name__}: "
                    f"{exc}"
                ),
            }

            append_dataframe(
                pd.DataFrame(
                    [error_row]
                ),
                SUMMARY_PATH,
            )


        finally:

            heartbeat_stop.set()

            heartbeat_thread.join(
                timeout=2
            )

            cleanup_simulator_processes()


        total_elapsed = (
            time.time()
            - batch_start
        ) / 60.0

        print(
            f"🕒 Total batch elapsed: "
            f"{total_elapsed:.1f} min",
            flush=True,
        )


    print()
    print("=" * 80)
    print("FINAL 50-EPISODE EVALUATION COMPLETE")
    print("=" * 80)

    print(
        f"Summary: {SUMMARY_PATH}"
    )

    print(
        f"Episodes: {EPISODES_PATH}"
    )


if __name__ == "__main__":
    main()
