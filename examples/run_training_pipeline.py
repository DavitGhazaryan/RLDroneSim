#!/usr/bin/env python3

import argparse
import subprocess
import time
from pathlib import Path
from datetime import datetime


def build_command(algo, instance, total_timesteps, config=None):
    cmd = [
        "xvfb-run",
        "-a",
        "python3",
        "-u",
        "examples/train_vectorized.py",
        str(instance),
        "--algo",
        algo,
        "--total_timesteps",
        str(total_timesteps),
    ]

    if config is not None:
        cmd += ["--config", config]

    return cmd


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--algos",
        nargs="+",
        default=["td3", "sac", "ppo"],
        choices=["td3", "sac", "ppo", "ddpg"],
        help="Algorithms to train."
    )

    parser.add_argument(
        "--total_timesteps",
        type=int,
        default=1_000_000,
        help="Training timesteps for each algorithm."
    )

    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run algorithms in parallel. If not set, runs sequentially."
    )

    parser.add_argument(
        "--start_delay",
        type=int,
        default=20,
        help="Delay in seconds between starting parallel jobs."
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional config file path."
    )

    parser.add_argument(
        "--show_output",
        action="store_true",
        help="Show live training output/progress bars in this terminal instead of redirecting to log files."
    )

    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("rl_training/pipeline_logs") / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)

    jobs = []

    for idx, algo in enumerate(args.algos):
        instance = idx + 1
        cmd = build_command(
            algo=algo,
            instance=instance,
            total_timesteps=args.total_timesteps,
            config=args.config,
        )

        log_path = log_dir / f"{algo}_instance_{instance}.log"

        print("=" * 80)
        print(f"Starting {algo.upper()} on instance {instance}")

        if args.show_output:
            print("Live output: enabled")
        else:
            print(f"Log file: {log_path}")

        print("Command:")
        print(" ".join(cmd))
        print("=" * 80)

        if args.show_output:
            log_file = None
            stdout_target = None
            stderr_target = None
        else:
            log_file = open(log_path, "w", encoding="utf-8")
            stdout_target = log_file
            stderr_target = subprocess.STDOUT

        process = subprocess.Popen(
            cmd,
            stdout=stdout_target,
            stderr=stderr_target,
            cwd="/home/pid_rl",
            start_new_session=True,
        )

        jobs.append((algo, instance, process, log_file))

        if args.parallel:
            time.sleep(args.start_delay)
        else:
            return_code = process.wait()

            if log_file is not None:
                log_file.close()

            if return_code != 0:
                print(f"{algo.upper()} failed with return code {return_code}")
                return

            print(f"{algo.upper()} completed successfully.")

    if args.parallel:
        print("\nAll jobs started. Waiting for completion...")

        for algo, instance, process, log_file in jobs:
            return_code = process.wait()

            if log_file is not None:
                log_file.close()

            if return_code == 0:
                print(f"{algo.upper()} on instance {instance} completed successfully.")
            else:
                print(f"{algo.upper()} on instance {instance} failed with return code {return_code}.")


if __name__ == "__main__":
    main()
# #!/usr/bin/env python3

# import argparse
# import subprocess
# import time
# from pathlib import Path
# from datetime import datetime


# def build_command(algo, instance, total_timesteps, config=None):
#     cmd = [
#         "xvfb-run",
#         "-a",
#         "python3",
#         "-u",
#         "examples/train_vectorized.py",
#         str(instance),
#         "--algo",
#         algo,
#         "--total_timesteps",
#         str(total_timesteps),
#     ]

#     if config is not None:
#         cmd += ["--config", config]

#     return cmd


# def main():
#     parser = argparse.ArgumentParser()

#     parser.add_argument(
#         "--algos",
#         nargs="+",
#         default=["td3", "sac", "ppo"],
#         choices=["td3", "sac", "ppo", "ddpg"],
#         help="Algorithms to train."
#     )

#     parser.add_argument(
#         "--total_timesteps",
#         type=int,
#         default=1_000_000,
#         help="Training timesteps for each algorithm."
#     )

#     parser.add_argument(
#         "--parallel",
#         action="store_true",
#         help="Run algorithms in parallel. If not set, runs sequentially."
#     )

#     parser.add_argument(
#         "--start_delay",
#         type=int,
#         default=20,
#         help="Delay in seconds between starting parallel jobs."
#     )

#     parser.add_argument(
#         "--config",
#         type=str,
#         default=None,
#         help="Optional config file path."
#     )

#     args = parser.parse_args()

#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     log_dir = Path("rl_training/pipeline_logs") / timestamp
#     log_dir.mkdir(parents=True, exist_ok=True)

#     jobs = []

#     for idx, algo in enumerate(args.algos):
#         instance = idx + 1
#         cmd = build_command(
#             algo=algo,
#             instance=instance,
#             total_timesteps=args.total_timesteps,
#             config=args.config,
#         )

#         log_path = log_dir / f"{algo}_instance_{instance}.log"

#         print("=" * 80)
#         print(f"Starting {algo.upper()} on instance {instance}")
#         print(f"Log file: {log_path}")
#         print("Command:")
#         print(" ".join(cmd))
#         print("=" * 80)

#         log_file = open(log_path, "w", encoding="utf-8")

#         if args.parallel:
#             process = subprocess.Popen(
#                 cmd,
#                 stdout=log_file,
#                 stderr=subprocess.STDOUT,
#                 cwd="/home/pid_rl",
#                 start_new_session=True,
# )
#             jobs.append((algo, instance, process, log_file))

#             time.sleep(args.start_delay)

#         else:
#             process = subprocess.Popen(
#                 cmd,
#                 stdout=log_file,
#                 stderr=subprocess.STDOUT,
#                 cwd="/home/pid_rl",
#             )
#             jobs.append((algo, instance, process, log_file))

#             return_code = process.wait()
#             log_file.close()

#             if return_code != 0:
#                 print(f"{algo.upper()} failed with return code {return_code}")
#                 return

#             print(f"{algo.upper()} completed successfully.")

#     if args.parallel:
#         print("\nAll jobs started. Waiting for completion...")

#         for algo, instance, process, log_file in jobs:
#             return_code = process.wait()
#             log_file.close()

#             if return_code == 0:
#                 print(f"{algo.upper()} on instance {instance} completed successfully.")
#             else:
#                 print(f"{algo.upper()} on instance {instance} failed with return code {return_code}.")


# if __name__ == "__main__":
#     main()