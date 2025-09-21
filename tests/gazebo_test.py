#!/usr/bin/env python3

import sys
sys.path.insert(0, "/home/pid_rl")

from rl_training.utils.utils import load_config
from rl_training.utils import GazeboInterface
import time

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

    config = load_config(config_path)
    gazebo = GazeboInterface(config['gazebo_config'], instance, True)
    gazebo.start_simulation()
    gazebo._wait_for_startup()
    gazebo.resume_simulation()
    time.sleep(5)
    print(gazebo.get_sim_time())

    gazebo.close()

if __name__ == "__main__":
    main() 