import sys
sys.path.insert(0, "/home/pid_rl")

from rl_training.environments.base_env import BaseEnv
from rl_training.utils.utils import load_config

def main():
    import argparse
    import sys
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

    config_path = '/home/pid_rl/rl_training/configs/default_config.yaml'

    try:
        config = load_config(config_path)
        env = BaseEnv(config, deploy=False, instance=instance)
        env.reset(12)
        while True:
            pass



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