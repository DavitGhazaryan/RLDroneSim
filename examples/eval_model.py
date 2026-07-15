import sys
sys.path.insert(0, "/home/pid_rl")

from rl_training.environments import SimGymEnv
from rl_training.utils.utils import load_config, evaluate_agent

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor


def main():
    algo = "td3"   # baseline, td3, ddpg, ppo

    # Replace these with your actual "algo" run and checkpoint
    training_number = "20260708_040521" #"20260625_123112"
    checkpoint_step = "844000"#"799000"

    # Visual evaluation parameters
    gui = True
    speedup = 20
    n_eval = 1

    if algo != "baseline":
        run_dir = (
            f"/home/pid_rl/rl_training/runs/hover/"
            f"{algo}/{training_number}"
        )

        config_path = f"{run_dir}/cfg.yaml"
        model_zip = (
            f"{run_dir}/models/"
            f"{algo}_{checkpoint_step}_steps.zip"
        )
        vecnorm_path = (
            f"{run_dir}/models/"
            f"{algo}_vecnormalize_{checkpoint_step}_steps.pkl"
        )
    else:
        # config_path = "/home/pid_rl/rl_training/configs/default_config.yaml"
        run_dir = (
            f"/home/pid_rl/rl_training/runs/hover/"
            f"{algo}/{training_number}"
        )

        config_path = f"{run_dir}/cfg.yaml"
        model_zip = None
        vecnorm_path = None

    config = load_config(config_path)

    config["sitl_config"]["speedup"] = speedup
    config["gazebo_config"]["gui"] = gui

    # Create evaluation environment
    def make_env():
        env = SimGymEnv(
            config,
            eval_baseline=(algo == "baseline")
        )
        return Monitor(env)

    env = DummyVecEnv([make_env])

    # Load observation-normalization statistics
    if vecnorm_path:
        env = VecNormalize.load(vecnorm_path, env)
        env.training = False
        env.norm_reward = False

    # Load trained model
    if algo == "td3":
        from stable_baselines3 import TD3

        model = TD3.load(
            model_zip,
            env=env,
            device=config[f"{algo}_params"].get("device", "auto")
        )
        gamma = config[f"{algo}_params"]["gamma"]

    elif algo == "ddpg":
        from stable_baselines3 import DDPG

        model = DDPG.load(
            model_zip,
            env=env,
            device=config[f"{algo}_params"].get("device", "auto")
        )
        gamma = config[f"{algo}_params"]["gamma"]

    elif algo == "ppo":
        from stable_baselines3 import PPO

        model = PPO.load(
            model_zip,
            env=env,
            device=config[f"{algo}_params"].get("device", "cpu")
        )
        gamma = config[f"{algo}_params"]["gamma"]

    elif algo == "baseline":
        model = None
        gamma = 0.99

    else:
        raise ValueError(f"Unsupported algorithm: {algo}")

    header = f"Evaluating: {algo}"

    if model:
        header += (
            f" | run: {training_number}"
            f" | step: {checkpoint_step}"
            f" | episodes: {n_eval}"
        )

    print(header)

    evaluate_agent(
        model,
        env,
        n_eval,
        gamma=gamma,
        log_pid_gains=True,
        gain_keys=config["environment_config"]["action_gains"].split("+"),
        save_dir=f"{run_dir}/evaluation_{checkpoint_step}_steps",        
        verbose=True
    )

    env.close()


if __name__ == "__main__":
    main()