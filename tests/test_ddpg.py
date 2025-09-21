# pip install stable-baselines3[extra] gymnasium[classic-control] tensorboard

import gymnasium as gym
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
import numpy as np

def make_env(mode=None):
    if mode:
        env = gym.make("Pendulum-v1", render_mode=mode)  # continuous Box action space
    else:
        env = gym.make("Pendulum-v1")
    return env

env = make_env()

# OU noise is standard for DDPG on continuous control
n_actions = env.action_space.shape[-1]
ou_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=0.1*np.ones(n_actions))

model = DDPG(
    "MlpPolicy",
    env,
    learning_rate=1e-3,
    buffer_size=100_000,
    learning_starts=1_000,
    batch_size=256,
    tau=0.005,
    gamma=0.98,
    train_freq=(1, "episode"),   # simple & stable for Pendulum
    gradient_steps=50,
    action_noise=ou_noise,
    tensorboard_log="./tb",       # <-- enable TensorBoard logs
    verbose=1,
    seed=0,
)

model.learn(total_timesteps=50_000, log_interval=10)

# quick evaluation
eval_env = make_env(mode="human")
obs, info = eval_env.reset(seed=1)
ep_returns = []
ep_ret = 0.0
for _ in range(5_000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = eval_env.step(action)
    ep_ret += reward
    if terminated or truncated:
        ep_returns.append(ep_ret)
        ep_ret = 0.0
        obs, info = eval_env.reset()

print(f"Mean return over {len(ep_returns)} episodes: {np.mean(ep_returns):.1f}")

eval_env.close()
env.close()
