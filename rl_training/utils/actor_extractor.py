# on your training machine
from stable_baselines3 import DDPG
import torch, os, pickle

folder = "/home/pid_rl/rl_training/runs/ddpg/hover/20251005_040415"
zip_file = "models/ddpg_ardupilot_5500_steps.zip"
pt_name = "policy_actor_scripted.pt"


model = DDPG.load(f"{folder}/{zip_file}")
actor = model.actor.mu  # nn.Module producing deterministic actions

scripted = torch.jit.script(actor)
scripted.save(f"{folder}/{pt_name}")
