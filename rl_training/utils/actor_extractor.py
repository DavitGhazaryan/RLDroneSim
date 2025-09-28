# on your training machine
from stable_baselines3 import DDPG
import torch, os, pickle

model = DDPG.load("/home/pid_rl/rl_training/runs/ddpg/hover/20250923_194305/models/ddpg_ardupilot_1000000_steps.zip")
actor = model.actor.mu  # nn.Module producing deterministic actions

# (A) TorchScript (simple + fast on Pi CPU)
scripted = torch.jit.script(actor)
scripted.save("policy_actor_scripted.pt")
