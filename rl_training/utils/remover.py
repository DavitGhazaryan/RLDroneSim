import os
import re

# Path to your checkpoints folder
folder = "/home/pid_rl/rl_training/runs/ddpg/hover/20251013_193506/models"

# Regex patterns for TD3 model and replay buffer files
model_pattern = re.compile(r"td3_ardupilot_(\d+)_steps\.zip$")
buffer_pattern = re.compile(r"td3_ardupilot_replay_buffer_(\d+)_steps\.pkl$")
vecnorm_pattern = re.compile(r"td3_ardupilot_vecnormalize_(\d+)_steps\.pkl$")

for filename in os.listdir(folder):
    match = model_pattern.match(filename) or buffer_pattern.match(filename) or vecnorm_pattern.match(filename)
    if match:
        step = int(match.group(1))
        # Keep only multiples of 5000
        if step % 5000 != 0:
            filepath = os.path.join(folder, filename)
            print(f"Deleting: {filepath}")
            os.remove(filepath)

print("Cleanup done.")

