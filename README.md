# Ardupilot RL Training System

A comprehensive framework for training reinforcement learning agents on Ardupilot SITL (Software In The Loop) with Gazebo simulation.

## Project Structure

```
pid_rl/
├── ardupilot/                # Ardupilot source code  (No changes)
│   ├── ArduCopter/           # Copter firmware
│   └── Tools/                # Ardupilot tools including sim_vehicle.py for SITL
│
├── ardupilot_gazebo/          # Gazebo plugins for Ardupilot
│   ├── worlds/               # Included simple_world.sdf
│   └── models/               
│
│
├── rl_training/               # in development
│   ├── agents/                
│   ├── environments/          # Custom gym-like environments
│   ├── utils/                 # Utility wrappers and functions
│   ├── configs/               
│   └── training/              # Training and evaluation scripts
│
├── docs/                      
└── tests/                     # Interface tests and simple scripts
```

## Setup

Dependency resolution is not yet finalized.

So far the system was tested on Linux. The setup should be automated.

1. Clone the repository.
2. Setup Ardupilot https://ardupilot.org/dev/docs/building-setup-linux.html#building-setup-linux
     it should be next to rl_training folder.
3. Make sure that SITL works https://ardupilot.org/dev/docs/setting-up-sitl-on-linux.html
4. Setup ardupilot-gazebo plugin https://ardupilot.org/dev/docs/sitl-with-gazebo.html
    Make sure the paths are written correctly.   It should be next to rl_training folder.
5. Copy the simple_world.sdf to ardupilot_gazebo/worlds/simple_world.sdf 

The environment should be ready. For now, troubleshoot the requirements manually.


## Quick Start: Simple Drone Test

This guide will help you run a basic drone simulation using Ardupilot SITL and Gazebo.

In a terminal, run:

```bash
# Run SITL with Gazebo integration
python3 pid_rl/tests/test_full_sitl_workflow.py

```
