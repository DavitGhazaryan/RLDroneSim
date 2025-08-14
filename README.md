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

## Setup for Ubuntu 22.04

This project uses Docker for easy deployment and consistent environment setup. The configuration has been tested on Ubuntu 22.04 host machines with NVIDIA GPU support.

### Prerequisites

- Docker and Docker Compose installed
- NVIDIA GPU with compatible drivers (for GPU acceleration)
- Linux host machine (recommended)
- X11 server access for GUI applications

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone git@github.com:GorArzanyanAUA/pid_rl.git
   cd pid_rl
   ```

2. **Configure X11 access for GUI applications**
   ```bash
   xhost +local:docker
   ```

3. **Build the Docker container**
   ```bash
   DOCKER_BUILDKIT=1 docker-compose build
   ```

4. **Start the container in detached mode**
   ```bash
   docker-compose up -d
   ```

5. **Enter the container**
   ```bash
   docker exec -it pid_rl_container bash
   ```

## Setup for Windows with WSLG

This section describes how to set up the project on Windows using WSLG (Windows Subsystem for Linux GUI) for GUI applications.

### Prerequisites

- Windows 11 with WSL2 and WSLG enabled
- Docker Desktop for Windows installed
- WSL2 distribution (Ubuntu 22.04 recommended)
- WSLG support enabled

### Installation Steps

1. **Clone the repository inside your WSL environment**
   ```bash
   git clone git@github.com:GorArzanyanAUA/pid_rl.git
   cd pid_rl
   ```

2. **Build the Docker container using the WSLG-specific compose file**
   ```bash
   DOCKER_BUILDKIT=1 docker compose -f docker-compose-wslg.yml build
   ```

3. **Start the container in detached mode**
   ```bash
   docker compose -f docker-compose-wslg.yml up -d
   ```

4. **Enter the container**
   ```bash
   docker exec -it pid_rl_container bash
   ```

### Verification Tests

Once inside the container, run these tests to verify the setup:

1. **Test GPU access (if GPU is available)**
   ```bash
   nvidia-smi
   ```

2. **Test Gazebo GUI**
   ```bash
   gz sim simple_world.sdf
   ```

3. **Test complete SITL workflow**
   ```bash
   python3 tests/test_full_sitl_workflow.py
   ```

### Troubleshooting

- **Different host OS**: The Dockerfile should work on any machine, but you may need to modify `docker-compose.yml` for non-Linux systems
- **GPU compatibility**: If you encounter CUDA version issues, update the CUDA version in the Dockerfile to match your GPU
- ****: If you encounter CUDA version issues, update the CUDA version in the Dockerfile to match your GPU
b07cc04124 Copter: PR feedback