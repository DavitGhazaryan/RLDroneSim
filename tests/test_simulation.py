#!/usr/bin/env python3
"""
SITL + Gazebo Integration Test

Starts both ArduPilot SITL and Gazebo using their respective interfaces,
waits for them to initialize, and then idles until Ctrl+C. Cleans up on exit.
"""

import sys
import os
import time
import logging
import argparse
import yaml
from pathlib import Path

# ensure project root on PYTHONPATH
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from rl_training.utils.ardupilot_sitl import ArduPilotSITL
from rl_training.utils.gazebo_interface import GazeboInterface

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("IntegrationTest")

def load_config(config_path):
    """Load configuration from YAML file."""
    if not config_path or not Path(config_path).exists():
        print(f"⚠️  Config file not found: {config_path}")
        print("Using default configuration...")
        return get_default_configs()
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        ardupilot_config = config.get('ardupilot_config', {})
        gazebo_config = config.get('gazebo_config', {})
        return ardupilot_config, gazebo_config
    except Exception as e:
        print(f"❌ Error loading config file: {e}")
        print("Using default configuration...")
        return get_default_configs()

def get_default_configs():
    """Return default configurations for both ArduPilot and Gazebo."""
    ardupilot_config = {
        'ardupilot_path': '/home/student/Dev/pid_rl/ardupilot',
        'vehicle': 'ArduCopter',
        'frame': 'gazebo-iris',
        'model': 'JSON',
        'timeout': 60.0,
        'min_startup_delay': 5.0,
    }
    
    gazebo_config = {
        'sdf_file': '/home/student/Dev/pid_rl/ardupilot_gazebo/worlds/simple_world.sdf',
        'gui': 'DISPLAY' in os.environ,
        'verbose': False,
        'timeout': 30.0,
    }
    
    return ardupilot_config, gazebo_config

def main():
    parser = argparse.ArgumentParser(description='Test SITL + Gazebo integration')
    parser.add_argument('--config', '-c', type=str, 
                       default='rl_training/configs/default_config.yaml',
                       help='Path to configuration YAML file')
    args = parser.parse_args()

    print(f"📋 Loading configuration from: {args.config}")
    ardupilot_config, gazebo_config = load_config(args.config)
    
    # locate resources
    ardupilot_dir = Path(ardupilot_config['ardupilot_path'])
    sdf_file = gazebo_config['sdf_file']

    print(f"🚁 ArduPilot path: {ardupilot_dir}")
    print(f"🌎 SDF file: {sdf_file}")
    print(f"🖥️  GUI enabled: {gazebo_config.get('gui', False)}")

    # validate paths
    if not ardupilot_dir.exists():
        logger.error(f"ArduPilot directory not found: {ardupilot_dir}")
        return
    if not Path(sdf_file).exists():
        logger.error(f"SDF world not found: {sdf_file}")
        return

    sitl = ArduPilotSITL(ardupilot_config)
    gazebo = GazeboInterface(gazebo_config)

    try:

        # start Gazebo
        print("🌎 Launching Gazebo simulation...")
        gazebo.start_simulation()
        gazebo._wait_for_startup()
        gazebo.resume_simulation()
        print("✅ Gazebo initialized")

        # start SITL
        print("🚁 Starting ArduPilot SITL...")
        sitl.start_sitl()
        info = sitl.get_process_info()
        print(f"✅ SITL running (PID {info['pid']})")

        print("\n▶️  Both SITL and Gazebo are up. Press Ctrl+C to terminate.")
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user. Cleaning up...")

    except Exception as e:
        logger.error(f"Error during integration test: {e}")

    finally:
        # teardown
        print("Stopping Gazebo...")
        try:
            gazebo.close()
            print("Gazebo stopped.")
        except Exception:
            pass

        print("Stopping SITL...")
        try:
            sitl.close()
            print("SITL stopped.")
        except Exception:
            pass

        print("✅ Integration test complete.")

if __name__ == "__main__":
    main()
