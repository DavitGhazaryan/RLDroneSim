#!/usr/bin/env python3
"""
Smoke tests for real ArduPilot SITL launch and basic operations without test frameworks.
Requires: ArduPilot repository cloned and environment set up.
Set ARDUPILOT_DIR env var to your ArduPilot path, or edit the hardcoded path below.
"""
import argparse
import os
import sys
import time
import asyncio
from pathlib import Path
import yaml

import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# adjust this path to point at your cloned ArduPilot dir
ARDUPILOT_DIR = "/home/student/Dev/pid_rl/ardupilot"
sys.path.append(str(os.path.dirname(__file__)))

from rl_training.utils.ardupilot_sitl import ArduPilotSITL
from rl_training.utils.gazebo_interface import GazeboInterface


async def mavsdk_task(sitl: ArduPilotSITL):
    # get or establish MAVSDK System
    drone = await sitl._get_mavsdk_connection()

    # wait until armable (and has GPS fix)
    print("Waiting for vehicle to become armable...")
    async for health in drone.telemetry.health():
        if health.is_armable and health.is_global_position_ok:
            print("Vehicle is armable and has GPS fix!")
            break
        await asyncio.sleep(1)

    # arm
    print("Arming...")
    await drone.action.arm()


    await asyncio.sleep(1)

    # takeoff to 5 m
    print("Taking off to 5 m...")
    await drone.action.takeoff()

    await asyncio.sleep(15.0)

    # land
    # print("Landing...")
    await drone.action.land()

    # hover for 5 s
    await asyncio.sleep(15.0)
    await sitl.reset_async(keep_params=True)

    await asyncio.sleep(15.0)
    # await asyncio.sleep(15.0)


def load_config(config_path):
    """Load configuration from YAML file."""
    if not config_path or not Path(config_path).exists():
        print(f"⚠️  Config file not found: {config_path}")
        print("Using default configuration...")
        return get_default_config()
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"❌ Error loading config file: {e}")
        print("Using default configuration...")
        return get_default_config()


def get_default_config():
    """Return default Gazebo configuration."""
    return {
        'ardupilot_config': {
            'ardupilot_path': ARDUPILOT_DIR,
            'vehicle': 'ArduCopter',
            'frame': 'gazebo-iris',
            'model': 'JSON',
            'timeout': 60.0
        },
        'gazebo_config': {
            'sdf_file': '/home/student/Dev/pid_rl/ardupilot_gazebo/worlds/simple_world.sdf',
            'gui': 'DISPLAY' in os.environ,
            'verbose': True,
            'timeout': 15.0
        }
    }


def main():

    parser = argparse.ArgumentParser(description='Test SITL workflow')
    parser.add_argument('--config', '-c', type=str, 
                       default='/home/student/Dev/pid_rl/rl_training/configs/default_config.yaml',
                       help='Path to configuration YAML file')
    args = parser.parse_args()

    print(f"📋 Loading configuration from: {args.config}")
    
    config = load_config(args.config)

    gazebo = GazeboInterface(config['gazebo_config'])
    sitl = ArduPilotSITL(config['ardupilot_config'])
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

        try:
            asyncio.run(mavsdk_task(sitl))
        except Exception as e:
            if "KeyboardInterrupt" in str(e):
                print("🛑 Interrupted by user. Cleaning up...")
                raise e
            else:
                print(f"Error during MAVSDK operations: {e}")
        finally:
            print("Stopping SITL...")
            sitl.stop_sitl()
            print("Stopping Gazebo...")
            gazebo.stop_simulation()
            print("✅ Gazebo stopped.")
            print("✅ SITL stopped.")

    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user. Cleaning up...")

    finally:
        # teardown
        print("Stopping Gazebo...")
        try:
            gazebo.close()
            print("Gazebo stopped.")
        except Exception:
            print("Failed to stop Gazebo.")
            pass

        print("Stopping SITL...")
        try:
            sitl.close()
            print("SITL stopped.")
        except Exception:
            print("Failed to stop SITL.")
            pass

        print("✅ Integration test complete.")


if __name__ == "__main__":
    main()
