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
from mavsdk import System
from mavsdk import mission_raw

import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from rl_training.utils.utils import load_config

# adjust this path to point at your cloned ArduPilot dir
ARDUPILOT_DIR = "/home/pid_rl/ardupilot"
sys.path.append(str(os.path.dirname(__file__)))

from rl_training.environments.ardupilot_env import ArdupilotEnv


async def upload_mission(drone):
    # 1) ALWAYS start by wiping whatever is on the FCU
    await drone.mission_raw.clear_mission()

    items = []

    # 0 - TAKE-OFF (current **0**)
    items.append(mission_raw.MissionItem(
        0, 3, 22, 0, 1,          # seq, frame, cmd, current, autocontinue
        0, 0, 0, float('nan'),
        int(47.40271757e7), int(8.54285027e7), 30.0, 0))

    # 1 - WAYPOINT A
    items.append(mission_raw.MissionItem(
        1, 3, 16, 0, 1,
        0, 10, 0, float('nan'),
        int(47.40271757e7), int(8.54361892e7), 30.0, 0))

    # 2 - RTL
    items.append(mission_raw.MissionItem(
        2, 3, 20, 0, 1,
        0, 0, 0, 0,
        0, 0, 0, 0))

    print("Uploading mission…")
    await drone.mission_raw.upload_mission(items)
    print("Mission accepted ✅")

    # Arm, AUTO mode, start
    await drone.action.arm()
    await drone.mission_raw.start_mission()



async def mavsdk_task(env: ArdupilotEnv):
    # get or establish MAVSDK System
    drone = await env.sitl._get_mavsdk_connection()

    pid_params = await env.sitl.get_pid_params_async()
    print(f"Initial PID params: {pid_params}")

    # wait until armable (and has GPS fix)
    print("Waiting for vehicle to become armable...")
    async for health in drone.telemetry.health():
        if health.is_armable and health.is_global_position_ok:
            print("Vehicle is armable and has GPS fix!")
            break
        await asyncio.sleep(1)



    # await upload_mission(drone)

    # arm
    print("Arming...")
    await drone.action.arm()


    await asyncio.sleep(1)

    # takeoff to 5 m
    print("Taking off to 5 m...")
    await drone.action.takeoff()

    await asyncio.sleep(5.0)

    # await env.sitl.set_params_async()

    await asyncio.sleep(5.0)


    # land
    # print("Landing...")
    await drone.action.land()

    # hover for 5 s
    await asyncio.sleep(15.0)
    # await env.sitl.reset_async(keep_params=True)

    # await asyncio.sleep(15.0)

def main():

    parser = argparse.ArgumentParser(description='Test SITL workflow')
    parser.add_argument('--config', '-c', type=str, 
                       default='/home/pid_rl/rl_training/configs/default_config.yaml',
                       help='Path to configuration YAML file')
    args = parser.parse_args()

    print(f"📋 Loading configuration from: {args.config}")
    
    config = load_config(args.config)
    env = ArdupilotEnv(config)
    env.reset()



    try:
        asyncio.run(mavsdk_task(env))
    except Exception as e:
        if "KeyboardInterrupt" in str(e):
            print("🛑 Interrupted by user. Cleaning up...")
            raise e
        else:
            print(f"Error during MAVSDK operations: {e}")
    finally:
        env.close()
        print("✅ Integration test complete.")


if __name__ == "__main__":
    main()
