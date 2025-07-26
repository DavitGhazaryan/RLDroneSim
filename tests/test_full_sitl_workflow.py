#!/usr/bin/env python3
"""
Smoke tests for real ArduPilot SITL launch and basic operations without test frameworks.
Requires: ArduPilot repository cloned and environment set up.
Set ARDUPILOT_DIR env var to your ArduPilot path, or edit the hardcoded path below.
"""
import os
import sys
import time
import asyncio

import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# adjust this path to point at your cloned ArduPilot dir
ARDUPILOT_DIR = "/home/student/Dev/pid_rl/ardupilot"

# add your package root so we can import ArduPilotSITL
sys.path.append(str(os.path.dirname(__file__)))

from rl_training.utils.ardupilot_sitl import ArduPilotSITL


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


def main():
    # config = {
    #     "ardupilot_path": ARDUPILOT_DIR,
    #     "vehicle":        "ArduCopter",
    #     "frame":          "quad",
    #     "timeout":        60.0,
    #     "min_startup_delay": 5.0,
    #     # optional: override ports if you like
    #     # "master_port": 14550,
    #     # "mavsdk_port": 14551,
    # }

    config = {
        "ardupilot_path": ARDUPILOT_DIR,
        "vehicle":        "ArduCopter",
        "frame":          "gazebo-iris",
        "model":          "JSON",
        "timeout":        60.0,
        "min_startup_delay": 5.0,
        # optional: override ports if you like
        # "master_port": 14550,
        # "mavsdk_port": 14551,
    }

    sitl = ArduPilotSITL(config)
    print("Starting SITL...")
    sitl.start_sitl()

    try:
        # run all MAVSDK ops in asyncio
        asyncio.run(mavsdk_task(sitl))
    except Exception as e:
        print(f"Error during MAVSDK operations: {e}")
    finally:
        print("Stopping SITL...")
        sitl.stop_sitl()


if __name__ == "__main__":
    main()
