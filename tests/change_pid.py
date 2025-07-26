#!/usr/bin/env python3

"""
Step 1: Start Gazebo Simulation
First, start the Gazebo simulation with our minimal world:
gz sim -v4 -r simple_world.sdf    

Step 2: Start Ardupilot SITL
sim_vehicle.py -v ArduCopter -f gazebo-iris --model JSON --map --console     # optionally --udp, --no-mavproxy

Step 3: 
python3 change_pid.py
"""


import asyncio
from mavsdk import System

async def run():
    drone = System()
    await drone.connect(system_address="udp://0.0.0.0:14550")

    print("Waiting for drone to connect...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print(f"-- Connected to drone!")
            break

    print("Waiting for drone to have a global position estimate...")
    async for health in drone.telemetry.health():
        if health.is_global_position_ok and health.is_home_position_ok:
            print("-- Global position estimate OK")
            break

    print("-- Arming")
    await drone.action.arm()

    print("-- Taking off")
    await drone.action.takeoff()

    # Wait a little for the drone to stabilize after takeoff
    await asyncio.sleep(25)

    # Modify PID gains to destabilize the drone (e.g., high Kp, Ki, Kd for Roll and Pitch)
    print("-- Modifying PID gains to destabilize the drone")

    # Set large values for the PID parameters to destabilize the drone
    await set_pid_param(drone, "ATC_ANG_PIT_P", 10.0)  # Large value for Pitch Proportional
    await set_pid_param(drone, "ATC_ANG_RLL_P", 10.0)  # Large value for Roll Proportional
    await set_pid_param(drone, "ATC_ANG_YAW_P", 10.0)  # Large value for Yaw Proportional

    await set_pid_param(drone, "ATC_RAT_PIT_P", 10.0)  # Large value for Rate Pitch Proportional
    await set_pid_param(drone, "ATC_RAT_RLL_P", 10.0)  # Large value for Rate Roll Proportional
    await set_pid_param(drone, "ATC_RAT_YAW_P", 10.0)  # Large value for Rate Yaw Proportional

    await asyncio.sleep(15)

    # Monitor the drone status after the instability is introduced
    print("-- Monitoring the unstable drone")

    # Optional: Monitor status and check if the drone is still stable or performing unwanted movements.
    status_text_task = asyncio.ensure_future(print_status_text(drone))

    # Let the drone fly with instability for a while
    await asyncio.sleep(10)

    print("-- Landing")
    await drone.action.land()

    status_text_task.cancel()


async def set_pid_param(drone, param_name, value):
    print(f"Setting {param_name} to {value}")
    # Setting the PID parameters. These values can destabilize the drone.
    await drone.param.set_param_float(param_name, value)


async def print_status_text(drone):
    try:
        async for status_text in drone.telemetry.status_text():
            print(f"Status: {status_text.type}: {status_text.text}")
    except asyncio.CancelledError:
        return


if __name__ == "__main__":
    asyncio.run(run())
