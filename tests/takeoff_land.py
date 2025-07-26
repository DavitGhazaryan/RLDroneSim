#!/usr/bin/env python3
import asyncio
from mavsdk import System
from pymavlink import mavutil

async def run():
    drone = System()

    # 1) Connect to SITL
    print("⏳ Connecting to drone…")
    await drone.connect(system_address="udp://0.0.0.0:14550")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("✅ Connected")
            break

    # 2) Switch into GUIDED via pymavlink on mirror port
    print("⚙️  Switching to GUIDED via pymavlink…")
    master = mavutil.mavlink_connection("udpin:0.0.0.0:14560")
    master.wait_heartbeat()  # wait for SITL heartbeat
    mapping = master.mode_mapping()
    guided_mode = mapping.get("GUIDED")
    if guided_mode is None:
        print("❌ GUIDED mode not supported")
    else:
        master.mav.set_mode_send(
            master.target_system,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            guided_mode
        )
        # wait for ACK
        ack = None
        for _ in range(10):
            msg = master.recv_match(type="COMMAND_ACK", blocking=True, timeout=1)
            if msg and msg.command == mavutil.mavlink.MAV_CMD_DO_SET_MODE:
                ack = msg
                break
        if ack and ack.result == mavutil.mavlink.MAV_RESULT_ACCEPTED:
            print("✅ Mode switched to GUIDED")
        else:
            print("❌ Mode switch failed or not ACKed")
    master.close()

    # 3) Wait for overall armable health
    print("⏳ Waiting for pre‐arm checks (GPS, sensors, battery)…")
    async for health in drone.telemetry.health():
        if health.is_armable:
            print("✅ Health reports armable")
            break
        await asyncio.sleep(0.5)

    # 4) Wait for global position & home position OK
    print("⏳ Waiting for global position & home OK…")
    async for health in drone.telemetry.health():
        if health.is_global_position_ok and health.is_home_position_ok:
            print("✅ Global position estimate OK")
            break
        await asyncio.sleep(0.5)

    # 5) Arm
    print("🔐 Arming…")
    try:
        await drone.action.arm()
        print("✅ Armed")
    except Exception as e:
        print(f"❌ Arm failed: {e}")
        return

    # 6) Takeoff
    print("🚀 Sending takeoff…")
    try:
        await drone.action.takeoff()
        print("✅ Takeoff command sent")
    except Exception as e:
        print(f"❌ Takeoff failed: {e}")
        return

    # hover for a bit
    await asyncio.sleep(10)

    # 7) Land
    print("✈️  Sending land…")
    try:
        await drone.action.land()
        print("✅ Land command sent")
    except Exception as e:
        print(f"❌ Land failed: {e}")

if __name__ == "__main__":
    asyncio.run(run())
