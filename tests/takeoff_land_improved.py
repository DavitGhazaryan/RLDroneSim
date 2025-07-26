#!/usr/bin/env python3
import asyncio
from mavsdk import System
from mavsdk.offboard import PositionNedYaw

async def run():
    drone = System()

    # 1) Connect
    print("⏳ Connecting to drone…")
    await drone.connect(system_address="udp://0.0.0.0:14550")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("✅ Connected")
            break

    # 2) Raw GPS 3D fix
    # print("⏳ Waiting for raw GPS 3D fix…")
    # async for gps in drone.telemetry.gps_info():
    #     if gps.fix_type.name.endswith("3D") and gps.satellites_visible >= 6:
    #         print(f"✅ GPS 3D fix: {gps.fix_type.name}, sats={gps.satellites_visible}")
    #         break
    #     await asyncio.sleep(0.5)

    # 3) EKF position OK
    print("⏳ Waiting for EKF local & global position…")
    async for health in drone.telemetry.health():
        if health.is_local_position_ok and health.is_global_position_ok:
            print("✅ EKF position OK")
            break
        await asyncio.sleep(0.5)

    # 4) Armable
    print("⏳ Waiting until armable…")
    async for health in drone.telemetry.health():
        if health.is_armable:
            print("✅ Vehicle is armable")
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

    # 6) Offboard climb 10 m
    print("⚙️  Starting Offboard to climb 10 m…")
    # You must set a reasonable timeout / retry in real code
    try:
        # build a single NED setpoint: north=0, east=0, down=-10 (i.e. up 10m)
        await drone.offboard.start()
        await drone.offboard.set_position_ned(PositionNedYaw(0.0, 0.0, -10.0, 0.0))
        print("✅ Offboard command sent, climbing…")
        # let it climb & hover
        await asyncio.sleep(10)
    except Exception as e:
        print(f"⚠️  Offboard climb failed: {e}")
    finally:
        # always stop offboard before other commands
        try:
            await drone.offboard.stop()
        except:
            pass

    # 7) Land
    print("✈️  Landing…")
    try:
        await drone.action.land()
        print("✅ Land command sent")
    except Exception as e:
        print(f"❌ Land failed: {e}")

if __name__ == "__main__":
    asyncio.run(run())
