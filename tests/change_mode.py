#!/usr/bin/env python3
from pymavlink import mavutil
import time

# 1) Open a connection to SITL (replace with your address/port)
master = mavutil.mavlink_connection('udp:0.0.0.0:14550')

# 2) Wait for the first heartbeat 
#    (so we know the system ID and component ID of the autopilot)
master.wait_heartbeat()
print(f"Heartbeat from system {master.target_system}, component {master.target_component}")

# 3) Helper to request a mode change
def set_mode(mode_str):
    """
    mode_str: one of 'STABILIZE', 'GUIDED', 'AUTO', etc.
    """
    # look up the mode number in the MAV_MODE enum
    mode_id = master.mode_mapping()[mode_str]
    # send the SET_MODE message
    master.mav.set_mode_send(
        master.target_system,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        mode_id
    )
    # wait for ack
    ack = False
    while not ack:
        msg = master.recv_match(type='COMMAND_ACK', blocking=True, timeout=5)
        if msg and msg.command == mavutil.mavlink.MAV_CMD_DO_SET_MODE:
            if msg.result == mavutil.mavlink.MAV_RESULT_ACCEPTED:
                print(f"Mode change to {mode_str} accepted")
            else:
                print(f"Mode change to {mode_str} failed: {msg.result}")
            ack = True

# 4) Switch into GUIDED
set_mode('GUIDED')

# 5) Now you can arm / takeoff as normal
master.mav.command_long_send(
    master.target_system,
    master.target_component,
    mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
    0,
    1, 0, 0, 0, 0, 0, 0
)
print("Arming…")
time.sleep(2)

# 6) Takeoff (in GUIDED mode) to 10 m
master.mav.command_long_send(
    master.target_system,
    master.target_component,
    mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
    0,
    0, 0, 0, 0,
    0, 0, 10  # altitude=10 m
)
print("Takeoff command sent")
