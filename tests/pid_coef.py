#!/usr/bin/env python3
import os, time
from pymavlink import mavutil

# Decode ArduPilot messages like PID_TUNING
os.environ.setdefault("MAVLINK_DIALECT", "ardupilotmega")

CONN_STR = "udp:127.0.0.1:14550"   # adjust if needed
PID_TUNING_MSG_ID = 194
INTERVAL_US = 20000                # 50 Hz
GCS_PID_MASK_VALUE = 0xFFFF        # enable all PID_TUNING groups
PARAM_NAME = "GCS_PID_MASK"        # compare as str, convert to bytes on send

AXIS_NAME = {
    1: "ROLL", 2: "PITCH", 3: "YAW",
    4: "VEL_X", 5: "VEL_Y", 6: "VEL_Z",
    7: "POS_X", 8: "POS_Y", 9: "POS_Z",
}

def _strip_id(x):
    """Return param_id as clean str, regardless of bytes/str."""
    if isinstance(x, (bytes, bytearray)):
        return x.decode("ascii", "ignore").rstrip("\x00")
    return str(x).rstrip("\x00")

def wait_heartbeat(m):
    print("Waiting for heartbeat...")
    hb = m.wait_heartbeat()
    # Force compid to autopilot if not set
    m.target_system = hb.get_srcSystem()
    m.target_component = hb.get_srcComponent() or mavutil.mavlink.MAV_COMP_ID_AUTOPILOT1
    print(f"HB from sys:{m.target_system} comp:{m.target_component}")

def set_param_and_confirm(m, name_str, value, timeout=3.0):
    # MAVLink param name payload must be <=16 bytes, null-padded by pymavlink
    name_bytes = name_str.encode("ascii", "ignore")
    m.mav.param_set_send(
        m.target_system, m.target_component,
        name_bytes, float(value), mavutil.mavlink.MAV_PARAM_TYPE_INT32
    )
    t0 = time.time()
    while time.time() - t0 < timeout:
        msg = m.recv_match(type="PARAM_VALUE", blocking=True, timeout=timeout)
        if not msg:
            break
        if _strip_id(msg.param_id) == name_str:
            print(f"PARAM {name_str} => {int(msg.param_value)} (ack)")
            return True
    print(f"WARNING: no PARAM_VALUE ack for {name_str}; continuing.")
    return False

def set_message_interval(m, msg_id, interval_us):
    m.mav.command_long_send(
        m.target_system, m.target_component,
        mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
        float(msg_id), float(interval_us), 0, 0, 0, 0, 0
    )
    ack = m.recv_match(type="COMMAND_ACK", blocking=True, timeout=2.0)
    if ack and ack.command == mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL:
        print(f"SET_MESSAGE_INTERVAL result: {ack.result}")
    else:
        print("No COMMAND_ACK for SET_MESSAGE_INTERVAL (may still take effect).")


def main():
    master = mavutil.mavlink_connection(CONN_STR)
    wait_heartbeat(master)

    set_param_and_confirm(master, PARAM_NAME, GCS_PID_MASK_VALUE)
    set_message_interval(master, PID_TUNING_MSG_ID, INTERVAL_US)

    set_message_interval(master, 62, 100000)   # NAV_CONTROLLER_OUTPUT @10Hz
    # or: set_message_interval(master, 32, 100000) # LOCAL_POSITION_NED

    while True:
        # existing PID_TUNING read...
        msg = master.recv_match(blocking=True, timeout=1.0)
        if not msg:
            continue

        if msg.get_type() == "NAV_CONTROLLER_OUTPUT":
            alt_err = float(getattr(msg, "alt_error", float("nan")))
            xtrk   = float(getattr(msg, "xtrack_error", float("nan")))
            nroll  = float(getattr(msg, "nav_roll", float("nan")))
            npitch = float(getattr(msg, "nav_pitch", float("nan")))
            print(f"[POS ] alt_err={alt_err:7.3f}m  xtrack_err={xtrk:7.3f}m  "
                f"nav_roll={nroll:6.2f}  nav_pitch={npitch:6.2f}")


        # (optional) elif msg.get_type() == "LOCAL_POSITION_NED": use msg.z (down, meters)

if __name__ == "__main__":
    main()
