#!/usr/bin/env python3
import os, time, math
from pymavlink import mavutil

os.environ.setdefault("MAVLINK_DIALECT", "ardupilotmega")

CONN_STR = "udp:127.0.0.1:14551"

# MAVLink message IDs
PID_TUNING = 194
NAV_CONTROLLER_OUTPUT = 62
LOCAL_POSITION_NED = 32

RATE_PID_HZ = 20
RATE_DBG_HZ = 20

# Enable AccZ (bit 5) plus others if you want
GCS_PID_MASK_VALUE = 0xFFFF  # 32=AccZ only; keep 0xFFFF if you really want all

def wait_heartbeat(m):
    print("Waiting for heartbeat...")
    hb = m.wait_heartbeat()
    m.target_system = hb.get_srcSystem()
    m.target_component = hb.get_srcComponent() or mavutil.mavlink.MAV_COMP_ID_AUTOPILOT1
    print(f"HB from sys:{m.target_system} comp:{m.target_component}")

def set_param_and_confirm(m, name_str, value, timeout=3.0):
    name_bytes = name_str.encode("ascii", "ignore")
    m.mav.param_set_send(m.target_system, m.target_component,
                         name_bytes, float(value),
                         mavutil.mavlink.MAV_PARAM_TYPE_INT32)
    t0 = time.time()
    while time.time() - t0 < timeout:
        msg = m.recv_match(type="PARAM_VALUE", blocking=True, timeout=timeout)
        if not msg:
            break
        pid = msg.param_id.decode("ascii","ignore").rstrip("\x00") if isinstance(msg.param_id,(bytes,bytearray)) else str(msg.param_id).rstrip("\x00")
        if pid == name_str:
            print(f"PARAM {name_str} => {int(msg.param_value)} (ack)")
            return True
    print(f"WARN: no PARAM_VALUE ack for {name_str}")
    return False

def set_message_interval(m, msg_id, hz):
    interval_us = int(1e6 / hz)
    m.mav.command_long_send(
        m.target_system, m.target_component,
        mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
        float(msg_id), float(interval_us), 0, 0, 0, 0, 0)
    m.recv_match(type="COMMAND_ACK", blocking=False, timeout=0.5)

def main():
    master = mavutil.mavlink_connection(CONN_STR)
    wait_heartbeat(master)

    # Ensure AccZ PID_TUNING is enabled
    set_param_and_confirm(master, "GCS_PID_MASK", GCS_PID_MASK_VALUE)

    # Ask for streams
    set_message_interval(master, PID_TUNING, RATE_PID_HZ)  # PID_TUNING (AccZ)

    print("\nStreaming altitude-related data… (Ctrl+C to stop)")
    while True:
        msg = master.recv_match(blocking=True, timeout=1.0)
        if not msg:
            continue

        t = msg.get_type()

        # --- Inner accel-Z loop from PID_TUNING ---
        if t == "PID_TUNING":
            axis = getattr(msg, "axis", None)
            # Keep only AccZ (exclude roll/pitch/yaw axes = 1/2/3)
            if axis in (1, 2, 3):
                continue
            desired = float(getattr(msg, "desired", float("nan")))   # m/s^2 (+down)
            achieved = float(getattr(msg, "achieved", float("nan"))) # m/s^2 (+down)
            err = desired - achieved
            print(f"[ACCZ] des_acc={desired:8.3f}  ach_acc={achieved:8.3f}  err={err:8.3f}")

        # --- Your custom velocity-Z debug (DEBUG_VECT "VELZPID") ---
        elif t == "DEBUG_VECT":
            # name is bytes in pymavlink
            name = msg.name.decode('ascii', 'ignore') if isinstance(msg.name, (bytes, bytearray)) else str(msg.name)
            if name == "VELZPID":
                vz_sp_up   = float(msg.x)  # you send target in x
                vz_meas_up = float(msg.y)  # you send actual in y
                vz_error = float(msg.z)  # you send error


                # Try to infer what 'third' is: if close to (sp - meas), call it err; else print both.
                print(f"[VELZ] vz_sp={vz_sp_up:7.3f}  vz={vz_meas_up:7.3f}  err={vz_error:7.3f} ")

if __name__ == "__main__":
    main()
