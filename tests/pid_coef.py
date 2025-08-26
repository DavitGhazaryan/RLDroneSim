#!/usr/bin/env python3
import os, time, math
from pymavlink import mavutil

os.environ.setdefault("MAVLINK_DIALECT", "ardupilotmega")

CONN_STR = "udp:127.0.0.1:14551"

# MAVLink message IDs
# PID_TUNING = 194
# NAV_CONTROLLER_OUTPUT = 62
# LOCAL_POSITION_NED = 32

# RATE_PID_HZ = 20
# RATE_DBG_HZ = 20

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
    # set_param_and_confirm(master, "GCS_PID_MASK", GCS_PID_MASK_VALUE)

    # Ask for streams
    # set_message_interval(master, PID_TUNING, RATE_PID_HZ)  # PID_TUNING (AccZ)
    # set_message_interval(master, NAV_CONTROLLER_OUTPUT, RATE_PID_HZ)  # alt_error

    print("\nStreaming altitude-related data… (Ctrl+C to stop)")
    # print(master.messages)
    while True:
        try:
            print(master.messages["NAV_CONTROLLER_OUTPUT"])
            print(master.messages["PID_TUNING[4]"])
            print(master.messages["DEBUG_VECT"])
        except:
            pass
        # accZ_err = None
    # vZ_err = None
    # alt_err = None
    # trials = 0
    # print("scsdc")
    # while (alt_err is None) or (vZ_err is None) or (accZ_err is None):
    #     trials += 1
    #     print("enter")
    #     try:
    #         vZ_err = master.messages["DEBUG_VECT"].z
    #         alt_err = master.messages["NAV_CONTROLLER_OUTPUT"].alt_error
    #         accZ_err = master.messages["PID_TUNING[4]"].desired - master.messages["PID_TUNING[4]"].achieved 

    #         print(vZ_err)
    #         print(alt_err)
    #         print(accZ_err)
    #     except:
    #         print("Exception")
    #         if trials > 3:
    #             break
    # print("received")
    # print(accZ_err)
    # print(vZ_err)
    # print(alt_err)
    
if __name__ == "__main__":
    main()
