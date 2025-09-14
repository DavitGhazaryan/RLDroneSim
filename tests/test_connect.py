from pymavlink import mavutil

# connect to SITL started with --udp (server at 127.0.0.1:5760)
m = mavutil.mavlink_connection("udp:127.0.0.1:5760", dialect="ardupilotmega")

m.wait_heartbeat(timeout=5)   # now m.target_system/component are valid

def set_rate(msgid, hz):
    interval_us = 0 if hz<=0 else int(1e6/hz)
    m.mav.command_long_send(
        m.target_system, m.target_component,
        mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
        msgid, interval_us, 0,0,0,0,0
    )
    # optional: read ack
    ack = m.recv_match(type="COMMAND_ACK", blocking=True, timeout=0.5)
    # print("ACK:", ack)

# start with always-supported msgs
# set_rate(30, 1)    # ATTITUDE
# set_rate(105, 50)   # HIGHRES_IMU
# set_rate(33, 10)    # GLOBAL_POSITION_INT
# set_rate(74, 10)    # VFR_HUD
# then the ones you want
set_rate(62, 20)    # NAV_CONTROLLER_OUTPUT
set_rate(194, 20)   # PID_TUNING (works on recent ArduPilot; else ignore)

# fallback (bundled streams) if needed:
m.mav.request_data_stream_send(m.target_system, m.target_component,
    mavutil.mavlink.MAV_DATA_STREAM_ALL, 20, 1)

while True:
    msg = m.recv_match(blocking=True, timeout=1.0)
    if msg:
        print(msg.get_type(), msg)
