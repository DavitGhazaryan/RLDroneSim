from pymavlink import mavutil

m = mavutil.mavlink_connection("udp:127.0.0.1:5760")
while True:
    msg = m.recv_match(blocking=True)
    if not msg:
        continue
    if msg.get_type() == "COMMAND_LONG" and msg.command == mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL:
        print("SET_MESSAGE_INTERVAL:", msg)
    elif msg.get_type() in ["PARAM_REQUEST_LIST", "PARAM_REQUEST_READ"]:
        print("PARAM REQUEST:", msg)
