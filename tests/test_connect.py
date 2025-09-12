# from pymavlink import mavutil

# master = mavutil.mavlink_connection("/dev/ttyUSB0", baud=57600)

# master.wait_heartbeat()
# print(" Heartbeat Received")

# # master.mav.command_long_send(
# #             master.target_system, master.target_component,
# #             mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
# #             float(msg_id), float(interval_us), 0, 0, 0, 0, 0)

# # master.recv_match(type="COMMAND_ACK", blocking=False, timeout=0.5)

# while True:
#     msg = master.recv_match(blocking=True)
#     if msg:
#         print(msg)