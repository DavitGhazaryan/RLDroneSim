# simple_mission.py
from pymavlink import mavutil
import math
import time

UDP = "udp:127.0.0.1:5760"  # adjust if needed

def wait_heartbeat(m):
    print("Waiting for heartbeat...")
    m.wait_heartbeat()
    print(f"HB from system {m.target_system} component {m.target_component}")

def mode_id(m, name):
    mapping = m.mode_mapping()
    if name not in mapping:
        raise RuntimeError(f"Mode {name} not in mapping: {list(mapping)}")
    return mapping[name]

def set_mode(m, name):
    m.set_mode(mode_id(m, name))
    print("Mode set to")

def arm(m, arm_it=True):
    m.mav.command_long_send(
        m.target_system, m.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0, 1 if arm_it else 0, 0, 0, 0, 0, 0, 0)

def meters_to_deg(dn, de, lat_deg):
    dlat = dn / 111319.49
    dlon = de / (111319.49 * math.cos(math.radians(lat_deg)))
    return dlat, dlon

def get_position(m):
    # Try GLOBAL_POSITION_INT, fall back to HOME_POSITION
    msg = m.recv_match(type='GLOBAL_POSITION_INT', blocking=True, timeout=5)
    if msg:
        return msg.lat / 1e7, msg.lon / 1e7, msg.alt / 1000.0
    hp = m.recv_match(type='HOME_POSITION', blocking=True, timeout=5)
    if not hp:
        raise RuntimeError("No position available")
    return hp.latitude / 1e7, hp.longitude / 1e7, hp.altitude / 1000.0

def mission_upload(m, items):
    # Clear any old mission
    m.mav.mission_clear_all_send(m.target_system, m.target_component)
    # Send count
    m.mav.mission_count_send(m.target_system, m.target_component, len(items), mavutil.mavlink.MAV_MISSION_TYPE_MISSION)

    pending = {i['seq']: i for i in items}
    while pending:
        req = m.recv_match(type=['MISSION_REQUEST_INT','MISSION_REQUEST'], blocking=True, timeout=10)
        if not req:
            raise RuntimeError("Timed out waiting for MISSION_REQUEST")
        seq = getattr(req, 'seq', getattr(req, 'seq', None))
        it = pending.pop(seq, None)
        if it is None:
            continue
        m.mav.mission_item_int_send(
            m.target_system,
            m.target_component,
            it['seq'],
            it['frame'],
            it['cmd'],
            it['current'],
            it['autocontinue'],
            it['p1'], it['p2'], it['p3'], it['p4'],
            it['x'], it['y'], it['z'],
            mavutil.mavlink.MAV_MISSION_TYPE_MISSION
        )
    ack = m.recv_match(type='MISSION_ACK', blocking=True, timeout=5)
    if not ack or ack.type != mavutil.mavlink.MAV_MISSION_ACCEPTED:
        raise RuntimeError(f"Mission upload failed: {ack}")

def mission_start(m, start_seq=0):
    # Set first item (defensive)
    m.mav.mission_set_current_send(m.target_system, m.target_component, start_seq)
    # Switch to AUTO and start
    set_mode(m, 'AUTO')
    m.mav.command_long_send(
        m.target_system, m.target_component,
        mavutil.mavlink.MAV_CMD_MISSION_START, 0,
        0, 0, 0, 0, 0, 0, 0)

def main():
    m = mavutil.mavlink_connection(UDP)
    wait_heartbeat(m)

    # Get current location for TAKEOFF/WP seeds
    lat0, lon0, _ = get_position(m)
    print(lat0, lon0)
    # Build mission
    items = []

    # 0: TAKEOFF to 10 m at current location
    items.append(dict(
        seq=0,
        frame=mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
        cmd=mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
        current=1,
        autocontinue=1,
        p1=0, p2=0, p3=0, p4=float('nan'),
        x=int(lat0 * 1e7),
        y=int(lon0 * 1e7),
        z=10.0
    ))

    # 1: WP1 ~30 m North, 0 m East, alt 15 m
    dlat, dlon = meters_to_deg(5, 0, lat0)
    items.append(dict(
        seq=1,
        frame=mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
        cmd=mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
        current=0,
        autocontinue=1,
        p1=0, p2=0, p3=0, p4=0,
        x=int((lat0 + dlat) * 1e7),
        y=int((lon0 + dlon) * 1e7),
        z=5.0
    ))

    # 2: WP2 ~0 m North, 30 m East, alt 15 m
    dlat, dlon = meters_to_deg(0, 5, lat0)
    items.append(dict(
        seq=2,
        frame=mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
        cmd=mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
        current=0,
        autocontinue=1,
        p1=0, p2=0, p3=0, p4=0,
        x=int((lat0 + dlat) * 1e7),
        y=int((lon0 + dlon) * 1e7),
        z=5.0
    ))


    time.sleep(1)

    mission_upload(m, items)
    mission_start(m, 0)

    print("Mission started.")

if __name__ == "__main__":
    main()
