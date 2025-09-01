#!/usr/bin/env python3
# Pause/resume Gazebo Harmonic via gz-transport service
# Usage: python3 world_ctrl.py simple_world pause|resume

import sys
from gz.transport13 import Node
from gz.msgs10.world_control_pb2 import WorldControl
from gz.msgs10.boolean_pb2 import Boolean

def world_control(world: str, pause_flag: bool) -> None:
    svc = f"/world/{world}/control"
    node = Node()

    req = WorldControl()
    req.pause = pause_flag              # True=pause, False=run

    ok, rep = node.request(svc, req, WorldControl,  Boolean, timeout=3000)
    if not ok:
        raise RuntimeError(f"Service call failed: {svc}")

if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[2] not in ("pause","resume"):
        print("Usage: python3 world_ctrl.py <world> pause|resume"); sys.exit(1)
    world_control(sys.argv[1], pause_flag=(sys.argv[2]=="pause"))
    print("Done.")
