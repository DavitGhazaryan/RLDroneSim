import time
from gz.msgs10.world_control_pb2 import WorldControl
from gz.msgs10.boolean_pb2 import Boolean
from gz.msgs10.empty_pb2 import Empty
from gz.msgs10.double_pb2 import Double

from gz.transport13 import Node

def pause_sim(node, world):
    svc = f"/world/{world}/control"
    req = WorldControl()
    req.pause = True
    ok, rep = node.request(svc, req, WorldControl, Boolean, timeout=2000)
    return ok and getattr(rep, "data", False)

def resume_sim(node, world):
    svc = f"/world/{world}/control"
    req = WorldControl()
    req.pause = False
    ok, rep = node.request(svc, req, WorldControl, Boolean, timeout=2000)
    return ok and getattr(rep, "data", False)

def get_sim_time(node):
    ok, rep = node.request("/sim_time", Empty(), Empty, Double, timeout=2000)
    if ok:
        print("sim_time:", rep.data)
    else:
        print("request failed")


if __name__ == "__main__":
    world_name = "simple_world"   # change to your world
    node = Node()

    interval = 0.03           # seconds between pause/resume
    count = 0
    while True:
        print(count)
        # if not pause_sim(node, world_name):
        #     print("Pause failed")
        print(get_sim_time(node))
        count += 1
        time.sleep(interval)

        # if not resume_sim(node, world_name):
        #     print("Resume failed")

        time.sleep(interval)
