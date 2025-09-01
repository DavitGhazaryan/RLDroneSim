#!/usr/bin/env python3
import threading, time
from gz.transport13 import Node
from gz.msgs10.clock_pb2 import Clock as ClockMsg
from gz.msgs10.world_stats_pb2 import WorldStatistics

sim_time = 0.0
event = threading.Event()

def on_clock(msg: WorldStatistics):
    global sim_time
    sim_time = int(msg.sim.sec) + int(msg.sim.nsec) * 1e-9
    if not event.is_set():
        event.set()

def main():
    node = Node()
    topic = "/world/simple_world/clock"   # replace with your world name
    sub = node.subscribe(topic, ClockMsg, on_clock)

    # wait until we get at least one clock message
    if not event.wait(1.0):
        print("No clock message received within 1s")
        return

    # now print sim time every second
    while True:
        print(f"Sim time: {sim_time:.6f} s")
        time.sleep(1)

if __name__ == "__main__":
    main()
