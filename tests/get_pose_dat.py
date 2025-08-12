from gz.msgs10.pose_v_pb2 import Pose_V
from gz.transport13 import Node
import time

def pose_callback(msg):
    print(msg)
    # for pose in msg.pose:
    #     if pose.name == "iris_with_ardupilot":
    #         pos = pose.position
    #         ori = pose.orientation
    #         print(f"Pos: ({pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f}) "
    #               f"Quat: ({ori.x:.3f}, {ori.y:.3f}, {ori.z:.3f}, {ori.w:.3f})")

def main():
    node = Node()
    topic = "/model/iris_with_ardupilot/pose"
    if not node.subscribe(Pose_V, topic, pose_callback):
        print(f"Failed to subscribe to {topic}")
        return
    print(f"Listening to {topic}...")
    try:
        while True:
            time.sleep(0.0002)
            # pass  # keep process alive
    except KeyboardInterrupt:
        print("\nShutting down.")

if __name__ == "__main__":
    main()
