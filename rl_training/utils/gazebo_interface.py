import subprocess
import time
import os
import signal
import xml.etree.ElementTree as ET
from typing import Dict, Any

from gz.transport13 import Node
from gz.msgs10.world_control_pb2 import WorldControl
from gz.msgs10.boolean_pb2 import Boolean
from gz.msgs10.pose_pb2 import Pose
# from gz.msgs10.pose_v_pb2 import Pose_V
from gz.msgs10.empty_pb2 import Empty
from gz.msgs10.double_pb2 import Double


class GazeboInterface:
    """
    Provides interface to control Gazebo Simulation. 
    Reads the config file, starts gazebo simulation.
    Initiates communication with simulator using gz transport library.
    """

    def __init__(self, config, instance, verbose):     
        self._node = Node()      # gz transportation node
        self._config = config
        self._instance = instance
        self._sdf_file = config.get('sdf_file')
        
        #  modify the sdf_file path to append _2 before .sdf
        if self._instance == 2:
            os.environ["GZ_PARTITION"] = "gz_i1"
            if self._sdf_file.endswith('.sdf'):
                self._sdf_file = self._sdf_file[:-4] + '_2.sdf'
        # else:
        #     os.environ["GZ_PARTITION"] = "gz_i0"

        if self._sdf_file and not os.path.exists(self._sdf_file):
            raise FileNotFoundError(f"SDF file not found: {self._sdf_file}")

        self._world_name = self._parse_world_name(self._sdf_file)        
        self._process = None

    def start_simulation(self):
        if self._process is not None and self._process.poll() is None:
            raise RuntimeError("Gazebo simulation already running")
        
        
        cmd = ['gz', 'sim']
        
        if not self._config.get("gui"):
            cmd.append('-s')
        if self._sdf_file:                # use this explicitly as can be modified
            cmd.append(self._sdf_file)
        if self._config.get('verbose'):
            cmd.append('-v 4')
        cmd.append("--physics-engine=gz-physics-dartsim-plugin")

        env = os.environ.copy()
        env["GZ_IP"] = "127.0.0.1"
        env["GZ_TRANSPORT_TOPIC_STATISTICS"] = "0"

        try:

            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=False,
                env=env
                )

            self._wait_for_startup()
        except Exception as e:
            raise RuntimeError(f"Failed to start Gazebo: {e}")

    def transport_position(self, name, position, orientation, timeout_ms=1000, retries=10):
        svc = f"/world/{self._world_name}/set_pose"
        pose = Pose()
        pose.name = str(name)
        pose.position.x, pose.position.y, pose.position.z = map(float, position)
        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = map(float, orientation)

        for _ in range(max(1, retries)):
            ok, rep = self._node.request(svc, pose, Pose, Boolean, timeout_ms)  # (service, request, response_type, timeout_ms)
            if ok and getattr(rep, "data", False):
                return True
            else:
                print("Transport failed")
        raise RuntimeError(f"set_pose failed on {svc} (partition={getattr(self,'partition', None)})")
    
    def pause_simulation(self, retries=4):
        svc = f"/world/{self._world_name}/control"
        req = WorldControl()
        req.pause = True                # True=pause, False=run
        for _ in range(max(1, retries)):
            ok, rep = self._node.request(svc, req, WorldControl,  Boolean, timeout=2000) # (service, request, response_type, timeout_ms)
            if ok and getattr(rep, "data", False):
                return True
            else:
                print("Pause failed")
        raise RuntimeError(f"Service call failed while pausing: {svc}")

    def resume_simulation(self, retries=4):
        svc = f"/world/{self._world_name}/control"
        req = WorldControl()
        req.pause = False                   # True=pause, False=run
        for _ in range(max(1, retries)):
            ok, rep = self._node.request(svc, req, WorldControl,  Boolean, timeout=2000) # (service, request, response_type, timeout_ms)
            if ok and getattr(rep, "data", False):
                return True
            else:
                print(ok)
                print(rep)
                print("Resume failed")
        raise RuntimeError(f"Service call failed while resuming: {svc}")
    
    def get_sim_time(self, wait_ms=1000):
        ok, rep = self._node.request("/sim_time", Empty(), Empty, Double, timeout=2000)
        if ok:
            return rep.data
        else:
            print(" time request failed")

    def close(self):
        if self._process is None:
            return
        
        try:
            if self._process.poll() is None:
                os.killpg(os.getpgid(self._process.pid), signal.SIGTERM)
                out, err = self._process.communicate(timeout=1)
                try:
                    self._process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(os.getpgid(self._process.pid), signal.SIGKILL)
                    self._process.wait()
        except ProcessLookupError:
            pass
        except Exception as e:
            pass
        self._process = None

    def _parse_world_name(self, sdf_path: str) -> str:
        """
        Parse an SDF file and return the name of the first <world> tag.
        Returns: Name of the world specified in the <world name="..."> attribute.
        """
        try:
            tree = ET.parse(sdf_path)
            root = tree.getroot()
            for world in root.iter("world"):
                name = world.attrib.get("name")
                if name:
                    return name
            raise ValueError("No <world> tag with name attribute found in SDF.")
        except Exception as e:
            raise RuntimeError(f"Failed to parse SDF world name: {e}")

    def _wait_for_startup(self):
        start_time = time.time()

        while time.time() - start_time < 10:    # timeout = 10
            if self._process.poll() is not None:
                stdout, stderr = self._process.communicate()
                raise RuntimeError(f"Gazebo failed to start. Error: {stderr.decode()}")
            try:
                result = subprocess.check_output(["gz", "topic", "--list"])
                if b"/world/" in result:
                    return
            except subprocess.CalledProcessError:
                pass
            time.sleep(0.5)

        self.close()
        raise RuntimeError(f"Gazebo startup timeout after {10} seconds")
