import subprocess
import time
import os
import signal
import logging
import xml.etree.ElementTree as ET
import json
from typing import Dict, Any
import threading

from gz.transport13 import Node
from gz.msgs10.world_control_pb2 import WorldControl
from gz.msgs10.boolean_pb2 import Boolean
from gz.msgs10.pose_pb2 import Pose
from gz.msgs10.clock_pb2 import Clock
from gz.msgs10.world_stats_pb2 import WorldStatistics





logger = logging.getLogger("GazeboInterface")


class GazeboInterface:
    
    def __init__(self, config: Dict[str, Any], instance, verbose):
        if verbose:
            logger.setLevel(logging.INFO) 
        else:
            logger.setLevel(logging.ERROR)

        self.config = config
        self.process = None
        self.is_started = False
        self.instance = instance
        self.sdf_file = config.get('sdf_file')
        
        # If self.instance == 2, modify the sdf_file path to append _2 before .sdf
        if self.instance == 2:
            os.environ["GZ_PARTITION"] = "gz_i1"
            if self.sdf_file.endswith('.sdf'):
                self.sdf_file = self.sdf_file[:-4] + '_2.sdf'
        else:
            os.environ["GZ_PARTITION"] = "gz_i0"
      
        self.node = Node()
        self.clock_node = Node()
   
        #logger.info(f"SDF file: {self.sdf_file}")
        self.world_name = self._parse_world_name(self.sdf_file)

        if self.sdf_file and not os.path.exists(self.sdf_file):
            raise FileNotFoundError(f"SDF file not found: {self.sdf_file}")
        
        self.gui = config.get('gui')
        self.int_verbose = config.get('verbose')    
        
        # for time synchronization 
        self.sim_time = 0.0
        self._timer_lock = threading.Lock()
        self._clock_event = threading.Event()
        self._clock_sub = self.clock_node.subscribe(WorldStatistics, f"/world/{self.world_name}/stats", self._on_clock)

    def start_simulation(self):
        if self.is_running():
            raise RuntimeError("Gazebo simulation already running")
        
        # logger.debug("Starting Gazebo simulation...")
        
        cmd = ['gz', 'sim']
        if not self.gui:
            cmd.append('-s')
        if self.sdf_file:
            cmd.append(self.sdf_file)
        if self.int_verbose:
            cmd.append('-v 4')
        
        # logger.debug(f"Command: {' '.join(cmd)}")
        
        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid
            )
            
            self._wait_for_startup()
            self.is_started = True
            #logger.debug("Gazebo simulation started successfully!")

            #logger.debug("Timer thread started")
        except FileNotFoundError:
            logger.error("gz command not found. Please install Gazebo Harmonic or newer.")
            raise
        except Exception as e:
            logger.exception("Failed to start Gazebo")
            raise RuntimeError(f"Failed to start Gazebo: {e}")
    
    def is_running(self):
        return self.process is not None and self.process.poll() is None
    
    def transport_position(self, name, position, orientation, timeout_ms=1000, retries=2):

        svc = f"/world/{self.world_name}/set_pose"
        pose = Pose()
        pose.name = str(name)
        pose.position.x, pose.position.y, pose.position.z = map(float, position)
        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = map(float, orientation)

        for _ in range(max(1, retries)):
            ok, rep = self.node.request(svc, pose, Pose, Boolean, timeout_ms)  # (service, request, response_type, timeout_ms)
            if ok and getattr(rep, "data", False):
                return True
        raise RuntimeError(f"set_pose failed on {svc} (partition={getattr(self,'partition', None)})")
    
    def pause_simulation(self):
        #logger.debug(f"⏸️  Pausing simulation in world: {self.world_name}")
        svc = f"/world/{self.world_name}/control"
        req = WorldControl()
        req.pause = True                # True=pause, False=run
        ok, rep = self.node.request(svc, req, WorldControl,  Boolean, timeout=2000)
        if not ok:
            raise RuntimeError(f"Service call failed: {svc}")

    def resume_simulation(self):
        svc = f"/world/{self.world_name}/control"
        req = WorldControl()
        req.pause = False                   # True=pause, False=run
        ok, rep = self.node.request(svc, req, WorldControl,  Boolean, timeout=2000)
        if not ok:
            raise RuntimeError(f"Service call failed: {svc}")
        # SOLUTION 1: Use world state service to get simulation time
    

    
    def get_sim_time(self, wait_ms=1000):
        if not self._clock_event.is_set():
            self._clock_event.wait(wait_ms / 1000.0)
        with self._timer_lock:
            return float(self.sim_time)
        


    def _on_clock(self, msg: WorldStatistics):
        t = int(msg.sim_time.sec) + int(msg.sim_time.nsec) * 1e-9
        with self._timer_lock:
            self.sim_time = t
        if not self._clock_event.is_set():
            self._clock_event.set()


    def close(self):
        if self.process is None:
            return
        #logger.debug("Stopping Gazebo simulation...")
        
        try:
            if self.process.poll() is None:
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    logger.warning("Graceful shutdown timeout, forcing termination...")
                    os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                    self.process.wait()
        except ProcessLookupError:
            pass
        except Exception as e:
            logger.error(f"Error stopping Gazebo: {e}")
        
        self.process = None
        self.is_started = False
        #logger.debug("Gazebo simulation stopped.")

    def _parse_world_name(self, sdf_path: str) -> str:
        try:
            tree = ET.parse(sdf_path)
            root = tree.getroot()
            for world in root.iter("world"):
                name = world.attrib.get("name")
                if name:
                    #logger.debug(f"Parsed world name: {name}")
                    return name
            raise ValueError("No <world> tag with name attribute found in SDF.")
        except Exception as e:
            logger.error(f"Failed to parse world name from SDF: {e}")
            raise RuntimeError(f"Failed to parse SDF world name: {e}")

    def _wait_for_startup(self):
        #logger.debug("Waiting for Gazebo to initialize...")
        start_time = time.time()

        while time.time() - start_time < 10:    # timeout = 10
            if self.process.poll() is not None:
                stdout, stderr = self.process.communicate()
                raise RuntimeError(f"Gazebo failed to start. Error: {stderr.decode()}")
            try:
                result = subprocess.check_output(["gz", "topic", "--list"])
                if b"/world/" in result:
                    #logger.debug("Gazebo is responding.")
                    return
            except subprocess.CalledProcessError:
                pass
            time.sleep(0.1)

        self.close()
        raise RuntimeError(f"Gazebo startup timeout after {10} seconds")
