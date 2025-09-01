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




logger = logging.getLogger("GazeboInterface")
# logger.setLevel(logging.INFO)  # Default level; can be overridden in your experiment runner


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
   
        logger.info(f"SDF file: {self.sdf_file}")
        self.world_name = self._parse_world_name(self.sdf_file)

        if self.sdf_file and not os.path.exists(self.sdf_file):
            raise FileNotFoundError(f"SDF file not found: {self.sdf_file}")
        
        self.gui = config.get('gui')
        self.int_verbose = config.get('verbose')    
        
        # for time synchronization 
        self.sim_time = 0.0
        self._timer_lock = threading.Lock()

    def start_simulation(self):
        if self.is_running():
            raise RuntimeError("Gazebo simulation already running")
        
        logger.debug("Starting Gazebo simulation...")
        
        cmd = ['gz', 'sim']
        if not self.gui:
            cmd.append('-s')
        if self.sdf_file:
            cmd.append(self.sdf_file)
        if self.int_verbose:
            cmd.append('-v 4')
        
        logger.debug(f"Command: {' '.join(cmd)}")
        
        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid
            )
            
            self._wait_for_startup()
            self.is_started = True
            logger.debug("Gazebo simulation started successfully!")

            self._timer_thread = threading.Thread(target=self._timer_thread, daemon=True)
            self._timer_thread.start()
            logger.debug("Timer thread started")
        except FileNotFoundError:
            logger.error("gz command not found. Please install Gazebo Harmonic or newer.")
            raise
        except Exception as e:
            logger.exception("Failed to start Gazebo")
            raise RuntimeError(f"Failed to start Gazebo: {e}")
    
    def is_running(self):
        return self.process is not None and self.process.poll() is None

    def transport_position(self, name, position, orientation):
        try:
            command = f"gz service -s /world/{self.world_name}/set_pose --reqtype gz.msgs.Pose --reptype gz.msgs.Boolean --timeout 300 --req 'name: {name}, position: {{x: {position[0]}, y: {position[1]}, z: {position[2]}}}, orientation: {{x: {orientation[0]}, y: {orientation[1]}, z: {orientation[2]}, w: {orientation[3]}}}'"
            logger.debug(f"Command: {command}")
            subprocess.run(
                [
                    "gz", "service",
                    "-s", f"/world/{self.world_name}/set_pose",
                    "--reqtype", "gz.msgs.Pose",
                    "--reptype", "gz.msgs.Boolean",
                    "--timeout", "300",
                    "--req", f"name: '{name}', position: {{x: {position[0]}, y: {position[1]}, z: {position[2]}}}, orientation: {{x: {orientation[0]}, y: {orientation[1]}, z: {orientation[2]}, w: {orientation[3]}}}"
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            logger.debug("✅ Drone transport successful.")
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Drone transport failed: {e.stderr.decode()}")
            raise RuntimeError("Gazebo transport service failed.")
 
    def pause_simulation(self):
        # if self.instance == 2:
        #     os.environ["GZ_PARTITION"] = "gz_i1"
        # else:
        #     os.environ["GZ_PARTITION"] = "gz_i0"
        logger.debug(f"⏸️  Pausing simulation in world: {self.world_name}")
        svc = f"/world/{self.world_name}/control"
        req = WorldControl()
        req.pause = True                # True=pause, False=run
        ok, rep = self.node.request(svc, req, WorldControl,  Boolean, timeout=2000)
        if not ok:
            raise RuntimeError(f"Service call failed: {svc}")

    def resume_simulation(self):
        # if self.instance == 2:
        #     os.environ["GZ_PARTITION"] = "gz_i1"
        # else:
        #     os.environ["GZ_PARTITION"] = "gz_i0"
        logger.debug(f"⏸️  Pausing simulation in world: {self.world_name}")
        svc = f"/world/{self.world_name}/control"
        req = WorldControl()
        req.pause = False                   # True=pause, False=run
        ok, rep = self.node.request(svc, req, WorldControl,  Boolean, timeout=2000)
        if not ok:
            raise RuntimeError(f"Service call failed: {svc}")

    # def pause_simulation(self):
    #     logger.debug(f"⏸️  Pausing simulation in world: {self.world_name}")
    #     try:
    #         subprocess.run(
    #             [
    #                 "gz", "service",
    #                 "-s", f"/world/{self.world_name}/control",
    #                 "--reqtype", "gz.msgs.WorldControl",
    #                 "--reptype", "gz.msgs.Boolean",
    #                 "--timeout", "3000",
    #                 "--req", "pause: true"
    #             ],
    #             check=True,
    #             stdout=subprocess.PIPE,
    #             stderr=subprocess.PIPE
    #         )
    #         logger.debug("✅ Simulation paused.")
    #     except subprocess.CalledProcessError as e:
    #         logger.error(f"❌ Failed to pause simulation: {e.stderr.decode()}")
    #         raise RuntimeError("Gazebo pause service failed.")

    # def resume_simulation(self):
    #     logger.debug(f"▶️  Resuming simulation in world: {self.world_name}")
    #     try:
    #         subprocess.run(
    #             [
    #                 "gz", "service",
    #                 "-s", f"/world/{self.world_name}/control",
    #                 "--reqtype", "gz.msgs.WorldControl",
    #                 "--reptype", "gz.msgs.Boolean",
    #                 "--timeout", "3000",
    #                 "--req", "pause: false"
    #             ],
    #             check=True,
    #             stdout=subprocess.PIPE,
    #             stderr=subprocess.PIPE
    #         )
    #         logger.debug("✅ Simulation resumed.")
    #     except subprocess.CalledProcessError as e:
    #         logger.error(f"❌ Failed to resume simulation: {e.stderr.decode()}")
    #         raise RuntimeError("Gazebo resume service failed.")
   
    def get_sim_time(self):
        """
        Called from the environment to get the simulation time.
        """
        with self._timer_lock:
            return self.sim_time

    def close(self):
        if self.process is None:
            return
        logger.debug("Stopping Gazebo simulation...")
        
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
        self._timer_thread.join(timeout=1)
        logger.debug("Gazebo simulation stopped.")
    
    # Private methods
    def _parse_world_name(self, sdf_path: str) -> str:
        try:
            tree = ET.parse(sdf_path)
            root = tree.getroot()
            for world in root.iter("world"):
                name = world.attrib.get("name")
                if name:
                    logger.debug(f"Parsed world name: {name}")
                    return name
            raise ValueError("No <world> tag with name attribute found in SDF.")
        except Exception as e:
            logger.error(f"Failed to parse world name from SDF: {e}")
            raise RuntimeError(f"Failed to parse SDF world name: {e}")

    def _timer_thread(self):
        proc = subprocess.Popen(
            ["gz", "topic", "-e", "--topic", "/clock", "--json-output"],
            stdout=subprocess.PIPE,
            text=True
        )
        for line in proc.stdout:
            if not self.is_running():
                logger.debug("Gazebo is not running, stopping timer thread")
                break
            try:
                msg = json.loads(line)
                sec = msg["system"]["sec"]
                nsec = msg["system"]["nsec"]
                t = int(sec) + int(nsec)     * 1e-9
                with self._timer_lock:
                    self.sim_time = t
            except json.JSONDecodeError:
                continue

    def _wait_for_startup(self):
        logger.debug("Waiting for Gazebo to initialize...")
        start_time = time.time()

        while time.time() - start_time < 10:    # timeout = 10
            if self.process.poll() is not None:
                stdout, stderr = self.process.communicate()
                raise RuntimeError(f"Gazebo failed to start. Error: {stderr.decode()}")
            try:
                result = subprocess.check_output(["gz", "topic", "--list"])
                if b"/world/" in result:
                    logger.debug("Gazebo is responding.")
                    return
            except subprocess.CalledProcessError:
                pass
            time.sleep(0.1)

        self.close()
        raise RuntimeError(f"Gazebo startup timeout after {10} seconds")
