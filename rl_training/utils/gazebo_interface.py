import subprocess
import time
import os
import signal
import psutil
import logging
import xml.etree.ElementTree as ET
import json
from typing import Dict, Any
import threading


logger = logging.getLogger("GazeboInterface")
logger.setLevel(logging.DEBUG)  # Default level; can be overridden in your experiment runner


class GazeboInterface:
    """
    Interface for modern Gazebo (gz sim) simulation control.
    Designed for RL integration with ArduPilot SITL and Gazebo.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.process = None
        self.is_started = False
        
        self.sdf_file = config.get('sdf_file', None)
        print(f"SDF file: {self.sdf_file}")
        self.world_name = self._parse_world_name(self.sdf_file)

        if self.sdf_file and not os.path.exists(self.sdf_file):
            raise FileNotFoundError(f"SDF file not found: {self.sdf_file}")
        
        self.gui = config.get('gui', True)
        self.verbose = config.get('verbose', True)
        self.timeout = config.get('timeout', 30.0)

        # not used yet
        self.step_size = config.get('step_size', 0.001)

        self.extra_args = config.get('extra_args', [])
        self.sim_time = 0.0
        self._timer_lock = threading.Lock()



    def start_simulation(self):
        if self.is_running():
            raise RuntimeError("Gazebo simulation already running")
        
        logger.info("Starting Gazebo simulation...")
        
        cmd = ['gz', 'sim']
        if not self.gui:
            cmd.append('-s')
        if self.sdf_file:
            cmd.append(self.sdf_file)
        if self.verbose:
            cmd.append('-v 4')
        if self.extra_args:
            cmd.extend(self.extra_args)
        
        logger.debug(f"Command: {' '.join(cmd)}")
        
        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid
            )
            
            logger.info(f"Gazebo started with PID: {self.process.pid}")
            self._wait_for_startup()
            self.is_started = True
            logger.info("Gazebo simulation started successfully!")

            self._timer_thread = threading.Thread(target=self._timer_thread, daemon=True)
            self._timer_thread.start()
        except FileNotFoundError:
            logger.error("gz command not found. Please install Gazebo Garden or newer.")
            raise
        except Exception as e:
            logger.exception("Failed to start Gazebo")
            raise RuntimeError(f"Failed to start Gazebo: {e}")
    
    def _wait_for_startup(self):
        logger.info("Waiting for Gazebo to initialize...")
        start_time = time.time()

        while time.time() - start_time < self.timeout:
            if self.process.poll() is not None:
                stdout, stderr = self.process.communicate()
                raise RuntimeError(f"Gazebo failed to start. Error: {stderr.decode()}")
            try:
                result = subprocess.check_output(["gz", "topic", "--list"])
                if b"/world/" in result:
                    logger.info("Gazebo is responding.")
                    return
            except subprocess.CalledProcessError:
                pass
            time.sleep(1)

        self.stop_simulation()
        raise RuntimeError(f"Gazebo startup timeout after {self.timeout} seconds")

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

    def stop_simulation(self):
        if self.process is None:
            return
        
        logger.info("Stopping Gazebo simulation...")
        
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
        logger.info("Gazebo simulation stopped.")
    
    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None
    
    def get_process_info(self) -> Dict[str, Any]:
        if not self.is_running():
            return {"status": "not_running"}
        
        try:
            process = psutil.Process(self.process.pid)
            return {
                "status": "running",
                "pid": self.process.pid,
                "cpu_percent": process.cpu_percent(),
                "memory_mb": process.memory_info().rss / 1024 / 1024,
                "create_time": process.create_time()
            }
        except psutil.NoSuchProcess:
            return {"status": "process_not_found"}
    
    def transport_position(self, name, position, orientation):
        try:
            command = f"gz service -s /world/{self.world_name}/set_pose --reqtype gz.msgs.Pose --reptype gz.msgs.Boolean --timeout 300 --req 'name: {name}, position: {{x: {position[0]}, y: {position[1]}, z: {position[2]}}}, orientation: {{x: {orientation[0]}, y: {orientation[1]}, z: {orientation[2]}, w: {orientation[3]}}}'"
            logger.info(f"Command: {command}")
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
            logger.info("✅ Drone transport successful.")
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Drone transport failed: {e.stderr.decode()}")
            raise RuntimeError("Gazebo transport service failed.")
 
    def pause_simulation(self):
        logger.info(f"⏸️  Pausing simulation in world: {self.world_name}")
        try:
            subprocess.run(
                [
                    "gz", "service",
                    "-s", f"/world/{self.world_name}/control",
                    "--reqtype", "gz.msgs.WorldControl",
                    "--reptype", "gz.msgs.Boolean",
                    "--timeout", "3000",
                    "--req", "pause: true"
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            logger.info("✅ Simulation paused.")
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to pause simulation: {e.stderr.decode()}")
            raise RuntimeError("Gazebo pause service failed.")

    def resume_simulation(self):
        logger.info(f"▶️  Resuming simulation in world: {self.world_name}")
        try:
            subprocess.run(
                [
                    "gz", "service",
                    "-s", f"/world/{self.world_name}/control",
                    "--reqtype", "gz.msgs.WorldControl",
                    "--reptype", "gz.msgs.Boolean",
                    "--timeout", "3000",
                    "--req", "pause: false"
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            logger.info("✅ Simulation resumed.")
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to resume simulation: {e.stderr.decode()}")
            raise RuntimeError("Gazebo resume service failed.")
   
    def get_sim_time(self) -> float:
        with self._timer_lock:
            return self.sim_time

    def close(self):
        self.stop_simulation()
    
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
