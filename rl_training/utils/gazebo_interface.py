import subprocess
import time
import os
import signal
import psutil
import logging
import xml.etree.ElementTree as ET
import json
from typing import Dict, Any


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
        self.world_name = self._parse_world_name(self.sdf_file)

        if self.sdf_file and not os.path.exists(self.sdf_file):
            raise FileNotFoundError(f"SDF file not found: {self.sdf_file}")
        
        self.gui = config.get('gui', True)
        self.verbose = config.get('verbose', True)
        self.timeout = config.get('timeout', 30.0)
        
        # not used yet
        self.real_time_factor = config.get('real_time_factor', 1.0)
        self.step_size = config.get('step_size', 0.001)

        self.extra_args = config.get('extra_args', [])
                
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
        logger.info("Gazebo simulation stopped.")
    
    def restart_simulation(self):
        logger.info("Restarting Gazebo simulation...")
        self.stop_simulation()
        time.sleep(3)
        self.start_simulation()
    
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
    
    def reset_world(self):
        logger.info(f"🔄 Resetting world: {self.world_name}")
        try:
            subprocess.run(
                [
                    "gz", "service",
                    "-s", f"/world/{self.world_name}/control",
                    "--reqtype", "gz.msgs.WorldControl",
                    "--reptype", "gz.msgs.Boolean",
                    "--timeout", "3000",
                    "--req", "reset: {all: true}"
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            logger.info("✅ World reset successful.")
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ World reset failed: {e.stderr.decode()}")
            raise RuntimeError("Gazebo reset service failed.")
 
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
   
    def get_simulation_time(self) -> float:
        try:
            result = subprocess.check_output(
                [
                    "gz", "topic",
                    "-e", f"/world/{self.world_name}/clock"
                ],
                timeout=2
            )
            msg = json.loads(result.decode("utf-8"))
            sec = msg["sim"]["sec"]
            nsec = msg["sim"]["nsec"]
            sim_time = sec + nsec * 1e-9
            logger.debug(f"⏱️  Simulation time: {sim_time:.3f} seconds")
            return sim_time
        except Exception as e:
            logger.warning(f"⚠️  Failed to get simulation time: {e}")
            return -1.0

    def get_model_pose(self, model_name: str) -> Dict[str, float]:
        import math
        """
        Get the pose of a model from /world/<world_name>/pose/info topic.

        Don't use this function for reward calculation as you have GPS.
        """
        try:
            output = subprocess.check_output(
                [
                    "gz", "topic", "-e",
                    "-n", "1",  # receive only 1 message
                    "--topic", f"/world/{self.world_name}/pose/info"
                ],
                timeout=5
            )
            msg = json.loads(output.decode("utf-8"))
            for pose in msg.get("pose", []):
                if pose.get("name") == model_name:
                    pos = pose["position"]
                    rot = pose["orientation"]

                    # Quaternion → Euler
                    qx, qy, qz, qw = rot["x"], rot["y"], rot["z"], rot["w"]

                    t0 = +2.0 * (qw * qx + qy * qz)
                    t1 = +1.0 - 2.0 * (qx * qx + qy * qy)
                    roll = math.atan2(t0, t1)

                    t2 = +2.0 * (qw * qy - qz * qx)
                    t2 = max(min(t2, 1.0), -1.0)
                    pitch = math.asin(t2)

                    t3 = +2.0 * (qw * qz + qx * qy)
                    t4 = +1.0 - 2.0 * (qy * qy + qz * qz)
                    yaw = math.atan2(t3, t4)

                    return {
                        "x": pos["x"],
                        "y": pos["y"],
                        "z": pos["z"],
                        "roll": roll,
                        "pitch": pitch,
                        "yaw": yaw
                    }

            logger.warning(f"Model '{model_name}' not found in pose info.")
            return {}

        except subprocess.TimeoutExpired:
            logger.error("Timeout while querying model pose.")
            return {}
        except Exception as e:
            logger.error(f"Failed to get model pose: {e}")
            return {}

    def close(self):
        self.stop_simulation()
    
    def __enter__(self):
        self.start_simulation()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close() 

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
