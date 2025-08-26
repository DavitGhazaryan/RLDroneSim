#!/usr/bin/env python3

import subprocess
import time
import os
import signal
import psutil
import logging
import atexit
import threading
import asyncio
import socket
import concurrent.futures
from typing import Dict, Any, List, Optional
from pathlib import Path

from pymavlink import mavutil

from mavsdk import System



logger = logging.getLogger("SITL")  

class ArduPilotSITL:
    """
    Implements the Async interface for Ardupilot SITL.

    _reset_async()
    set_params_async()
    get_pid_params_async()
    get_pid_param_async()
    set_pid_param_async()

    """

    def __init__(self, config: Dict[str, Any], verbose=True):
        if verbose:
            logging.basicConfig(level=logging.INFO)
        else:
            logging.basicConfig(level=logging.ERROR)

        self.config = config
        self.process: Optional[subprocess.Popen] = None
        self.child_processes: List[int] = []
        self.log_threads: List[threading.Thread] = []  # Store references to logging threads
        self._shutdown_event = threading.Event()  # Signal for thread shutdown
        self._thread_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="SITL")

        # cached connections
        self._mavlink_master: Optional[mavutil.mavlink_connection] = None  # Cached low level connection 
        self._mavsdk_system: Optional[System] = None  # Cached MAVSDK connection

        # required
        self.ardupilot_path = Path(config['ardupilot_path'])
        self.vehicle         = 'ArduCopter'
        self.frame           = config.get('frame')
        self.ideal_sensors   = config.get('ideal_sensors')
        # optional
        self.name             = config.get('name')
        self.instance         = config.get('instance')
        self.count            = config.get('count')
        self.location_str     = config.get('location')
        self.speedup          = config.get('speedup')
        self.wipe              = config.get('wipe_eeprom')
        self.use_dir           = config.get('use_dir')
        self.delay_start       = config.get('delay_start')
        self.model             = config.get('model')
        self.clean             = config.get('clean')
        self.no_rebuild        = config.get('no_rebuild')
        self.no_configure      = config.get('no_configure')
        self.no_mavproxy       = config.get('no_mavproxy')
        self.udp               = config.get('udp')
        self.udp_out           = config.get('udp_out')
        self.map               = config.get('map')
        self.console           = config.get('console')
        self.mavproxy_args     = config.get('mavproxy_args')
        self.timeout           = config.get('timeout')
        self.min_startup_delay = config.get('min_startup_delay')
        self.master_port       = config.get('master_port')  # Configurable MAVLink port
        self.mavsdk_port       = config.get('mavsdk_port')  # Configurable MAVSDK port
        self.port_check_timeout = config.get('port_check_timeout')  # Timeout for port availability


        # parse home location
        if self.location_str:
            parts = [float(x) for x in self.location_str.split(',')]
            assert len(parts) == 4, "location must be 'lat,lon,alt,yaw'"
            self.home = tuple(parts)
        else:
            self.home = None

        self.sim_vehicle_script = Path(config.get(
            'sim_vehicle_script',
            self.ardupilot_path / 'Tools' / 'autotest' / 'sim_vehicle.py'
        ))

        self._default_params = self._load_default_params()
        atexit.register(self._cleanup_on_exit)
        self._validate_paths()

    def start_sitl(self):
        if self.is_running():
            raise RuntimeError("SITL already running")
        cmd = self._build_command()
        logger.debug(f"Launching SITL")
        logger.debug(f"{cmd}")

        self.process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            preexec_fn=os.setsid, cwd=str(self.ardupilot_path)
        )
        self._shutdown_event.clear()  # set to False, enables logging threads to run
        self._start_log_threads()
        self._wait_for_startup()      # ensures that the process is running and the port(s) are available
        self._track_child_processes()

        logger.debug("Establishing connections...")
        self._get_mavlink_connection()
        logger.debug("MAVLink connection established")

        try:
            self._set_mode_sync('GUIDED')
        except Exception as e:
            logger.warning(f"Error setting GUIDED mode after startup: {e}")



        logger.debug("SITL started successfully.")

    def set_message_interval(self, master, msg_id, hz):
        # master = self._get_mavlink_connection()
        interval_us = int(1e6 / hz)
        master.mav.command_long_send(
            master.target_system, master.target_component,
            mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
            float(msg_id), float(interval_us), 0, 0, 0, 0, 0)
        master.recv_match(type="COMMAND_ACK", blocking=False, timeout=0.5)      

    def set_param_and_confirm(self, master, name_str, value, timeout=3.0):
        # master = self._get_mavlink_connection()
        name_bytes = name_str.encode("ascii", "ignore")
        master.mav.param_set_send(master.target_system, master.target_component,
                            name_bytes, float(value),
                            mavutil.mavlink.MAV_PARAM_TYPE_INT32)
        t0 = time.time()
        while time.time() - t0 < timeout:
            msg = master.recv_match(type="PARAM_VALUE", blocking=True, timeout=timeout)
            if not msg:
                break
            pid = msg.param_id.decode("ascii","ignore").rstrip("\x00") if isinstance(msg.param_id,(bytes,bytearray)) else str(msg.param_id).rstrip("\x00")
            if pid == name_str:
                print(f"PARAM {name_str} => {int(msg.param_value)} (ack)")
                return True
        print(f"WARN: no PARAM_VALUE ack for {name_str}")
        return False


    # Mode Setting
    def _set_mode_sync(self, mode_name: str, timeout: float = 10.0) -> bool:
        """
        Synchronous mode setting using pymavlink (for use in thread executor).
        
        Args:
            mode_name: Name of the mode
            timeout: Timeout in seconds
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            master = self._get_mavlink_connection()
            mapping = master.mode_mapping()

            if mode_name not in mapping:
                logger.error(f"Mode '{mode_name}' not found. Available: {list(mapping.keys())}")
                return False

            mode_id = mapping[mode_name]
            logger.debug(f"Setting mode {mode_name} (ID: {mode_id})")
            
            # Send the mode change
            master.mav.set_mode_send(
                master.target_system,
                mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                mode_id
            )

            # Wait for mode change confirmation with non-blocking reads
            start_time = time.time()
            while time.time() - start_time < timeout:
                if self._shutdown_event.is_set():  # ???
                    return False
                    
                # Non-blocking read
                msg = master.recv_match(type='HEARTBEAT', blocking=False, timeout=0.1)
                if msg and msg.custom_mode == mode_id:
                    logger.debug(f"Mode change to {mode_name} confirmed")
                    logger.debug("   ")
                    return True
                    
                time.sleep(0.1)
            
            logger.warning(f"Mode change to {mode_name} not confirmed within {timeout}s")
            return False
            
        except Exception as e:
            logger.error(f"Failed to set mode {mode_name}: {e}")
            return False

    async def transport_and_reset_async(self, position, quaternion):
        drone = await self._get_mavsdk_connection()
        # reset trigger
        await drone.param.set_param_float("SCR_USER1", 0)
        # set position
        await drone.param.set_param_float("SCR_USER2", position[0])   # x : east
        await drone.param.set_param_float("SCR_USER3", position[1])   # y : north
        await drone.param.set_param_float("SCR_USER4", position[2])   # z :     
        # trigger
        await drone.param.set_param_float("SCR_USER1", 1)

    async def set_param_async(self, param_name: str, value: float):
        drone = await self._get_mavsdk_connection()
        await drone.param.set_param_float(param_name, value)

    async def get_param_async(self, param_name: str):
        drone = await self._get_mavsdk_connection()
        return await drone.param.get_param_float(param_name)

    # Pose Getting
    async def get_pose_gps_async(self):
        drone = await self._get_mavsdk_connection()
        position = await anext(drone.telemetry.position())
        return position
    
    async def get_pose_NED_async(self):
        drone = await self._get_mavsdk_connection()
        pose_vel = await anext(drone.telemetry.position_velocity_ned())
        return pose_vel.position
    

    async def get_attitude_async(self):
        drone = await self._get_mavsdk_connection()
        attitude = await anext(drone.telemetry.attitude_euler())
        return attitude
    
    def is_running(self) -> bool:
        return bool(self.process and self.process.poll() is None)

    def get_mode(self) -> Optional[str]:
        """
        Get the current flight mode of the vehicle using cached connection.
        
        Returns:
            str: Current mode name, or None if unable to determine
        """
        if not self.is_running():
            return None
            
        try:
            master = self._get_mavlink_connection()
            
            # Get a fresh heartbeat message with non-blocking read
            msg = master.recv_match(type='HEARTBEAT', blocking=False, timeout=2.0)
            if msg is None:
                # Try one blocking read as fallback
                msg = master.recv_match(type='HEARTBEAT', blocking=True, timeout=5.0)
                if msg is None:
                    return None
                
            # Get mode mapping and find current mode
            mapping = master.mode_mapping()
            current_mode_id = msg.custom_mode
            
            for mode_name, mode_id in mapping.items():
                if mode_id == current_mode_id:
                    return mode_name
                    
            return f"UNKNOWN_MODE_{current_mode_id}"
            
        except Exception as e:
            logger.error(f"Failed to get current mode: {e}")
            # Connection might be stale, reset it
            self._close_mavlink_connection()
            return None

    def get_process_info(self) -> Dict[str, Any]:
        if not self.is_running():
            return {'status': 'not_running'}
        p = psutil.Process(self.process.pid)
        info = {
            'status':       'running',
            'pid':          p.pid,
            'cpu_percent':  p.cpu_percent(),
            'memory_mb':    p.memory_info().rss / (1024**2),
            'num_children': len(p.children(recursive=True)),
            'uptime_s':     time.time() - p.create_time()
        }
        
        return info

    def stop_sitl(self):
        if not self.is_running():
            return
        logger.debug("Stopping SITL...")
        
        # Close MAVLink connection first
        self._close_mavlink_connection()
        
        # Close MAVSDK connection
        self._close_mavsdk_connection()
        
        try:
            os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            logger.warning("Graceful shutdown timed out; forcing kill")
            os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
            self.process.wait(timeout=5)
        except ProcessLookupError:
            pass
        for pid in list(self.child_processes):
            try:
                psutil.Process(pid).terminate()
            except:
                pass
        
        # Clean up logging threads gracefully
        if hasattr(self, '_shutdown_event'):
            self._shutdown_event.set()  # Signal threads to stop
        
        if hasattr(self, 'log_threads') and self.log_threads:
            for thread in self.log_threads:
                if thread.is_alive():
                    try:
                        thread.join(timeout=2.0)  # Wait up to 2 seconds for thread to finish
                        if thread.is_alive():
                            logger.warning(f"Log thread {thread.name} did not terminate gracefully")
                    except Exception as e:
                        logger.debug(f"Error joining log thread: {e}")
            self.log_threads.clear()
        
        self.process = None
        self.child_processes.clear()
        logger.debug("SITL stopped.")

    # utils for the SITL
    def _validate_paths(self):
        if not self.ardupilot_path.exists():
            raise FileNotFoundError(f"ArduPilot path not found: {self.ardupilot_path}")
        if not self.sim_vehicle_script.exists():
            raise FileNotFoundError(f"sim_vehicle.py not found: {self.sim_vehicle_script}")
        logger.debug(f"Using sim_vehicle.py: {self.sim_vehicle_script}")

    def _load_default_params(self) -> Dict[str, float]:
        # Map vehicle types to their parameter file names
        vehicle_param_map = {
            'ArduCopter': 'copter.parm',
            'ArduPlane': 'plane.parm', 
            'ArduRover': 'rover.parm',
            'ArduSub': 'sub.parm'
        }
        
        # Use frame-specific params if available, otherwise fall back to vehicle default
        frame_param_map = {
            'gazebo-iris': 'gazebo-iris.parm',
            'gazebo-zephyr': 'gazebo-zephyr.parm'
        }
        
        # Choose parameter file: frame-specific > vehicle-specific > copter default
        param_file = frame_param_map.get(self.frame) or vehicle_param_map.get(self.vehicle, 'copter.parm')
        path = self.ardupilot_path / "Tools" / "autotest" / "default_params" / param_file
        defaults: Dict[str, float] = {}
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        defaults[parts[0]] = float(parts[1])
                    except ValueError:
                        pass
        logger.debug(f"Loaded {len(defaults)} default params from {path}")
        return defaults

    def _build_command(self) -> List[str]:

        cmd = [
            'python3', str(self.sim_vehicle_script),
            '-v', self.vehicle,
            '-f', self.frame,
        ]
        if self.clean                   : cmd.append('--clean')
        if self.no_rebuild              : cmd.append('--no-rebuild')
        if self.no_configure            : cmd.append('--no-configure')

        if self.instance    is not None: cmd += ['-I', str(self.instance)]
        if self.count       is not None: cmd += ['-n', str(self.count)]
        if self.location_str:            cmd += ['-l', self.location_str]
        if self.speedup    is not None:  cmd += ['--speedup', str(self.speedup)]
        if self.model                   : cmd += ['--model', self.model]
        if self.wipe                    : cmd.append('-w')
        if self.use_dir                 : cmd += ['--use-dir', self.use_dir]
        if self.delay_start is not None: cmd += ['--delay-start', str(self.delay_start)]
        if self.no_mavproxy             : cmd.append('--no-mavproxy')
        if self.udp                     : cmd.append('--udp')
        if self.map                     : cmd.append('--map')
        if self.console                 : cmd.append('--console')

        if self.ideal_sensors:
            cmd.append(f'--add-param-file=/home/pid_rl/rl_training/configs/ideal_sensors.param')


        if self.master_port is not None and self.mavsdk_port is not None:
            cmd.append(f'--mavproxy-args=--out udp:127.0.0.1:{self.master_port} --out udp:127.0.0.1:{self.mavsdk_port}')
            # cmd.append(f'--mavproxy-args=--out udp:127.0.0.1:{self.master_port}')
        else:
            cmd.append(f'--mavproxy-args={self.mavproxy_args}')

        return cmd

    def _start_log_threads(self):
        assert self.process is not None
        def reader(pipe, level):
            try:
                while not self._shutdown_event.is_set():
                    line = pipe.readline()
                    if not line:  # EOF
                        break
                    logger.log(level, line.decode().rstrip())
            except Exception:
                pass  # Ignore errors during shutdown
            finally:
                try:
                    pipe.close()
                except:
                    pass
        
        # Create non-daemon threads and store references
        stdout_thread = threading.Thread(target=reader, args=(self.process.stdout, logging.INFO), daemon=True)
        stderr_thread = threading.Thread(target=reader, args=(self.process.stderr, logging.ERROR), daemon=True)
        
        stdout_thread.start()
        stderr_thread.start()
        
        self.log_threads = [stdout_thread, stderr_thread]

    def _wait_for_startup(self):
        """
        Wait for SITL to initialize by checking both process health and port availability.
        """
        logger.debug("Waiting for SITL to initialize...")
        logger.debug("   ")
        
        start = time.time()

        while time.time() - start < self.min_startup_delay:
            if self.process.poll() is not None:
                out, err = self.process.communicate(timeout=1)
                raise RuntimeError(f"SITL crashed during startup:\n{err.decode()}\n{out.decode()}")
            time.sleep(0.5)
        logger.debug(f"Minimum startup delay of {self.min_startup_delay}s completed")
        
        
        # Wait for the ports to become available

        # Wait for the ports to become available
        if not self._wait_for_ports():
            if self.process.poll() is not None:
                out, err = self.process.communicate(timeout=1)
                raise RuntimeError(f"SITL crashed while waiting for ports:\n{err.decode()}\n{out.decode()}")
            else:
                raise TimeoutError(f"Ports are not available after {self.port_check_timeout}s, but SITL process is still running")
        
    def _wait_for_ports(self) -> bool:

        logger.debug("   ")
        logger.debug(f"Waiting for Master port {self.master_port} to become available...")
        ports_available = False
        start_time = time.time()
        while time.time() - start_time < self.port_check_timeout:
            if self._check_port_available(port=self.master_port):
                logger.debug(f"Master port {self.master_port} is now available")
                ports_available = True
                break
            time.sleep(0.5)
        if not ports_available:
            logger.error(f"Master port {self.master_port} did not become available within {self.port_check_timeout}s")
            return False

        logger.debug(f"Waiting for MAVSDK port {self.mavsdk_port} to become available...")
        ports_available = False
        start_time = time.time()
        while time.time() - start_time < self.port_check_timeout:
            if self._check_port_available(port=self.mavsdk_port):
                logger.debug(f"MAVSDK port {self.mavsdk_port} is now available")
                ports_available = True
                break
            time.sleep(0.5)
        if not ports_available:
            logger.error(f"MAVSDK port {self.mavsdk_port} did not become available within {self.port_check_timeout}s")
            return False
        return True
    
    def _check_port_available(self, host: str = '127.0.0.1',
                            port: Optional[int] = None,
                            timeout: float = 1.0) -> bool:
        """
        Check whether SITL has already bound the given UDP port.
        Returns True if *you* can bind (i.e. SITL is *not* listening),
        or False if bind() fails (i.e. SITL *is* already listening there).
        """
        if port is None:
            raise ValueError("Port is required")

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(timeout)
        try:
            sock.bind((host, port))
            return True   # bind succeeded → port *not* in use by SITL
        except OSError:
            logger.error(f"Port {host}:{port} is already in use !!!")
            return False  # bind failed → port *in* use by SITL
        finally:
            sock.close()

    def _track_child_processes(self):
        """
        Any external processes that are started by the SITL process will be tracked here
        """
        try:
            parent = psutil.Process(self.process.pid)
            self.child_processes = [c.pid for c in parent.children(recursive=True)]
            logger.debug(f"Tracked {len(self.child_processes)} child processes")
        except Exception as e:
            logger.warning(f"Could not track child processes: {e}")

    # utils for connections
    def _get_mavlink_connection(self) -> mavutil.mavlink_connection:
        """
        Get or create a cached MAVLink connection.
        
        Returns:
            mavutil.mavlink_connection: Active connection
            
        Raises:
            RuntimeError: If connection cannot be established
        """
        if self._mavlink_master is None or not hasattr(self._mavlink_master, 'target_system'):
            addr = f'udp:127.0.0.1:{self.master_port}'
            
            try:
                print("new connection")
                self._mavlink_master = mavutil.mavlink_connection(addr)
                hb = self._mavlink_master.wait_heartbeat()
                self._mavlink_master.target_system = hb.get_srcSystem()

                self._mavlink_master.target_component = hb.get_srcComponent() or mavutil.mavlink.MAV_COMP_ID_AUTOPILOT1
                logger.debug(f"HB from sys:{self._mavlink_master.target_system} comp:{self._mavlink_master.target_component}")

                # channels
                PID_TUNING = 194
                NAV_CONTROLLER_OUTPUT = 62

                RATE_HZ = 100    # TODO 
                self.set_message_interval(self._mavlink_master, PID_TUNING, RATE_HZ)
                self.set_message_interval(self._mavlink_master, NAV_CONTROLLER_OUTPUT, RATE_HZ)

                GCS_PID_MASK_VALUE = 0xFFFF
                self.set_param_and_confirm(self._mavlink_master, "GCS_PID_MASK", GCS_PID_MASK_VALUE)

                print(f"Established MAVLink connection to {addr}")
                logger.debug(f"Established MAVLink connection to {addr}")
            except Exception as e:
                self._mavlink_master = None
                raise RuntimeError(f"Failed to establish MAVLink connection to {addr}: {e}")
                
        return self._mavlink_master

    def _close_mavlink_connection(self):
        """Close and cleanup cached MAVLink connection."""
        if self._mavlink_master is not None:
            try:
                self._mavlink_master.close()
            except:
                pass
            self._mavlink_master = None

    async def _get_mavsdk_connection(self) -> System:
        """
        Get or create a cached MAVSDK connection using singleton pattern.
        
        Returns:
            System: Active MAVSDK connection
            
        Raises:
            RuntimeError: If connection cannot be established
        """
        # If we already have a connection, return it
        if self._mavsdk_system is not None:
            logger.debug("Returning existing MAVSDK connection")
            return self._mavsdk_system

        # Create new connection instance
        logger.debug("Creating new MAVSDK System instance")
        self._mavsdk_system = System()
        
        # Build connection address
        port = self.mavsdk_port
        address = f"udpin://127.0.0.1:{port}"
        logger.debug(f"Connecting MAVSDK to {address}")

        try:
            # Establish connection
            await self._mavsdk_system.connect(system_address=address)
            logger.debug(f"Connection initiated to {address}")

            # Wait for connection to be established
            connected = False
            start_time = time.time()
            
            while time.time() - start_time < self.timeout:
                try:
                    # Simple connection state check
                    state = await asyncio.wait_for(
                        self._mavsdk_system.core.connection_state().__anext__(),
                        timeout=1.0
                    )
                    
                    if state.is_connected:
                        connected = True
                        logger.debug(f"Successfully connected MAVSDK to {address}")
                        break
                        
                except (asyncio.TimeoutError, StopAsyncIteration):
                    # Continue waiting
                    await asyncio.sleep(0.5)
                    continue
                except Exception as e:
                    logger.warning(f"Error checking connection state: {e}")
                    await asyncio.sleep(0.5)
                    continue
            
            if not connected:
                # Clean up on failure
                self._mavsdk_system = None
                raise RuntimeError(f"Failed to establish connection to {address} within {self.timeout} seconds")

            logger.debug(f"Established MAVSDK connection to {address}")
            return self._mavsdk_system

        except Exception as e:
            # Clean up on failure so next call starts fresh
            logger.error(f"Could not connect MAVSDK to {address}: {e}")
            self._mavsdk_system = None
            raise RuntimeError(f"Failed to establish MAVSDK connection to {address}: {e}")

    def _close_mavsdk_connection(self):
        """Close and cleanup cached MAVSDK connection."""
        if self._mavsdk_system is not None:
            # Note: MAVSDK doesn't have an explicit disconnect method,
            # the connection will be cleaned up when the object is destroyed
            self._mavsdk_system = None

    # Closing and Cleanup
    def close(self):
        self.stop_sitl()
        
        # Shutdown thread executor
        if hasattr(self, '_thread_executor'):
            self._thread_executor.shutdown(wait=False)

    def _cleanup_on_exit(self):
        if self.is_running():
            logger.debug("Cleanup on exit: stopping SITL")
            self.stop_sitl()
        
        # Shutdown thread executor
        if hasattr(self, '_thread_executor'):
            self._thread_executor.shutdown(wait=False)

    def __enter__(self):
        self.start_sitl()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        try:
            self.stop_sitl()
            if hasattr(self, '_thread_executor'):
                self._thread_executor.shutdown(wait=False)
        except:
            pass