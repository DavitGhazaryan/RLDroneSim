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


logger = logging.getLogger("SITL")  

class ArduPilotSITL:

    def __init__(self, config: Dict[str, Any], instance, verbose=True):
        if verbose:
            logging.basicConfig(level=logging.INFO)
        else:
            logging.basicConfig(level=logging.ERROR)
        
        self.config = config
        self.instance = instance

        self.process= None
        # self.child_processes: List[int] = []
        # self.log_threads: List[threading.Thread] = []  # Store references to logging threads
        # self._shutdown_event = threading.Event()  # Signal for thread shutdown
        # self._thread_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="SITL")

        # cached connections
        self._mavlink_master = None  # Cached low level connection 

        # required
        self.ardupilot_path = Path(config['ardupilot_path'])
        self.vehicle         = 'ArduCopter'
        self.frame           = config.get('frame')
        self.ideal_sensors   = config.get('ideal_sensors')

        # optional
        self.name             = config.get('name')
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
        self.map               = config.get('map')
        self.console           = config.get('console')
        self.timeout           = config.get('timeout')
        self.min_startup_delay = config.get('min_startup_delay')
        self.master_port       = config.get('master_port')  # Configurable MAVLink port
        # self.mavsdk_port       = config.get('mavsdk_port')  # Configurable MAVSDK port
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
        #logger.debug(f"Launching SITL")
        #logger.debug(f"{cmd}")

        self.process = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            preexec_fn=os.setsid, cwd=str(self.ardupilot_path),
        )

        self._wait_for_startup()      # ensures that the process is running and the port(s) are available
        self._get_mavlink_connection()
        self._set_mode_sync('GUIDED')

    def set_message_interval(self, master, msg_id, hz):
        # master = self._get_mavlink_connection()
        interval_us = int(1e6 / hz)
        master.mav.command_long_send(
            master.target_system, master.target_component,
            mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
            float(msg_id), float(interval_us), 0, 0, 0, 0, 0)
        master.recv_match(type="COMMAND_ACK", blocking=False, timeout=0.5)      


    def set_param_and_confirm(self, master, name_str, value, timeout=3.0):
        name_bytes = name_str.encode("ascii", "ignore")

        is_float = isinstance(value, float)
        ptype = (mavutil.mavlink.MAV_PARAM_TYPE_REAL32
                if is_float else mavutil.mavlink.MAV_PARAM_TYPE_INT32)

        master.mav.param_set_send(master.target_system, master.target_component,
                                name_bytes, float(value), ptype)

        t0 = time.time()
        while time.time() - t0 < timeout:
            msg = master.recv_match(type="PARAM_VALUE", blocking=True, timeout=timeout)
            print(msg)
            if not msg:
                break
            pid = (msg.param_id.decode("ascii","ignore") if isinstance(msg.param_id,(bytes,bytearray))
                else str(msg.param_id)).rstrip("\x00")
            if pid == name_str:
                return True
        print("Param is NOT SET")
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
            #logger.debug(f"Setting mode {mode_name} (ID: {mode_id})")
            
            # Send the mode change
            master.mav.set_mode_send(
                master.target_system,
                mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                mode_id
            )

            # Wait for mode change confirmation with non-blocking reads
            start_time = time.time()
            while time.time() - start_time < timeout:
                # if self._shutdown_event.is_set():  # ???
                #     return False
                    
                # Non-blocking read
                msg = master.recv_match(type='HEARTBEAT', blocking=False, timeout=0.1)
                if msg and msg.custom_mode == mode_id:
                    #logger.debug(f"Mode change to {mode_name} confirmed")
                    return True
                    
                time.sleep(0.1)
            
            logger.warning(f"Mode change to {mode_name} not confirmed within {timeout}s")
            return False
            
        except Exception as e:
            logger.error(f"Failed to set mode {mode_name}: {e}")
            return False

    def get_param(self, master, param_name, timeout=3.0, resend_every=0.5):

        name16 = param_name[:16]  # enforce MAVLink 16-char limit
        name_bytes = name16.encode("ascii", "ignore")

        t0 = time.time()
        last = 0.0
        while time.time() - t0 < timeout:
            if time.time() - last >= resend_every:
                master.mav.param_request_read_send(master.target_system, master.target_component, name_bytes, -1)
                last = time.time()
            msg = master.recv_match(type="PARAM_VALUE", blocking=True, timeout=0.5)
            if not msg:
                continue
            pid = (msg.param_id.decode("ascii", "ignore") if isinstance(msg.param_id, (bytes, bytearray))
                else str(msg.param_id)).rstrip("\x00")
            if pid == name16:
                return msg.param_value
            
        raise TimeoutError(f"Timeout: param {param_name} not received")
   
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
        #logger.debug("Stopping SITL...")
        
        # Close MAVLink connection first
        self._close_mavlink_connection()
                
        try:
            os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            logger.warning("Graceful shutdown timed out; forcing kill")
            os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
            self.process.wait(timeout=5)
        except ProcessLookupError:
            pass
        # for pid in list(self.child_processes):
        #     try:
        #         psutil.Process(pid).terminate()
        #     except:
        #         pass
        
        # Clean up logging threads gracefully
        # if hasattr(self, '_shutdown_event'):
        #     self._shutdown_event.set()  # Signal threads to stop
        
        if hasattr(self, 'log_threads') and self.log_threads:
            for thread in self.log_threads:
                if thread.is_alive():
                    try:
                        thread.join(timeout=2.0)  # Wait up to 2 seconds for thread to finish
                        if thread.is_alive():
                            logger.warning(f"Log thread {thread.name} did not terminate gracefully")
                    except Exception as e:
                        #logger.debug(f"Error joining log thread: {e}")
                        pass
            self.log_threads.clear()
        
        self.process = None
        # self.child_processes.clear()
        #logger.debug("SITL stopped.")

    # utils for the SITL
    def _validate_paths(self):
        if not self.ardupilot_path.exists():
            raise FileNotFoundError(f"ArduPilot path not found: {self.ardupilot_path}")
        if not self.sim_vehicle_script.exists():
            raise FileNotFoundError(f"sim_vehicle.py not found: {self.sim_vehicle_script}")
        #logger.debug(f"Using sim_vehicle.py: {self.sim_vehicle_script}")

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
        #logger.debug(f"Loaded {len(defaults)} default params from {path}")
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

        if self.instance    is not None: cmd += ['-I', str(self.instance-1)]
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

        if self.instance == 2:
            self.master_port += 10

        return cmd

    def _wait_for_startup(self):
        """
        Wait for SITL to initialize by checking both process health and port availability.
        """
        
        start = time.time()

        while time.time() - start < self.min_startup_delay:
            if self.process.poll() is not None:
                out, err = self.process.communicate(timeout=1)
                raise RuntimeError(f"SITL crashed during startup:\n{err.decode()}\n{out.decode()}")
            time.sleep(0.5)

        if not self._wait_for_ports():
            if self.process.poll() is not None:
                out, err = self.process.communicate(timeout=1)
                raise RuntimeError(f"SITL crashed while waiting for ports:\n{err.decode()}\n{out.decode()}")
            else:
                raise TimeoutError(f"Ports are not available after {self.port_check_timeout}s, but SITL process is still running")
        
    def _wait_for_ports(self) -> bool:

        #logger.debug("   ")
        #logger.debug(f"Waiting for Master port {self.master_port} to become available...")
        ports_available = False
        start_time = time.time()
        while time.time() - start_time < self.port_check_timeout:
            if self._check_port_available(port=self.master_port):
                #logger.debug(f"Master port {self.master_port} is now available")
                ports_available = True
                break
            time.sleep(0.5)
        if not ports_available:
            logger.error(f"Master port {self.master_port} did not become available within {self.port_check_timeout}s")
            return False

        # #logger.debug(f"Waiting for MAVSDK port {self.mavsdk_port} to become available...")
        # ports_available = False
        # start_time = time.time()
        # while time.time() - start_time < self.port_check_timeout:
        #     if self._check_port_available(port=self.mavsdk_port):
        #         #logger.debug(f"MAVSDK port {self.mavsdk_port} is now available")
        #         ports_available = True
        #         break
        #     time.sleep(0.5)
        # if not ports_available:
        #     logger.error(f"MAVSDK port {self.mavsdk_port} did not become available within {self.port_check_timeout}s")
        #     return False
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

    # def _track_child_processes(self):
    #     """
    #     Any external processes that are started by the SITL process will be tracked here
    #     """
    #     try:
    #         parent = psutil.Process(self.process.pid)
    #         self.child_processes = [c.pid for c in parent.children(recursive=True)]
    #         #logger.debug(f"Tracked {len(self.child_processes)} child processes")
    #     except Exception as e:
    #         logger.warning(f"Could not track child processes: {e}")

    # utils for connections
    def _get_mavlink_connection(self):
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
                self._mavlink_master = mavutil.mavlink_connection(addr)
                hb = self._mavlink_master.wait_heartbeat()
                self._mavlink_master.target_system = hb.get_srcSystem()

                self._mavlink_master.target_component = hb.get_srcComponent() or mavutil.mavlink.MAV_COMP_ID_AUTOPILOT1
                #logger.debug(f"HB from sys:{self._mavlink_master.target_system} comp:{self._mavlink_master.target_component}")

                # channels
                PID_TUNING = 194
                NAV_CONTROLLER_OUTPUT = 62

                RATE_HZ = 20     
                self.set_message_interval(self._mavlink_master, PID_TUNING, RATE_HZ)
                self.set_message_interval(self._mavlink_master, NAV_CONTROLLER_OUTPUT, RATE_HZ)

                GCS_PID_MASK_VALUE = 0xFFFF
                self.set_param_and_confirm(self._mavlink_master, "GCS_PID_MASK", GCS_PID_MASK_VALUE)

                print(f"Established MAVLink connection to {addr}")
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

    # Closing and Cleanup
    def close(self):
        self.stop_sitl()
        
        # Shutdown thread executor
        if hasattr(self, '_thread_executor'):
            self._thread_executor.shutdown(wait=False)

    def _cleanup_on_exit(self):
        if self.is_running():
            #logger.debug("Cleanup on exit: stopping SITL")
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