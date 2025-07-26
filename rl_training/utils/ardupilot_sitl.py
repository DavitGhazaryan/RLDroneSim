#!/usr/bin/env python3
"""
ArduPilot SITL interface for RL training (MAVSDK reset).
"""

import subprocess
import time
import os
import signal
import psutil
import logging
import atexit
import threading
import asyncio
import math
import socket
import concurrent.futures
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

from pymavlink import mavutil

from mavsdk import System
from mavsdk.offboard import PositionNedYaw
from mavsdk.param import Param

logger = logging.getLogger("SITL")
logging.basicConfig(level=logging.INFO)

class ArduPilotSITL:
    def __init__(self, config: Dict[str, Any]):
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
        self.vehicle         = config.get('vehicle', 'ArduCopter')
        self.frame           = config.get('frame',   'quad')

        # optional
        self.instance         = config.get('instance', None)
        self.count            = config.get('count')
        self.location_str     = config.get('location')
        self.speedup          = config.get('speedup')
        self.wipe              = config.get('wipe_eeprom', False)
        self.use_dir           = config.get('use_dir')
        self.delay_start       = config.get('delay_start')
        self.model             = config.get('model')
        self.clean             = config.get('clean', True)
        self.no_rebuild        = config.get('no_rebuild', True)
        self.no_configure      = config.get('no_configure', True)
        self.no_mavproxy       = config.get('no_mavproxy', False)
        self.udp               = config.get('udp', True)
        self.udp_out           = config.get('udp_out')
        self.map               = config.get('map', False)
        self.console           = config.get('console', False)
        self.mavproxy_args     = config.get('mavproxy_args')
        self.timeout           = config.get('timeout', 30.0)
        self.min_startup_delay = config.get('min_startup_delay', 5.0)
        self.master_port       = config.get('master_port', 14550)  # Configurable MAVLink port
        self.mavsdk_port       = config.get('mavsdk_port', 14551)  # Configurable MAVSDK port
        self.port_check_timeout = config.get('port_check_timeout', 30.0)  # Timeout for port availability

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

    def start_sitl(self):
        if self.is_running():
            raise RuntimeError("SITL already running")
        cmd = self._build_command()
        logger.info(f"Launching SITL: {' '.join(cmd)}")

        self.process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            preexec_fn=os.setsid, cwd=str(self.ardupilot_path)
        )
        self._shutdown_event.clear()  # set to False, enables logging threads to run
        self._start_log_threads()
        self._wait_for_startup()      # ensures that the process is running and the port(s) are available
        self._track_child_processes()


        ### establish connections
        logger.info("Establishing connections...")
        self._get_mavlink_connection()
        logger.info("MAVLink connection established")

        try:
            self.set_mode('GUIDED')
        except Exception as e:
            logger.warning(f"Error setting GUIDED mode after startup: {e}")
        logger.info("SITL started successfully.")

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
        if self.speedup    is not None:  cmd += ['-S', str(self.speedup)]
        if self.model                   : cmd += ['--model', self.model]
        if self.wipe                    : cmd.append('-w')
        if self.use_dir                 : cmd += ['--use-dir', self.use_dir]
        if self.delay_start is not None: cmd += ['--delay-start', str(self.delay_start)]
        if self.no_mavproxy             : cmd.append('--no-mavproxy')
        if self.udp                     : cmd.append('--udp')
        if self.map                     : cmd.append('--map')
        if self.console                 : cmd.append('--console')

        if self.master_port is not None and self.mavsdk_port is not None:
            cmd.append(f'--mavproxy-args=--out udp:0.0.0.0:{self.master_port} --out udp:0.0.0.0:{self.mavsdk_port}')
        else:
            cmd.append(f'--mavproxy-args={self.mavproxy_args}')
        logger.info(f"Command: {' '.join(cmd)}")
        return cmd

    def _start_log_threads(self):
        assert self.process is not None
        def reader(pipe, level):
            '''
            Reads the stdout and stderr of the SITL process and logs them to the logger.
            This is a non-daemon thread, so it will block the main thread from exiting.
            Will end when shutdown_event is set to True.
            '''
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
        stdout_thread = threading.Thread(target=reader, args=(self.process.stdout, logging.INFO), daemon=False)
        stderr_thread = threading.Thread(target=reader, args=(self.process.stderr, logging.ERROR), daemon=False)
        
        stdout_thread.start()
        stderr_thread.start()
        
        self.log_threads = [stdout_thread, stderr_thread]

    def _wait_for_startup(self):
        """
        Wait for SITL to initialize by checking both process health and port availability.
        """
        logger.info("Waiting for SITL to initialize...")
        logger.info("   ")
        
        start = time.time()
        # First, wait minimum delay
        while time.time() - start < self.min_startup_delay:
            if self.process.poll() is not None:
                out, err = self.process.communicate(timeout=1)
                raise RuntimeError(f"SITL crashed during startup:\n{err.decode()}\n{out.decode()}")
            time.sleep(0.5)
        logger.debug(f"Minimum startup delay of {self.min_startup_delay}s completed")
        
        # Wait for the ports to become available
        if not self._wait_for_ports():
            # Still check if process crashed
            if self.process.poll() is not None:
                out, err = self.process.communicate(timeout=1)
                raise RuntimeError(f"SITL crashed while waiting for ports:\n{err.decode()}\n{out.decode()}")
            else:
                raise TimeoutError(f"Ports are not available after {self.port_check_timeout}s, but SITL process is still running")
        
    def _wait_for_ports(self) -> bool:
        """
        Wait for MAVLink port to become available by polling.
        
        Returns:
            bool: True if port became available, False if timeout
        """
        logger.info("   ")
        logger.info(f"Waiting for Master port {self.master_port} to become available...")
        ports_available = False
        start_time = time.time()
        while time.time() - start_time < self.port_check_timeout:
            if self._check_port_available(port=self.master_port):
                logger.info(f"Master port {self.master_port} is now available")
                ports_available = True
                break
            time.sleep(0.5)
        if not ports_available:
            logger.error(f"Master port {self.master_port} did not become available within {self.port_check_timeout}s")
            return False

        logger.info(f"Waiting for MAVSDK port {self.mavsdk_port} to become available...")
        ports_available = False
        start_time = time.time()
        while time.time() - start_time < self.port_check_timeout:
            if self._check_port_available(port=self.mavsdk_port):
                logger.info(f"MAVSDK port {self.mavsdk_port} is now available")
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
            logger.info(f"Port {host}:{port} is already in use !!!")
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
                self._mavlink_master = mavutil.mavlink_connection(addr)
                self._mavlink_master.wait_heartbeat(timeout=10.0)
                logger.info(f"Established MAVLink connection to {addr}")
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
        Get or create a cached MAVSDK connection.
        
        Returns:
            System: Active MAVSDK connection
            
        Raises:
            RuntimeError: If connection cannot be established
        """
        if self._mavsdk_system is None:
            logger.debug("Creating new MAVSDK System instance")
            self._mavsdk_system = System()
        else:
        # Check if already connected - but only if system was previously created
            try:
                # Quick check with short timeout to avoid hanging
                connection_task = asyncio.create_task(
                    self._mavsdk_system.core.connection_state().__anext__()
                )
                try:
                    state = await asyncio.wait_for(connection_task, timeout=0.5)
                    if state.is_connected:
                        logger.debug("MAVSDK already connected, reusing connection")
                        return self._mavsdk_system
                except (asyncio.TimeoutError, StopAsyncIteration, Exception):
                    # Cancel the task to prevent unawaited coroutine warning
                    connection_task.cancel()
                    try:
                        await connection_task
                    except asyncio.CancelledError:
                        pass
                    # Not connected or error checking - proceed to connect
                    logger.debug("Connection check failed or not connected, proceeding to connect")
            except Exception as e:
                logger.debug(f"Connection check failed: {e}")

        # Not connected yet → build address and attempt to connect
        port = self.mavsdk_port
        address = f"udp://0.0.0.0:{port}"
        logger.debug(f"Connecting MAVSDK to {address}")

        try:
            # Kick off the connection
            await self._mavsdk_system.connect(system_address=address)
            logger.debug(f"Connection initiated to {address}")

            # Now wait for connection to be established
            connected = False
            start_time = time.time()
            
            while time.time() - start_time < self.timeout:
                state_task = None
                try:
                    # Check connection state with timeout
                    state_task = asyncio.create_task(
                        self._mavsdk_system.core.connection_state().__anext__()
                    )
                    state = await asyncio.wait_for(state_task, timeout=1.0)
                    
                    if state.is_connected:
                        connected = True
                        logger.info(f"Successfully connected MAVSDK to {address}")
                        break
                        
                except (asyncio.TimeoutError, StopAsyncIteration):
                    # Cancel the task to prevent unawaited coroutine warning
                    if state_task and not state_task.done():
                        state_task.cancel()
                        try:
                            await state_task
                        except asyncio.CancelledError:
                            pass
                    # Continue waiting
                    await asyncio.sleep(0.5)
                    continue
                except Exception as e:
                    # Cancel the task to prevent unawaited coroutine warning
                    if state_task and not state_task.done():
                        state_task.cancel()
                        try:
                            await state_task
                        except asyncio.CancelledError:
                            pass
                    logger.warning(f"Error checking connection state: {e}")
                    await asyncio.sleep(0.5)
                    continue
            
            if not connected:
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
            logger.info(f"Setting mode {mode_name} (ID: {mode_id})")
            
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
                    logger.info(f"Mode change to {mode_name} confirmed")
                    logger.info("   ")
                    return True
                    
                time.sleep(0.1)
            
            logger.warning(f"Mode change to {mode_name} not confirmed within {timeout}s")
            return False
            
        except Exception as e:
            logger.error(f"Failed to set mode {mode_name}: {e}")
            return False

    async def set_mode_async(self, mode_name: str, timeout: float = 10.0) -> bool:
        """
        Async mode setting method that doesn't block event loop.
        
        Args:
            mode_name: Name of the mode
            timeout: Timeout in seconds
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.is_running():
            logger.error("SITL not running, cannot set mode")
            return False
            
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._thread_executor, 
            self._set_mode_sync, 
            mode_name, 
            timeout
        )

    def set_mode(self, mode_name: str, timeout: float = 10.0) -> bool:
        """
        General method to set any flight mode using pymavlink.
        Uses thread executor to avoid blocking if called from async context.
        
        Args:
            mode_name: Name of the mode (e.g., 'GUIDED', 'STABILIZE', 'LOITER', 'RTL', etc.)
            timeout: Timeout in seconds to wait for mode change confirmation
            
        Returns:
            bool: True if mode change was successful, False otherwise
        """
        if not self.is_running():
            logger.error("SITL not running, cannot set mode")
            return False
        logger.info(f"Starting mode change to {mode_name}")
        # Check if we're in an async context
        try:
            loop = asyncio.get_running_loop()
            # We're in an async context - use executor to avoid blocking
            future = asyncio.run_coroutine_threadsafe(
                self.set_mode_async(mode_name, timeout), 
                loop
            )
            return future.result(timeout=timeout + 5.0)  # Extra buffer for executor overhead
        except RuntimeError:
            # No running event loop - safe to call sync version
            return self._set_mode_sync(mode_name, timeout)

    async def _wait_for_armable_async(self, timeout: float = 30.0, poll_interval: float = 0.5) -> bool:
        """
        Async helper: wait until the vehicle reports is_armable via MAVSDK.telemetry.health()
        """
        logger.debug(f"Checking if vehicle is armable (timeout={timeout}s)")
        
        try:
            # get or connect MAVSDK System
            drone = await self._get_mavsdk_connection()
            logger.debug("Got MAVSDK connection for armable check")

            start = time.time()
            async for health in drone.telemetry.health():
                logger.debug(f"Health check: armable={health.is_armable}")
                if health.is_armable:
                    logger.debug("Vehicle is armable!")
                    return True
                if time.time() - start > timeout:
                    logger.warning(f"Armable check timed out after {timeout}s")
                    return False
                await asyncio.sleep(poll_interval)

            logger.warning("Health telemetry stream ended unexpectedly")
            return False  # if telemetry.health() stream ends for some reason
            
        except Exception as e:
            logger.error(f"Exception in _wait_for_armable_async: {e}")
            raise

    def check_is_armable(self, timeout: float = 30.0, poll_interval: float = 0.5) -> bool:
        """
        Synchronous wrapper around _wait_for_armable_async.
        Returns True if armable within `timeout` seconds, False otherwise.
        """
        if not self.is_running():
            logger.error("SITL not running, cannot check armable state")
            return False
            
        # Check if we're in an async context
        try:
            loop = asyncio.get_running_loop()
            # We're in an async context - use executor to avoid blocking
            future = asyncio.run_coroutine_threadsafe(
                self._wait_for_armable_async(timeout, poll_interval), 
                loop
            )
            return future.result(timeout=timeout + 5.0)  # Extra buffer for executor overhead
        except RuntimeError:
            # No running event loop - safe to call asyncio.run
            try:
                return asyncio.run(self._wait_for_armable_async(timeout, poll_interval))
            except Exception as e:
                logger.error(f"check_is_armable failed: {e}")
                return False

    def stop_sitl(self):
        if not self.is_running():
            return
        logger.info("Stopping SITL...")
        
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
        logger.info("SITL stopped.")

    def restart_sitl(self, wipe_params: bool = True):
        """
        Full restart. wipe_params=True → uses -w to restore defaults.
        """
        logger.info(f"Restarting SITL (wipe_params={wipe_params})")
        self.stop_sitl()
        time.sleep(2)
        self.wipe = wipe_params
        self.start_sitl()

    async def reset_async(self, keep_params: bool = True):
        """
        Async in-place reset via MAVSDK:
          - teleports vehicle home
          - clears mission
          - optionally restores defaults from .parm if keep_params=False
          - re-arms and sets guided mode
        """
        if not self.is_running():
            raise RuntimeError("SITL not running")
        print(f"Resetting SITL with keep_params={keep_params} from reset_async()")
        
        try:
            # Get or establish persistent MAVSDK connection
            drone = await self._get_mavsdk_connection()
            print(f"Using persistent MAVSDK connection")
            
            # Small delay to ensure connection is stable
            await asyncio.sleep(1)
            
            # # disarm vehicle (with error handling)
            # try:
            #     await drone.action.disarm()
            #     print("Vehicle disarmed successfully")
            # except Exception as e:
            #     print(f"Warning: Disarm failed: {e}")
            #     # Not critical for reset operation, continue anyway
            
            # teleport vehicle to home position
            if self.home:
                try:
                    await drone.offboard.start()
                    lat, lon, alt, yaw = self.home
                    # Teleport to home position (0,0,0 in NED relative to home with original yaw)
                    await drone.offboard.set_position_ned(PositionNedYaw(0.0, 0.0, 0.0, math.radians(yaw)))
                    await asyncio.sleep(0.5)
                    await drone.offboard.stop()
                    print("Vehicle teleported to home position")
                except Exception as e:
                    print(f"Warning: Teleport failed: {e}")
            
            # clear mission (less critical, so continue on failure)
            try:
                await drone.mission.clear_mission()
                print("Mission cleared successfully")
            except Exception as e:
                print(f"Warning: Clear mission failed: {e}")
            
            # restore defaults if requested (with batching for reliability)
            if not keep_params:
                try:
                    param_items = list(self._default_params.items())
                    batch_size = 10  # Process parameters in batches
                    for i in range(0, len(param_items), batch_size):
                        batch = param_items[i:i + batch_size]
                        for name, val in batch:
                            await drone.param.set_param_float(name, val)
                        # Small delay between batches to let autopilot process
                        if i + batch_size < len(param_items):
                            await asyncio.sleep(0.1)
                    print(f"Restored {len(param_items)} default parameters")
                except Exception as e:
                    print(f"Warning: Parameter restore failed: {e}")
            
            # Wait a bit for parameters to settle
            await asyncio.sleep(1.0)
            
            print("Reset operations completed successfully")
            
        except Exception as e:
            print(f"Reset failed with error: {e}")
            raise
        finally:
            # Ensure we always give time for operations to complete
            try:
                await asyncio.sleep(0.5)  # Give time for operations to complete
            except:
                pass
        
        # Set GUIDED mode after reset using async mode setting
        print("Setting GUIDED mode after reset...")
        await asyncio.sleep(1.0)  # Give time for operations to settle
        
        try:
            if await self.set_mode_async('GUIDED'):
                print("Successfully set GUIDED mode after reset")
            else:
                print("Warning: Failed to set GUIDED mode after reset")
        except Exception as e:
            print(f"Warning: Error setting GUIDED mode after reset: {e}")

    def reset(self, keep_params: bool = True):
        """
        Synchronous wrapper for reset_async.
        In-place reset via MAVSDK:
          - teleports vehicle home
          - clears mission  
          - optionally restores defaults from .parm if keep_params=False
          - sets guided mode after reset
        """
        return asyncio.run(self.reset_async(keep_params))

    def is_running(self) -> bool:
        return bool(self.process and self.process.poll() is None)

    # Getters
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
        
        # Add current mode if available
        current_mode = self.get_mode()
        if current_mode:
            info['current_mode'] = current_mode
            
        return info


    def _cleanup_on_exit(self):
        if self.is_running():
            logger.info("Cleanup on exit: stopping SITL")
            self.stop_sitl()
        
        # Shutdown thread executor
        if hasattr(self, '_thread_executor'):
            self._thread_executor.shutdown(wait=False)

    def close(self):
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