#!/usr/bin/env python3

import subprocess
import time
import os
import signal
import logging
import atexit
import socket
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
        atexit.register(self.close)
        self._validate_paths()

    def start_sitl(self):
        if self.is_running():
            raise RuntimeError("SITL already running")
        cmd = self._build_command()

        self.process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=False,
            cwd=str(self.ardupilot_path),
        )
        self._wait_for_startup()      # ensures that the process is running and the port(s) are available
        self._get_mavlink_connection()
        self.set_mode('GUIDED')

    def set_message_interval(self, master, msg_id, hz):
        # master = self._get_mavlink_connection()
        interval_us = 0 if hz<=0 else int(1e6/hz)

        master.mav.command_long_send(
            master.target_system, master.target_component,
            mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
            float(msg_id), float(interval_us), 0, 0, 0, 0, 0)
        master.recv_match(type="COMMAND_ACK", blocking=True, timeout=0.5)      

    def set_param_and_confirm(self, name_str, value, timeout=3.0):
        """
        Sets the parameter and waits for the acknowledgement message.
        """
        name_bytes = name_str.encode("ascii", "ignore")
        is_float = isinstance(value, float)
        ptype = (mavutil.mavlink.MAV_PARAM_TYPE_REAL32
                if is_float else mavutil.mavlink.MAV_PARAM_TYPE_INT32)

        self._mavlink_master.mav.param_set_send(self._mavlink_master.target_system, self._mavlink_master.target_component,
                                name_bytes, float(value), ptype)

        t0 = time.time()
        while time.time() - t0 < timeout:
            msg = self._mavlink_master.recv_match(type="PARAM_VALUE", blocking=True, timeout=timeout)
            if not msg:
                print("msg missed in set")
                continue
            pid = (msg.param_id.decode("ascii","ignore") if isinstance(msg.param_id,(bytes,bytearray))
                else str(msg.param_id)).rstrip("\x00")
            if pid == name_str:
                if abs(float(msg.param_value) - float(value)) <= 0.01:
                    return True  # confirmed exact (within tol)
                else:
                    print(msg)
                return True
            else: 
                print(f"another one was requested here {pid} , {name_str}")
        print("Param is NOT SET")
        return False

    def get_param(self, param_name, timeout=3.0, resend_every=0.5):
        name16 = param_name[:16]  # enforce MAVLink 16-char limit
        name_bytes = name16.encode("ascii", "ignore")
        t0 = time.time()
        last = 0.0
        # print(f"REquested {name16}")
        while time.time() - t0 < timeout:
            if time.time() - last >= resend_every:
                self._mavlink_master.mav.param_request_read_send(self._mavlink_master.target_system, self._mavlink_master.target_component, name_bytes, -1)
                last = time.time()
            msg = self._mavlink_master.recv_match(type="PARAM_VALUE", blocking=True, timeout=0.5)
            if not msg:
                print("msg missed in get")
                continue
            pid = (msg.param_id.decode("ascii", "ignore") if isinstance(msg.param_id, (bytes, bytearray))
                else str(msg.param_id)).rstrip("\x00")
            if pid == name16:
                return msg.param_value
            else:
                print(f"another one was requested here {pid}, {name16}")
                print(msg)
            
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
            return None

    def set_mode(self, mode_name: str, timeout: float = 10.0) -> bool:
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

    def send_reset(self, n, e, agl, seq=None, retries=3, ack_timeout=1.5):
        CMD = 31010
        if seq is None:
            seq = int(time.time() * 1000) & 0x7FFFFFFF  # monotonic-ish

        for _ in range(retries):
            self._mavlink_master.mav.command_long_send(
                self._mavlink_master.target_system, self._mavlink_master.target_component,
                CMD, 0,          # confirmation=0
                float(n), float(e), float(agl),
                float(seq), 0, 0, 0
            )
            msg = self._mavlink_master.recv_match(type="COMMAND_ACK", blocking=True, timeout=0.5)
            if msg and int(msg.command) == CMD:
                return int(msg.result)  # 0 = ACCEPTED
        return False

    def arm_drone(self, timeout=10):
        self._mavlink_master.wait_heartbeat()
        t0 = time.time()
        while time.time() - t0 < timeout:
            hb = self._mavlink_master.recv_match(type="HEARTBEAT", blocking=True, timeout=1.0)
            if not hb:
                continue
            if hb.system_status == mavutil.mavlink.MAV_STATE_STANDBY:
                break
            time.sleep(0.1)
        if time.time() - t0 >= timeout:
            raise TimeoutError("Failed to arm the drone")
        
        self._mavlink_master.mav.command_long_send(
            self._mavlink_master.target_system, self._mavlink_master.target_component,
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 1, 0, 0, 0, 0, 0, 0
        )

        self._mavlink_master.recv_match(type='COMMAND_ACK', blocking=True, timeout=10)

    def takeoff_drone(self, altitude):
        self._mavlink_master.wait_heartbeat()
        self._mavlink_master.mav.command_long_send(
            self._mavlink_master.target_system,
            self._mavlink_master.target_component,
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
            0,          # confirmation
            0, 0, 0, 0, # params 1–4 (unused here)
            0, 0,       # lat, lon (0 = current location)
            altitude    # param7 = target altitude (meters, AMSL)
        )
        ack = self._mavlink_master.recv_match(type='COMMAND_ACK', blocking=True, timeout=10)
        if ack and ack.result == mavutil.mavlink.MAV_RESULT_ACCEPTED:
            pass
        else:
            raise Exception(f"Failed to take off: {ack}")

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

        print(cmd)
        return cmd

    def _wait_for_startup(self):
        """
        Wait for SITL to initialize by checking both process health and port availability.
        """
        
        # check process itself
        start = time.time()
        while time.time() - start < self.min_startup_delay:
            if self.process.poll() is not None:
                out, err = self.process.communicate(timeout=1)
                raise RuntimeError(f"SITL crashed during startup:\n{err.decode()}\n{out.decode()}")
            time.sleep(0.5)
        
        # Check for ports
        start_time = time.time()
        while time.time() - start_time < self.port_check_timeout:
            if self._check_port_available(port=self.master_port):
                #logger.debug(f"Master port {self.master_port} is now available")
                port_available = True
                break
            
        if not port_available:
            if self.process.poll() is not None:
                out, err = self.process.communicate(timeout=1)
                raise RuntimeError(f"SITL crashed while waiting for ports:\n{err.decode()}\n{out.decode()}")
            else:
                raise TimeoutError(f"Master port {self.master_port} did not become available within {self.port_check_timeout}s")
        
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

    def _get_mavlink_connection(self):
        """
        Get or create a cached MAVLink connection.
        Also sets the appropriate parameters on the drone.
        
        Returns:
            mavutil.mavlink_connection: Active connection
            
        Raises:
            RuntimeError: If connection cannot be established
        """
        if self._mavlink_master is None or not hasattr(self._mavlink_master, 'target_system'):
            addr = f'udp:127.0.0.1:{self.master_port}'
            # addr = "udp:127.0.0.1:5760"
            try:
                self._mavlink_master = mavutil.mavlink_connection(addr, dialect="ardupilotmega")
                hb = self._mavlink_master.wait_heartbeat()
                self._mavlink_master.target_system = hb.get_srcSystem()

                self._mavlink_master.target_component = hb.get_srcComponent() or mavutil.mavlink.MAV_COMP_ID_AUTOPILOT1
                #logger.debug(f"HB from sys:{self._mavlink_master.target_system} comp:{self._mavlink_master.target_component}")

                # channels
                PID_TUNING = 194
                NAV_CONTROLLER_OUTPUT = 62
                LOCAL_POSITION_NED = 32
                ATTITUDE = 30

                RATE_HZ = 20     
                self.set_message_interval(self._mavlink_master, PID_TUNING, RATE_HZ)
                self.set_message_interval(self._mavlink_master, NAV_CONTROLLER_OUTPUT, RATE_HZ)
                self.set_message_interval(self._mavlink_master, LOCAL_POSITION_NED, RATE_HZ)
                self.set_message_interval(self._mavlink_master, ATTITUDE, RATE_HZ)

                GCS_PID_MASK_VALUE = 0xFFFF
                self.set_param_and_confirm("GCS_PID_MASK", GCS_PID_MASK_VALUE)

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

    def close(self):
        if not self.is_running():
            return

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
                
        self.process = None