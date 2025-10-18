#!/usr/bin/env python3

import subprocess
import time
import os
import signal
import logging
import atexit
import socket
from typing import Dict, Any, List
from pathlib import Path
from .gazebo_interface import GazeboInterface
from .drone import Drone

from pymavlink import mavutil


logger = logging.getLogger("SITL")  

class ArduPilotSITL(Drone):

    def __init__(self, config: Dict[str, Any], gazebo_config,  instance, verbose=True):
        super().__init__(config, verbose)

        self.ardupilot_path = Path(config['ardupilot_path'])
        self.vehicle         = 'ArduCopter'
        self.frame           = config.get('frame')
        self.ideal_sensors   = config.get('ideal_sensors')

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
        self.ip                = config.get('ip')   # !!!        
        self.master_port       = config.get('master_port') 
        self.port_check_timeout = config.get('port_check_timeout')
        self.console           = config.get('console')
        self.timeout           = config.get('timeout')
        self.min_startup_delay = config.get('min_startup_delay')

        self.instance = instance
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

        self.process= None
        self.gazebo = GazeboInterface(gazebo_config, instance, verbose)
        self._default_params = self._load_default_params()
        self._validate_paths()

    def start(self):
        self.gazebo.start_simulation()   # waiting is done internally.
        self.gazebo.resume_simulation()

        self.start_sitl()
        time.sleep(25/self.speedup)
        self.arm_drone()

    def reset(self, pose, attitude):
        self.gazebo.pause_simulation()

        self.gazebo.transport_position(self.name, pose, attitude)

        self.gazebo.resume_simulation()
    
        self.send_reset(pose[1], pose[0], pose[2])

    def start_sitl(self):
        if self.is_running():
            raise RuntimeError("SITL already running")
        cmd = self._build_command()

        self.process = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=False,
            cwd=str(self.ardupilot_path),
        )
        self._wait_for_startup()      # ensures that the process is running and the port(s) are available
        self._get_mavlink_connection()
        self.set_mode('GUIDED')

    def wait(self, duration):
        """
        Sleep for the given duration (in seconds) using Gazebo simulation time.
        """
        start_time = self.gazebo.get_sim_time()
        while True:
            time.sleep(0.001)
            current_time = self.gazebo.get_sim_time()
            if current_time - start_time >= duration:
                break


    def is_running(self) -> bool:
        return bool(self.process and self.process.poll() is None)

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