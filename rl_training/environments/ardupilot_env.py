"""
Main Ardupilot environment implementing Gymnasium API.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import logging
import sys
import asyncio

sys.path.insert(0, "/home/student/Dev/pid_rl")

from rl_training.utils.gazebo_interface import GazeboInterface
from rl_training.utils.ardupilot_sitl import ArduPilotSITL
from rl_training.utils.utils import load_config

logger = logging.getLogger("Env")
logger.setLevel(logging.INFO)

PID_KEYS = [
    # Not Complete
    "ATC_ANG_PIT_P",     # Attitude angle PID Gains
    "ATC_ANG_RLL_P",
    "ATC_ANG_YAW_P",
    "ATC_RAT_PIT_P",     # Attitude rate PID Gains
    "ATC_RAT_RLL_P",
    "ATC_RAT_YAW_P",
    "PSC_POSZ_P",       # Position Z PID Gains
    "PSC_VELZ_P",       # Velocity Z PID Gains
]

class ArdupilotEnv(gym.Env):
    """
    Initializes Gazebo and Ardupilot SITL, and provides a Gymnasium-compatible interface.
    Environment is intended to enable training if an RL agent that will find the optimal PID gains to put on an agent.  
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.__compatibility_checks()

        self.gazebo = GazeboInterface(self.config['gazebo_config'])
        self.sitl = ArduPilotSITL(self.config['ardupilot_config'])
        self.loop = asyncio.get_event_loop()


        # Episode tracking
        self.initialized = False
        self.episode_step = 0
        self.environment_config = config.get('environment_config', {})
        self.observable_variables = self.environment_config['observable_variables'].split('+')
        self.action_variables = self.environment_config['action_variables'].split('+')
        self.max_episode_steps = self.environment_config.get('max_episode_steps', 100)
        self.initial_pose = None
        self.old_observation = None
        self._async_mission_function = None

        # Initialize spaces
        self.observation_space = self._define_observation_space()
        self.action_space = self._define_action_space()
        
    def _define_observation_space(self):
        """Needs to be updated to allow different observation spaces"""
        low = np.full(len(self.observable_variables), -np.inf, dtype=np.float32)
        high = np.full(len(self.observable_variables), np.inf, dtype=np.float32)
        return spaces.Box(low=low, high=high, dtype=np.float32)
    
    def _define_action_space(self):
        """Needs to be updated to allow different action spaces"""
        low = np.array([-0.1] * len(self.action_variables), dtype=np.float32)
        high = np.array([0.1] * len(self.action_variables), dtype=np.float32)
        return spaces.Box(low=low, high=high, dtype=np.float32)

    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)

        self.episode_step = 0

        if not self.initialized:
            logger.info("🌎 Launching Gazebo simulation...")
            self.gazebo.start_simulation()
            self.gazebo._wait_for_startup()
            self.gazebo.resume_simulation()
            logger.info("✅ Gazebo initialized")
            logger.info("--------------------------------")
            logger.info("🚁 Starting ArduPilot SITL...")
            self.sitl.start_sitl()
            info = self.sitl.get_process_info()
            logger.info(f"✅ SITL running (PID {info['pid']})")
            self.initialized = True
            self.loop.run_until_complete(self._async_arm())
            ## Resetting Ends Here: Drone is Armed

            ## Setup Mission
            logger.info("Getting Initial Pose--------------------------------")
            self.initial_pose = self.loop.run_until_complete(self.sitl.get_pose_async())
            self._async_mission_function = self._setup_mission()
            logger.info("--------------------------------")

            self.loop.run_until_complete(self._async_mission_function())


        else:
            self.gazebo.transport_position(self.sitl.name, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
        
        logger.info("Before Observation--------------------------------")
        self.initial_pose = self.loop.run_until_complete(self.sitl.get_pose_async())
        logger.info(f"Initial Pose Checked: {self.initial_pose}")
        self.old_observation, info = self.loop.run_until_complete(self._async_get_observation())
        logger.info("After Observation--------------------------------")

        return self.old_observation, info  # observation, info

    async def _async_get_observation(self):
        """
        Get the observations of the environment.
        This function is called when the environment is reset.
        """
        drone = await self.sitl._get_mavsdk_connection()
        logger.info("Getting Parameters--------------------------------")
        # observed_gains = {
        #     variable: await drone.param.get_param_float(variable)
        #     for variable in self.observable_variables if variable in PID_KEYS    # all PID names are longer than 3 characters
        # }
        # async for state in drone.core.connection_state():
        #     logger.info(f"[MAVSDK] Connected: {state.is_connected}")
        # position = await anext(drone.telemetry.position())
        # logger.info(f"Position was able to be retrieved: {position}")


        # observation = np.concatenate([
        #     np.array([observed_gains[k] for k in observed_gains], dtype=np.float32),
        #     # np.array([altitude.relative_altitude_m], dtype=np.float32)
        # ])
        return [0.0, 0.0, 0.0], {}  # observation, info
        # return observation, {}  # observation, info
    
    def _setup_mission(self):
        match self.config['environment_config']['mode']:
            case 'altitude':
                return self._async_takeoff
            case 'position' | 'attitude' | 'stabilize' | 'althold':
                raise NotImplementedError("Position, attitude, stabilize, and althold modes are not implemented yet")
            case _:
                raise ValueError(f"Invalid mode: {self.config['environment_config']['mode']}")
    
    def step(self, action   ):
        self.episode_step += 1
        obs, reward, done, truncated, info = self.loop.run_until_complete(self._async_step(action))
        self.old_observation = obs
        return obs, reward, done, truncated, info

    async def _async_step(self, action):
        # pid_params = self.old_observation + action
        # await self.sitl.set_params_async(pid_params)
        pid_params = await self.sitl.get_pid_params_async()
        pose = await self.sitl.get_pose_async()
        reward = await self._compute_reward(pid_params, pose)
        # INSERT_YOUR_CODE
        
        def _check_terminated(pose):
            # Assume pose is a dict with keys: 'pitch', 'roll', 'relative_altitude', 'x', 'y', 'z'
            # 1. Crash: very big pitch or roll (e.g., > 60 deg)
            
            # pitch = abs(getattr(pose, 'pitch', pose.get('pitch', 0)))
            # roll = abs(getattr(pose, 'roll', pose.get('roll', 0)))
            # print(f"Pitch: {pitch}, Roll: {roll}")
            # if pitch > 60 or roll > 60:
            #     return True, "crash: excessive pitch/roll"

            # # # 2. About to hit ground (relative_altitude < 0.1m)
            # # rel_alt = getattr(pose, 'relative_altitude', pose.get('relative_altitude', 0))
            # # if rel_alt < 0.1:
            # #     return True, "crash: hit ground"

            # # 3. Flip (pitch or roll > 90 deg)
            # if pitch > 90 or roll > 90:
            #     return True, "crash: flip"

            # # 4. 2x farther from goal than originally
            # # Assume self.goal is a dict/object with x, y, z; initial_pose is dict/object with x, y, z
            # if initial_pose is not None and hasattr(self, "goal"):
            #     def dist(p1, p2):
            #         return ((getattr(p1, 'x', p1.get('x', 0)) - getattr(p2, 'x', p2.get('x', 0))) ** 2 +
            #                 (getattr(p1, 'y', p1.get('y', p1.get('y', 0))) - getattr(p2, 'y', p2.get('y', 0))) ** 2 +
            #                 (getattr(p1, 'z', p1.get('z', 0)) - getattr(p2, 'z', p2.get('z', 0))) ** 2) ** 0.5
            #     dist_init = dist(initial_pose, self.goal)
            #     dist_now = dist(pose, self.goal)
            #     if dist_now > 2 * dist_init:
            #         return True, "too far from goal"

            return False

        terminated = _check_terminated(pose)
        if terminated:
            truncated = False
        else:
            terminated = False
            truncated = self.episode_step >= self.max_episode_steps
        info = {}
        return self._dict_to_obs(pid_params), reward, terminated, truncated, info

    async def _async_arm(self):
        logger.info("Arming...")
        drone = await self.sitl._get_mavsdk_connection()

        print("Waiting for vehicle to become armable...")
        async for health in drone.telemetry.health():
            if health.is_armable and health.is_global_position_ok:
                print("Vehicle is armable and has GPS fix!")
                break

        await drone.action.arm()
        await asyncio.sleep(1.0)

    async def _async_takeoff(self):
        logger.info("Taking off to 5 m...")

        drone = await self.sitl._get_mavsdk_connection()
        # async for state in drone.core.connection_state():
        #     logger.info(f"[MAVSDK] Connected: {state.is_connected}")
        logger.info("Taking off to 5 m--------------------------------")
        await drone.action.takeoff()
        await asyncio.sleep(5.0)

    async def _compute_reward(self, pid_params, pose):
        await asyncio.sleep(0.0000001)
        return 0.0

    def __compatibility_checks(self):
        if self.config['environment_config']['mode'] not in ['position', 'attitude', 'stabilize', 'althold', 'altitude']:
            raise ValueError(f"Invalid mode: {self.config['environment_config']['mode']}")
        logger.warning(f"compatibility checks is not implemented")

    def close(self):
        """Clean up resources."""
        print("Closing environment...")
        self.gazebo.close()
        self.sitl.close()
