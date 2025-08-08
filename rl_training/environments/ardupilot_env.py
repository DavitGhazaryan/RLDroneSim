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

        self.old_observation = None

        # Episode tracking
        self.initialized = False
        self.episode_step = 0
        self.environment_config = config.get('environment_config', {})
        self.max_episode_steps = self.environment_config.get('max_episode_steps', 100)

        # Initialize spaces
        self.observation_space = self._define_observation_space()
        self.action_space = self._define_action_space()
            
    def _define_observation_space(self):
        ## TODO: Define the observation space based on the environment_config
        low = np.full(6, -np.inf, dtype=np.float32)
        high = np.full(6, np.inf, dtype=np.float32)
        return spaces.Box(low=low, high=high, dtype=np.float32)
    
    def _define_action_space(self):
        ## TODO: Define the action space based on the environment_config
        # Each action is a delta for a PID gain, bounded between -0.1 and 0.1
        low = np.array([-0.1] * 6, dtype=np.float32)
        high = np.array([0.1] * 6, dtype=np.float32)
        return spaces.Box(low=low, high=high, dtype=np.float32)

    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        # TODO reset the world  using the reset interfaces functions

        self.episode_step = 0

        if not self.initialized:
            logger.info("🌎 Launching Gazebo simulation...")
            self.gazebo.start_simulation()
            self.gazebo._wait_for_startup()
            self.gazebo.resume_simulation()
            logger.info("✅ Gazebo initialized")

            logger.info("🚁 Starting ArduPilot SITL...")
            self.sitl.start_sitl()
            info = self.sitl.get_process_info()
            logger.info(f"✅ SITL running (PID {info['pid']})")
            self.initialized = True
        else:
            raise NotImplementedError("Resetting the environment SECOND TIME is not implemented for this environment.")
    
        self.old_observation, info = self.loop.run_until_complete(self._async_reset())
        return self.old_observation, info  # observation, info

    async def _async_reset(self):
        """
        Reset the environment to initial state.
        This function is called when the environment is reset.

        It arms the vehicle, takes off to 5 m, and returns the initial observation.
        """
        observation = await self.sitl.get_pid_params_async()
        drone = await self.sitl._get_mavsdk_connection()

        # position = await anext(drone.telemetry.position())
        # print(f"Position: {position}")

        print("Waiting for vehicle to become armable...")
        async for health in drone.telemetry.health():
            if health.is_armable and health.is_global_position_ok:
                print("Vehicle is armable and has GPS fix!")
                break
        
        await asyncio.sleep(1)
        # arm
        print("Arming...")
        await drone.action.arm()
        await asyncio.sleep(1)

        # takeoff to 5 m
        print("Taking off to 5 m...")
        await drone.action.takeoff()
        await asyncio.sleep(5.0)
        
        return self.dict_to_obs(observation), {}  # observation, info

    def step(self, action):
        self.episode_step += 1
        obs, reward, done, truncated, info = self.loop.run_until_complete(self._async_step(action))
        self.old_observation = obs
        return obs, reward, done, truncated, info

    async def _async_step(self, action):
        pid_params = self.old_observation + action
        await self.sitl.set_params_async(pid_params)
        
        obs = await self.sitl.get_pid_params_async()
        
        reward = await self._compute_reward(obs, pid_params)
        done = False
        truncated = self.episode_step >= self.max_episode_steps
        info = {}
        await asyncio.sleep(10)
        return self.dict_to_obs(obs), reward, done, truncated, info

    async def _compute_reward(self, obs, pid_params):
        print(f"Calculating reward for")
        await asyncio.sleep(1)
        return 0.0

    def render(self, mode="human"):
        """Render the environment."""
        pass 

    def dict_to_obs(self, pid_dict):
        PID_KEYS = [
            "ATC_ANG_PIT_P",
            "ATC_ANG_RLL_P",
            "ATC_ANG_YAW_P",
            "ATC_RAT_PIT_P",
            "ATC_RAT_RLL_P",
            "ATC_RAT_YAW_P",
        ]
        return np.array([pid_dict[k] for k in PID_KEYS], dtype=np.float32)

    def __compatibility_checks(self):
        if self.config['environment_config']['mode'] not in ['position', 'attitude', 'stabilize', 'althold']:
            raise ValueError(f"Invalid mode: {self.config['environment_config']['mode']}")
        logger.warning(f"compatibility checks is not implemented")

    def close(self):
        """Clean up resources."""
        print("Closing environment...")
        self.gazebo.close()
        self.sitl.close()


if __name__ == "__main__":
    config = load_config('/home/pid_rl/rl_training/configs/default_config.yaml')
    env = ArdupilotEnv(config)

    env.reset()

    env.step(np.array([0.3, 0.3, 0.3, 0.3, 0.3, 0.3]))
    env.close()