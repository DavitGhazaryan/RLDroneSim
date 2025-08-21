import os
import yaml
from pathlib import Path
import numpy as np

ARDUPILOT_DIR = '/home/student/Dev/ardupilot'

def load_config(config_path):
    """Load configuration from YAML file."""
    if not config_path or not Path(config_path).exists():
        print(f"⚠️  Config file not found: {config_path}")
        print("Using default configuration...")
        return get_default_config()
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"❌ Error loading config file: {e}")
        print("Using default configuration...")
        return get_default_config()


def get_default_config():
    """Return default Gazebo configuration."""
    return {
        'ardupilot_config': {
            'ardupilot_path': ARDUPILOT_DIR,
            'vehicle': 'ArduCopter',
            'frame': 'gazebo-iris',
            'model': 'JSON',
            'timeout': 60.0
        },
        'gazebo_config': {
            'sdf_file': '/home/student/Dev/pid_rl/ardupilot_gazebo/worlds/simple_world.sdf',
            'gui': 'DISPLAY' in os.environ,
            'verbose': True,
            'timeout': 15.0
        }
    }


def euler_to_quaternion(euler):
    """
    Convert Euler angles (yaw, pitch, roll) to a quaternion.
    Angles are expected in radians.
    """
    # roll, pitch, yaw = euler['roll_deg'], euler['pitch_deg'], euler['yaw_deg']
    roll, pitch, yaw = 0.0, 0.0, 90.0
    roll, pitch, yaw = np.deg2rad([roll, pitch, yaw])
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    return [qx, qy, qz, qw]

def lat_lon_to_xy_meters(origin_pose, lat_deg, lon_deg):
    """
    Convert latitude and longitude to x,y coordinates in meters relative to the origin.
    
    Args:
        lat_deg (float): Latitude in degrees
        lon_deg (float): Longitude in degrees
        
    Returns:
        tuple: (x_meters, y_meters) relative to origin
        
    Note:
        Precision depends on distance from origin:
        - Within 1km: ~1-10cm precision
        - Within 10km: ~10cm-1m precision  
        - Within 100km: ~1-10m precision
        - Beyond 100km: precision degrades significantly
    """
    if origin_pose is None:
        raise ValueError("Origin pose not set. Call reset() first.")
        
    # Convert degrees to radians
    lat_rad = np.radians(lat_deg)
    lon_rad = np.radians(lon_deg)
    origin_lat_rad = np.radians(origin_pose['latitude_deg'])
    origin_lon_rad = np.radians(origin_pose['longitude_deg'])
    
    # Earth's radius in meters (approximate)
    earth_radius = 6371000.0  # meters
    
    # Calculate differences
    dlat = lat_rad - origin_lat_rad
    dlon = lon_rad - origin_lon_rad
    
    # Convert to meters using small-angle approximation
    # This is accurate for distances up to ~100km
    y_meters = dlat * earth_radius
    x_meters = dlon * earth_radius * np.cos(origin_lat_rad)
    
    return x_meters, y_meters

def xy_meters_to_lat_lon(origin_pose, x_meters, y_meters):
    """
    Convert x,y coordinates in meters relative to origin back to latitude/longitude.
    
    Args:
        x_meters (float): X coordinate in meters relative to origin
        y_meters (float): Y coordinate in meters relative to origin
        
    Returns:
        tuple: (latitude_deg, longitude_deg)
    """
    if origin_pose is None:
        raise ValueError("Origin pose not set. Call reset() first.")
        
    # Earth's radius in meters
    earth_radius = 6371000.0
    
    # Convert meters back to radians
    dlat_rad = y_meters / earth_radius
    dlon_rad = x_meters / (earth_radius * np.cos(np.radians(origin_pose['latitude_deg'])))
    
    # Add to origin coordinates
    lat_deg = origin_pose['latitude_deg'] + np.degrees(dlat_rad)
    lon_deg = origin_pose['longitude_deg'] + np.degrees(dlon_rad)
    
    return lat_deg, lon_deg


def validate_config(config, model):
    """
    Validate that the configuration contains all required parameters.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        bool: True if config is valid, False otherwise
    """
    if model != "ddpg":
        raise NotImplementedError("Only ddpg validation is implemented")

    required_sections = ['environment_config', 'ardupilot_config', 'gazebo_config', 'ddpg_params', 'training_config']
    
    for section in required_sections:
        if section not in config:
            print(f"❌ Missing required configuration section: {section}")
            return False
    
    # Check DDPG parameters
    ddpg_params = config['ddpg_params']
    required_ddpg_params = ['learning_rate', 'buffer_size', 'batch_size', 'tau', 'gamma']
    
    for param in required_ddpg_params:
        if param not in ddpg_params:
            print(f"❌ Missing required DDPG parameter: {param}")
            return False
    
    # Check training parameters
    training_params = config['training_config']
    required_training_params = ['total_timesteps', 'save_freq']
    
    for param in required_training_params:
        if param not in training_params:
            print(f"❌ Missing required training parameter: {param}")
            return False
    
    print("✅ Configuration validation passed!")
    return True


def demonstrate_observation_action_format(env):
    """Demonstrate the new observation and action format."""
    
    print(f"\n🔍 Demonstrating Observation and Action Format")
    print("=" * 60)
    
    # Show sample observations and actions
    sample_obs = env.observation_space.sample()
    sample_action = env.action_space.sample()
    
    print(f"📊 Sample Observation (Array):")
    print(f"   Shape: {sample_obs.shape}")
    print(f"   Values: {sample_obs}")
    
    print(f"\n🎯 Sample Action (Array):")
    print(f"   Shape: {sample_action.shape}")
    print(f"   Values: {sample_action}")
    
    # Show what each index represents
    obs_mapping = env.get_observation_key_mapping()
    action_mapping = env.get_action_key_mapping()
    
    print(f"\n🗺️  Observation Index Meaning:")
    for key, idx in obs_mapping.items():
        print(f"   obs[{idx}] = {key} = {sample_obs[idx]:.3f}")
    
    print(f"\n🎯 Action Index Meaning:")
    for key, idx in action_mapping.items():
        print(f"   action[{idx}] = {key} adjustment = {sample_action[idx]:.3f}")



def evaluate_agent(model, env, num_episodes=None):
    """
    Evaluate the trained agent.
    
    Args:
        model: Trained DDPG model
        env: Modified ArdupilotEnv
        num_episodes: Number of evaluation episodes (overrides config if provided)
        
    Returns:
        Evaluation results
    """
    # Get evaluation parameters from config
    if num_episodes is None:
        evaluation_config = env.config.get('evaluation_config', {}) if hasattr(env, 'config') else {}
        num_episodes = evaluation_config.get('n_eval_episodes', 5)
    
    print(f"\n🧪 Evaluating agent over {num_episodes} episodes...")
    
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0.0
        episode_length = 0
        
        print(f"   Episode {episode + 1}: ", end="")
        
        while True:
            # Get action from the trained model
            action, _ = model.predict(obs, deterministic=True)
            
            # Take step in environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            
            if terminated or truncated:
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        print(f"Reward: {episode_reward:.2f}, Length: {episode_length}")
    
    # Calculate statistics
    avg_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    avg_length = np.mean(episode_lengths)
    
    print(f"\n📊 Evaluation Results:")
    print(f"   Average reward: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"   Average episode length: {avg_length:.1f} steps")
    print(f"   Success rate: {sum(1 for r in episode_rewards if r > 0) / len(episode_rewards):.1%}")
    
    return {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'avg_reward': avg_reward,
        'std_reward': std_reward,
        'avg_length': avg_length
    }
