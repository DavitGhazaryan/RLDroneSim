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
    roll, pitch, yaw = euler['roll_deg'], euler['pitch_deg'], euler['yaw_deg']
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

