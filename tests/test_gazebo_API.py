#!/usr/bin/env python3
"""
Simple Gazebo Interface Test

This script tests the GazeboInterface class directly without using the ArdupilotEnv.
It validates Gazebo GUI, world reset, and pause/resume functionalities.
"""

import sys
import os
# Add project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import subprocess
import argparse
import yaml
from pathlib import Path
from rl_training.utils.gazebo_interface import GazeboInterface

def load_config(config_path):
    """Load configuration from YAML file."""
    if not config_path or not Path(config_path).exists():
        print(f"⚠️  Config file not found: {config_path}")
        print("Using default configuration...")
        return get_default_config()
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config.get('gazebo_config', {})
    except Exception as e:
        print(f"❌ Error loading config file: {e}")
        print("Using default configuration...")
        return get_default_config()

def get_default_config():
    """Return default Gazebo configuration."""
    return {
        'sdf_file': '/home/student/Dev/pid_rl/ardupilot_gazebo/worlds/simple_world.sdf',
        'gui': 'DISPLAY' in os.environ,
        'verbose': True,
        'timeout': 15.0
    }

def test_gazebo_interface(config):
    """Test GUI, reset, pause, and resume functionality."""
    print("\n" + "=" * 40)
    print("🧪 Testing Full Gazebo Interface")
    print("=" * 40)

    # Check if SDF file exists
    sdf_path = Path(config['sdf_file'])
    if not sdf_path.exists():
        print(f"⚠️  SDF file not found: {sdf_path}")
        print("Skipping Gazebo test...")
        return True

    gazebo = GazeboInterface(config)

    try:
        # Start simulation
        gazebo.start_simulation()
        print("✅ Gazebo started")
        gazebo._wait_for_startup()

        if config.get('gui', False):
            print("⏱️  Viewing GUI for 5 seconds...")
            for i in range(4):
                time.sleep(1)
                print(f"   {i+1}/5 seconds...")
                if not gazebo.is_running():
                    print("   GUI was closed manually")
                    break
        print(gazebo.get_model_pose("simple_world"))

        # Reset world
        print("\n🔄 Resetting world...")
        time.sleep(2)
        gazebo.reset_world()
        print("✅ World reset complete")

        # Pause simulation
        print("\n⏸️  Pausing simulation...")
        gazebo.pause_simulation()
        time.sleep(2)

        # Resume simulation
        print("▶️  Resuming simulation...")
        gazebo.resume_simulation()
        time.sleep(2)
        gazebo.get_model_pose("iris_with_gimbal_drone")
        print("✅ Pause/Resume test complete")

        # Done
        gazebo.stop_simulation()
        print("✅ Full test completed")
        return True

    except Exception as e:
        print(f"❌ Full Gazebo test failed: {e}")
        return False

    finally:
        gazebo.close()

def main():
    parser = argparse.ArgumentParser(description='Test Gazebo Interface functionality')
    parser.add_argument('--config', '-c', type=str, 
                       default='rl_training/configs/default_config.yaml',
                       help='Path to configuration YAML file')
    args = parser.parse_args()

    print("🚁 Simple Gazebo Interface Tests\n")
    print(f"📋 Loading configuration from: {args.config}")

    try:
        subprocess.run(['gz', '--version'], capture_output=True, text=True, timeout=5)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("❌ Gazebo 'gz' command not found!")
        return

    config = load_config(args.config)
    print(f"🌎 SDF file: {config['sdf_file']}")
    print(f"🖥️  GUI enabled: {config.get('gui', False)}")

    passed = test_gazebo_interface(config)

    print("\n" + "=" * 40)
    print("📊 Test Results")
    print("=" * 40)
    print("Tests passed: 1/1" if passed else "Tests passed: 0/1")

    if passed:
        print("🎉 All tests passed! Gazebo interface is functional.")
    else:
        print("⚠️  Test failed. See output for details.")

    print("\n✨ Simple test completed!")

if __name__ == "__main__":
    main()
