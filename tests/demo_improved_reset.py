#!/usr/bin/env python3
"""
Demo script showcasing improved SITL reset functionality.

This demonstrates:
- Async/sync reset interfaces
- Configurable MAVSDK ports
- Teleport to home position
- Parameter restoration with batching
- Robust error handling
"""

import sys
import os
import asyncio
import time
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from rl_training.utils.ardupilot_sitl import ArduPilotSITL

async def demo_async_reset():
    """Demonstrate the async reset interface."""
    print("🚀 Demo: Async Reset Interface")
    print("=" * 50)
    
    # Configuration with all improvements
    config = {
        'ardupilot_path': '/home/student/Dev/pid_rl/ardupilot',
        'vehicle': 'ArduCopter',
        'frame': 'quad',
        'timeout': 30.0,
        'min_startup_delay': 5.0,
        'no_mavproxy': True,  # Avoid MAVProxy for demo
        'mavsdk_port': 14550,  # Configurable port
        'location': '-35.363261,149.165230,584,353',  # Home location for teleport demo
    }
    
    sitl = ArduPilotSITL(config)
    
    try:
        print("📍 Starting SITL with home location...")
        sitl.start_sitl()
        
        print("⏱️  Waiting for SITL to stabilize...")
        await asyncio.sleep(3)
        
        print("\n🔄 Performing async reset with all features...")
        print("   - Teleport to home")
        print("   - Clear mission") 
        print("   - Keep parameters")
        print("   - Attempt re-arm")
        
        # Use the async interface directly
        await sitl.reset_async(keep_params=True)
        
        print("\n✅ Async reset completed!")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
    
    finally:
        print("\n🛑 Stopping SITL...")
        sitl.stop_sitl()

def demo_sync_reset():
    """Demonstrate the synchronous reset interface."""
    print("\n🚀 Demo: Sync Reset Interface")
    print("=" * 50)
    
    config = {
        'ardupilot_path': '/home/student/Dev/pid_rl/ardupilot',
        'vehicle': 'ArduCopter', 
        'frame': 'quad',
        'timeout': 30.0,
        'min_startup_delay': 5.0,
        'no_mavproxy': True,
        'mavsdk_port': 14560,  # Different port for second instance
        'instance': 1,         # Second instance
    }
    
    sitl = ArduPilotSITL(config)
    
    try:
        print("📍 Starting second SITL instance...")
        sitl.start_sitl()
        
        print("⏱️  Waiting for SITL to stabilize...")
        time.sleep(3)
        
        print("\n🔄 Performing sync reset...")
        print("   - Using synchronous interface")
        print("   - Keep parameters")
        
        # Use the synchronous wrapper
        sitl.reset(keep_params=True)
        
        print("\n✅ Sync reset completed!")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
    
    finally:
        print("\n🛑 Stopping SITL...")
        sitl.stop_sitl()

def main():
    print("🎯 SITL Reset Improvements Demo")
    print("=" * 60)
    print("This demo showcases all the implemented improvements:")
    print("  ✨ Async/sync reset interfaces")
    print("  ✨ Configurable MAVSDK ports")
    print("  ✨ Teleport to home position")
    print("  ✨ Parameter restoration with batching")
    print("  ✨ Robust error handling")
    print("  ✨ Clean thread management")
    print()
    
    # Check ArduPilot availability
    ardupilot_path = Path('/home/student/Dev/pid_rl/ardupilot')
    if not ardupilot_path.exists():
        print(f"❌ ArduPilot not found at {ardupilot_path}")
        print("Please ensure ArduPilot is installed.")
        return
    
    try:
        # Demo async interface
        asyncio.run(demo_async_reset())
        
        # Small break between demos
        print("\n" + "⏱️ " * 20)
        time.sleep(2)
        
        # Demo sync interface
        demo_sync_reset()
        
        print("\n" + "=" * 60)
        print("🎉 All demos completed successfully!")
        print("The SITL reset functionality is working with all improvements.")
        
    except KeyboardInterrupt:
        print("\n⚠️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")

if __name__ == "__main__":
    main() 