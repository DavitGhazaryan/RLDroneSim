import unittest
import time
import logging
from pathlib import Path
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Add project root to Python path
from rl_training.utils.ardupilot_sitl import ArduPilotSITL



# Configure logging for the test
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TestArduPilotSITL(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Ensure ArduPilot directory exists for tests
        cls.ardupilot_path = Path('/home/student/Dev/pid_rl/ardupilot')
        if not cls.ardupilot_path.exists():
            raise unittest.SkipTest(f"ArduPilot directory not found: {cls.ardupilot_path}")

    def setUp(self):
        # Base configuration for tests
        self.config = {
            'ardupilot_path': str(self.ardupilot_path),
            'vehicle': 'ArduCopter',
            'frame': 'quad',
            'timeout': 40.0,
            'min_startup_delay': 5.0,
            'no_mavproxy': False,     # avoid launching MAVProxy
            'mavsdk_port': 14550,     # configurable MAVSDK port
        }
        self.sitl = ArduPilotSITL(self.config)

    # def test_start_and_info(self):
    #     """
    #     Test that SITL starts and returns valid process info.
    #     """
    #     self.sitl.start_sitl()
    #     self.assertTrue(self.sitl.is_running(), "SITL should be running after start_sitl()")

    #     info = self.sitl.get_process_info()
    #     self.assertEqual(info.get('status'), 'running', "Process info status must be 'running'")
    #     self.assertIn('pid', info, "Process info should contain 'pid'")
    #     self.assertGreater(info.get('pid'), 0, "PID must be a positive integer")
    #     self.assertIn('memory_mb', info)
    #     self.assertIn('cpu_percent', info)
        
    #     # Let SITL run briefly
    #     time.sleep(3)
    #     self.assertTrue(self.sitl.is_running(), "SITL should still be running after sleep")

    # def test_restart(self):
    #     """
    #     Test that restart_sitl properly restarts the process.
    #     """
    #     self.sitl.start_sitl()
    #     pid_before = self.sitl.get_process_info().get('pid')
    #     self.sitl.restart_sitl()
    #     pid_after = self.sitl.get_process_info().get('pid')
        
    #     # After restart, PID should change
    #     self.assertNotEqual(pid_before, pid_after, "PID should differ after restart")
    #     self.assertTrue(self.sitl.is_running())

    def test_reset_keep_params(self):
        """
        Test that reset(keep_params=True) teleports without changing PID.
        """
        self.sitl.start_sitl()
        print("Starting SITL")
        pid_before = self.sitl.get_process_info().get('pid')
        print(f"PID before reset: {pid_before}")
        self.sitl.reset(keep_params=True)
        print("Resetting SITL with keep_params=True")
        time.sleep(1)
        pid_after = self.sitl.get_process_info().get('pid')
        print(f"PID after reset: {pid_after}")
        # PID must remain the same
        self.assertEqual(pid_before, pid_after, "PID should not change on keep_params reset")
        self.assertTrue(self.sitl.is_running(), "SITL should be running after reset(keep_params=True)")

    # def test_reset_wipe_params(self):
    #     """
    #     Test that reset(keep_params=False) teleports and wipes parameters without changing PID.
    #     """
    #     self.sitl.start_sitl()
    #     pid_before = self.sitl.get_process_info().get('pid')
    #     self.sitl.reset(keep_params=False)
    #     time.sleep(1)
    #     pid_after = self.sitl.get_process_info().get('pid')

    #     # PID must remain the same
    #     self.assertEqual(pid_before, pid_after, "PID should not change on wipe_params reset")
    #     self.assertTrue(self.sitl.is_running(), "SITL should be running after reset(keep_params=False)")

    # def test_stop(self):
    #     """
    #     Test that stop_sitl cleanly terminates the process.
    #     """
    #     self.sitl.start_sitl()
    #     time.sleep(1)
    #     self.sitl.stop_sitl()
    #     self.assertFalse(self.sitl.is_running(), "SITL should not be running after stop_sitl()")

    # def test_process_info_fields(self):
    #     """
    #     Test that get_process_info returns all expected keys.
    #     """
    #     self.sitl.start_sitl()
    #     info = self.sitl.get_process_info()
    #     self.sitl.stop_sitl()
    #     expected_keys = {'status','pid','cpu_percent','memory_mb','num_children','uptime_s'}
    #     self.assertTrue(expected_keys.issubset(info.keys()), f"Missing keys: {expected_keys - set(info.keys())}")

    def tearDown(self):
        # Ensure clean shutdown
        try:
            self.sitl.stop_sitl()
        except:
            pass
        self.assertFalse(self.sitl.is_running(), "SITL should be stopped after tearDown")

if __name__ == '__main__':
    unittest.main()
