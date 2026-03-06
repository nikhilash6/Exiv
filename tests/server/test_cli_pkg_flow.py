import unittest
import subprocess
import os
import sys

# Point to the assets folder
TEST_APPS_DIR = "tests/test_utils/assets/apps"

class CLITest(unittest.TestCase):
    def test_cli_run_success(self):
        env = os.environ.copy()
        env["EXIV_APPS_DIR"] = TEST_APPS_DIR
        
        result = subprocess.run(
            [sys.executable, "-m", "exiv.main", "run", "success_app"],
            capture_output=True,
            text=True,
            check=False,
            env=env
        )

        self.assertEqual(result.returncode, 0, "Script should exit with code 0.")
        # Check task status in stdout, relying on json-like structure or printed logs
        self.assertIn("Success script finished.", result.stdout)

    def test_cli_run_fail(self):
        env = os.environ.copy()
        env["EXIV_APPS_DIR"] = TEST_APPS_DIR
        
        result = subprocess.run(
            [sys.executable, "-m", "exiv.main", "run", "fail_app"],
            capture_output=True,
            text=True,
            check=False,
            env=env
        )
        
        print("---- result stdout: ", result.stdout)
        print("---- result stderr: ", result.stderr)
        self.assertEqual(result.returncode, 0)
        self.assertIn("Running fail script...", result.stdout)
        self.assertIn("This is a test error", result.stderr)