import unittest
import subprocess


SUCCESS_APP_PATH = "tests/test_utils/assets/apps/success_app.py"
FAILED_APP_PATH = "tests/test_utils/assets/apps/fail_app.py"

class CLITest(unittest.TestCase):
    def test_cli_run_success(self):
        result = subprocess.run(
            ["exiv", "run", SUCCESS_APP_PATH],
            capture_output=True,
            text=True,
            check=False
        )

        self.assertEqual(result.returncode, 0, "Script should exit with code 0 on success.")
        self.assertIn("Success script finished.", result.stdout)

    def test_cli_run_fail(self):
        result = subprocess.run(
            ["exiv", "run", FAILED_APP_PATH],
            capture_output=True,
            text=True,
            check=False
        )
        
        print("---- result: ", result)
        self.assertEqual(result.returncode, 0, "Script should exit with code 0 on failure.")
        self.assertIn("Exception occured:", result.stderr)