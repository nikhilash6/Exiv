import json
import os
import time
import tempfile
import threading
import unittest
import requests
import uvicorn
import importlib

TEST_APPS_DIR = os.path.abspath("tests/test_utils/assets/apps")
os.environ["EXIV_APPS_DIR"] = TEST_APPS_DIR

from exiv.server import server
importlib.reload(server)

from exiv.server.server import get_app, load_apps_from_directory
from exiv.config import global_config

TEST_HOST = "127.0.0.1"
TEST_PORT = 8009
BASE_URL = f"http://{TEST_HOST}:{TEST_PORT}"

EXPECTED_CONFIG_KEYS = {
    "log_level", "low_vram", "no_oom", "normal_load",
    "auto_download",
}


class ConfigApiTest(unittest.TestCase):

    def setUp(self):
        # ensure apps are loaded for the test
        load_apps_from_directory()
        app = get_app()

        class StoppableUvicorn(uvicorn.Server):
            def run(self, *args, **kwargs):
                self._thread = threading.Thread(target=super().run, args=args, kwargs=kwargs)
                self._thread.start()
            def stop(self):
                self.should_exit = True
                self._thread.join(timeout=1)

        config = uvicorn.Config(app, host=TEST_HOST, port=TEST_PORT, log_level="warning")
        self.server = StoppableUvicorn(config=config)
        self.server.run()
        time.sleep(1)

    def tearDown(self):
        self.server.stop()

    def test_get_config_returns_all_keys(self):
        response = requests.get(f"{BASE_URL}/api/config")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        for key in EXPECTED_CONFIG_KEYS:
            self.assertIn(key, data, f"Key '{key}' missing from /api/config response")

    def test_post_config_updates_in_memory(self):
        current = requests.get(f"{BASE_URL}/api/config").json()
        new_val = not current["auto_download"]

        response = requests.post(
            f"{BASE_URL}/api/config",
            json={"auto_download": new_val},
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["config"]["auto_download"], new_val)

        updated = requests.get(f"{BASE_URL}/api/config").json()
        self.assertEqual(updated["auto_download"], new_val)


class ConfigFileHelperTest(unittest.TestCase):

    def setUp(self):
        self._tmp_dir = tempfile.mkdtemp()
        self._config_file = os.path.join(self._tmp_dir, "config.json")

    def tearDown(self):
        if os.path.exists(self._config_file):
            os.remove(self._config_file)
        os.rmdir(self._tmp_dir)

    def test_section_isolation_settings_does_not_touch_extensions(self):
        from pathlib import Path
        from exiv.utils.config_file import save_section

        cfg_path = Path(self._config_file)
        initial = {
            "extensions": ["path/to/ext_a", "path/to/ext_b"],
            "settings": {"auto_download": False},
        }
        with open(cfg_path, "w") as f:
            json.dump(initial, f)

        save_section(cfg_path, "settings", {"auto_download": True, "low_vram": False})

        with open(cfg_path) as f:
            result = json.load(f)

        self.assertEqual(result["extensions"], ["path/to/ext_a", "path/to/ext_b"])
        self.assertTrue(result["settings"]["auto_download"])
        self.assertFalse(result["settings"]["low_vram"])

    def test_section_isolation_extensions_does_not_touch_settings(self):
        from pathlib import Path
        from exiv.utils.config_file import save_section

        cfg_path = Path(self._config_file)
        initial = {
            "extensions": [],
            "settings": {"auto_download": True},
        }
        with open(cfg_path, "w") as f:
            json.dump(initial, f)

        save_section(cfg_path, "extensions", ["path/to/new_ext"])

        with open(cfg_path) as f:
            result = json.load(f)

        self.assertEqual(result["extensions"], ["path/to/new_ext"])
        self.assertTrue(result["settings"]["auto_download"])

    def test_load_config_creates_file_with_defaults(self):
        from pathlib import Path
        from exiv.utils.config_file import load_config, DEFAULT_SETTINGS

        cfg_path = Path(self._config_file)
        self.assertFalse(cfg_path.exists())

        data = load_config(cfg_path)

        self.assertTrue(cfg_path.exists())
        self.assertIsInstance(data.get("extensions"), list)
        self.assertIsInstance(data.get("settings"), dict)
        for key in DEFAULT_SETTINGS:
            self.assertIn(key, data["settings"])

    def test_load_config_adds_settings_to_legacy_file(self):
        from pathlib import Path
        from exiv.utils.config_file import load_config

        cfg_path = Path(self._config_file)
        with open(cfg_path, "w") as f:
            json.dump({"extensions": ["some/path"]}, f)

        data = load_config(cfg_path)

        self.assertEqual(data["extensions"], ["some/path"])
        self.assertIsInstance(data.get("settings"), dict)


if __name__ == "__main__":
    unittest.main()
