import unittest
import time
import json
import requests
import threading
import uvicorn
import websocket
import os
import importlib


TEST_APPS_DIR = os.path.abspath("tests/test_utils/assets/apps")
os.environ["EXIV_APPS_DIR"] = TEST_APPS_DIR

# reload server to pick up the new ENV variable and load test apps
from exiv.server import server, task_manager, ScriptStatus
importlib.reload(server)

from exiv.server.task_manager import task_manager, ScriptStatus
from exiv.server.server import app, start_worker

TEST_HOST = "127.0.0.1"
TEST_PORT = 8008
BASE_URL = f"http://{TEST_HOST}:{TEST_PORT}"

class ServerIntegrationTest(unittest.TestCase):
    server_thread = None
    worker_thread = None

    def setUp(self):
        """Set up and run the server and worker in background threads before any tests."""
        task_manager.task_dict = {}
        task_manager.task_queue.queue.clear()

        class StoppableUvicorn(uvicorn.Server):
            def run(self, *args, **kwargs):
                self._thread = threading.Thread(target=super().run, args=args, kwargs=kwargs)
                self._thread.start()
            def stop(self):
                self.should_exit = True
                self._thread.join(timeout=1)

        # start the uvicorn server thread
        config = uvicorn.Config(app, host=TEST_HOST, port=TEST_PORT, log_level="warning")
        self.server = StoppableUvicorn(config=config)
        self.server.run()
        
        # start the worker thread
        self.worker_thread = threading.Thread(target=start_worker)
        self.worker_thread.daemon = True
        self.worker_thread.start()
        
        time.sleep(1)

    def tearDown(self):
        """Shut down the server and worker after all tests are done."""
        task_manager.task_queue.put((None, None))
        try:
            self.worker_thread.join(timeout=1)
        except: pass
        self.server.stop()

    def test_server_is_starting(self):
        response = requests.get(f"{BASE_URL}/docs") # default fastapi endpoint
        self.assertEqual(response.status_code, 200)

    def test_successful_task_flow(self):
        # queue a new task
        payload = {"app_name": "success_app", "params": {}}
        response = requests.post(f"{BASE_URL}/api/apps/run", json=payload)
        
        self.assertEqual(response.status_code, 200, f"Response: {response.text}")
        task_id = response.json().get("task_id")
        self.assertIsNotNone(task_id)

        # poll the status endpoint until completion
        final_status = None
        for _ in range(10):
            status_response = requests.get(f"{BASE_URL}/status/{task_id}")
            data = status_response.json()
            if data["status"] == "completed":
                final_status = data
                break
            time.sleep(0.5)
        
        self.assertIsNotNone(final_status)
        self.assertEqual(final_status["status"], "completed")
        self.assertEqual(final_status["progress"], 1.0)

    def test_failed_task_flow(self):
        # queue a failing task
        payload = {"app_name": "fail_app", "params": {}}
        response = requests.post(f"{BASE_URL}/api/apps/run", json=payload)
        self.assertEqual(response.status_code, 200)
        task_id = response.json()["task_id"]

        final_status = None
        for _ in range(10):
            status_response = requests.get(f"{BASE_URL}/status/{task_id}")
            data = status_response.json()
            if data["status"] == "failed":
                final_status = data
                break
            time.sleep(0.5)
        
        self.assertIsNotNone(final_status)
        self.assertEqual(final_status["status"], "failed")
        self.assertIn("This is a test error", final_status["data"]["err_message"])

    def test_websocket_flow(self):
        payload = {"app_name": "success_app", "params": {}}
        response = requests.post(f"{BASE_URL}/api/apps/run", json=payload)
        task_id = response.json()["task_id"]

        ws_url = f"ws://{TEST_HOST}:{TEST_PORT}/ws/status/{task_id}"
        ws = websocket.create_connection(ws_url)
        
        messages = []
        try:
            for _ in range(10):      # to avoid infinite loops
                message = ws.recv()
                data = json.loads(message)
                messages.append(data)
                if data['status'] in [ScriptStatus.COMPLETED.value, ScriptStatus.FAILED.value]:
                    break
        finally:
            ws.close()
        
        statuses = [msg['status'] for msg in messages]
        self.assertIn(ScriptStatus.COMPLETED.value, statuses)