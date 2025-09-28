import unittest
import time
import json
import requests
import threading
import uvicorn
import websocket
from kirin.server import ScriptStatus
from kirin.server.server import app, start_worker, task_manager


TEST_HOST = "127.0.0.1"
TEST_PORT = 8008
BASE_URL = f"http://{TEST_HOST}:{TEST_PORT}"
SUCCESS_APP_PATH = "tests/test_utils/assets/apps/success_app.py"
FAILED_APP_PATH = "tests/test_utils/assets/apps/fail_app.py"

class ServerIntegrationTest(unittest.TestCase):
    server_thread = None
    worker_thread = None

    def setUp(self):
        """Set up and run the server and worker in background threads before any tests."""
        class StoppableUvicorn(uvicorn.Server):
            def run(self, *args, **kwargs):
                self._thread = threading.Thread(target=super().run, args=args, kwargs=kwargs)
                self._thread.start()
            def stop(self):
                self.should_exit = True
                self._thread.join(timeout=1)

        # start the uvicorn server thread
        task_manager.__init__()
        config = uvicorn.Config(app, host=TEST_HOST, port=TEST_PORT, log_level="info")
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
        self.worker_thread.join(timeout=1)
        self.server.stop()

    def test_server_is_starting(self):
        response = requests.get(f"{BASE_URL}/docs") # default fastapi endpoint
        self.assertEqual(response.status_code, 200)

    def test_successful_task_flow(self):
        # queue a new task
        response = requests.post(f"{BASE_URL}/queue", json={"filename": SUCCESS_APP_PATH})
        self.assertEqual(response.status_code, 200)
        task_id = response.json()["task_id"]
        self.assertIsNotNone(task_id)

        # poll the status endpoint until completion
        for _ in range(5):
            status_response = requests.get(f"{BASE_URL}/status/{task_id}")
            if status_response.json()["status"] == "completed":
                break
            time.sleep(1)
        
        final_status = requests.get(f"{BASE_URL}/status/{task_id}").json()
        self.assertEqual(final_status["status"], "completed")
        self.assertEqual(final_status["progress"], 1.0)

    def test_failed_task_flow(self):
        # queue a failing task
        response = requests.post(f"{BASE_URL}/queue", json={"filename": FAILED_APP_PATH})
        self.assertEqual(response.status_code, 200)
        task_id = response.json()["task_id"]

        time.sleep(2)
        final_status = requests.get(f"{BASE_URL}/status/{task_id}").json()
        
        self.assertEqual(final_status["status"], "failed")
        self.assertIn("This is a test error.", final_status["data"]["err_message"])

    def test_websocket_flow(self):
        response = requests.post(f"{BASE_URL}/queue", json={"filename": SUCCESS_APP_PATH})
        task_id = response.json()["task_id"]

        ws_url = f"ws://{TEST_HOST}:{TEST_PORT}/ws/status/{task_id}"
        ws = websocket.create_connection(ws_url)
        
        messages = []
        try:
            for _ in range(5):      # to avoid infinite loops
                message = ws.recv()
                data = json.loads(message)
                messages.append(data)
                if data['status'] in [ScriptStatus.COMPLETED.value, ScriptStatus.FAILED.value]:
                    break
        finally:
            ws.close()
        
        self.assertTrue(any(msg['status'] == ScriptStatus.PROCESSING.value for msg in messages))
        self.assertEqual(messages[-1]['status'], ScriptStatus.COMPLETED.value)