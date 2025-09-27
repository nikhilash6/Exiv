import unittest
import time
import requests
import threading
import uvicorn
import websocket
from kirin.server.server import app, start_worker, task_manager
from kirin.server.task_manager import ScriptRequest

# --- Configuration for the test server ---
TEST_HOST = "127.0.0.1"
TEST_PORT = 8008
BASE_URL = f"http://{TEST_HOST}:{TEST_PORT}"

class ServerIntegrationTest(unittest.TestCase):
    server_thread = None
    worker_thread = None

    def setUp(cls):
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
        cls.server = StoppableUvicorn(config=config)
        cls.server.run()
        
        # start the worker thread
        cls.worker_thread = threading.Thread(target=start_worker)
        cls.worker_thread.daemon = True
        cls.worker_thread.start()
        
        time.sleep(1)

    def tearDown(cls):
        """Shut down the server and worker after all tests are done."""
        task_manager.task_queue.put((None, None))
        cls.worker_thread.join(timeout=1)
        cls.server.stop()

    def test_server_is_starting(self):
        response = requests.get(f"{BASE_URL}/docs") # default fastapi endpoint
        self.assertEqual(response.status_code, 200)


if __name__ == '__main__':
    unittest.main()