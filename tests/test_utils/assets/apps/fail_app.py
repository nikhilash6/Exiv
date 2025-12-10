import time
from exiv.server.app_core import App

def handler(report_progress=None, **kwargs):
    print("Running fail script...")
    raise ValueError("This is a test error.")

app = App(
    name="fail_app",
    description="A test app that fails",
    inputs={},
    outputs=[],
    handler=handler
)