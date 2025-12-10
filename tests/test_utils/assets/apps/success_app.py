import time
from exiv.server.app_core import App

def handler(report_progress=None, **kwargs):
    print("Running success script...")
    if report_progress:
        report_progress(0.1, "Working")
    time.sleep(1) # Simulate work
    print("Success script finished.")
    return {"result": "ok"}

app = App(
    name="success_app",
    description="A test app that succeeds",
    inputs={},
    outputs=[],
    handler=handler
)