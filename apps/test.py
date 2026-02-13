from exiv.server.app_core import App, AppOutputType, Output
from exiv.utils.inputs import Input

def main(*args, **kwargs):
    print("testing")

app = App(
    name="Text to Video",
    inputs={
        'conditions': Input(label="Conditions (JSON)", type="json",),
    },
    outputs=[Output(id=1, type=AppOutputType.VIDEO.value)],
    handler=main
)

if __name__ == "__main__":
    app.run_standalone()