from exiv.server.app_core import App, Input, Output, AppOutputType

def add_numbers(num1, num2, context=None):
    if context:
        context.progress(0.5, "Calculating...")
    
    result = num1 + num2
    return {"1": str(result)}

app = App(
    name="calculator",
    inputs={
        "num1": Input(label="Number 1", type="number", default=0),
        "num2": Input(label="Number 2", type="number", default=0),
    },
    outputs=[Output(id=1, type=AppOutputType.STRING.value)],
    handler=add_numbers
)
