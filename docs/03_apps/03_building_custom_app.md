# Building a Simple App

Creating a custom App in Exiv allows you to encapsulate a specific ML workflow into a repeatable, easily shareable unit.
All apps extend the base `App` class from `exiv.server.app_core`. 

**Steps:**
1.  Define your `Input` expectations.
2.  Define your `Output` types.
3.  Write a `handler` function to do the actual inference work.

## The Calculator App
Let's build a basic calculator app to understand the flow. You will need to create a new folder inside `apps` and place an `app.py` file inside it defining your core logic.

1. Create a directory: `apps/calculator/`
2. Create a file: `apps/calculator/app.py`

Here is the implementation for `app.py`:

```python
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
```

### How it Works
1. **Dynamic UI:** Notice that we haven't written any UI code. Because we defined our inputs using the `Input` class (specifying type `number`), the Exiv frontend will automatically generate and display a dynamic form for this app.
2. **Data Flow:** When a user submits the dynamically generated form in the UI, the frontend sends the inputs (`num1` and `num2`) to the backend.
3. **Processing:** The data arrives at the backend and gets passed into our `handler` function (`add_numbers`), which performs the calculation.
4. **Output:** The result is structured into a dictionary (keyed by output ID, `"1"`) and sent back to the frontend, which displays the calculated string.
