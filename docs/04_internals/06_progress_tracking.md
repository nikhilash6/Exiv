# Progress Tracking

Exiv provides a dynamic progress tracking mechanism for tasks (apps and extensions) using the `TaskContext` class. This allows complex workflows to report continuous progress to the client without needing to manually calculate global percentages at every step.

## How it works

The `TaskContext` class, accessible within an app or extension via the `context` parameter, manages progress in "chunks" or "stages" using **anchors**.

When you start a new logical stage of your task, you create an anchor. The `TaskContext` handles mapping the local progress of that stage (from 0.0 to 1.0) to the overall global progress of the entire task.

### 1. The `TaskContext` Object
When an app or extension handler is executed, the server injects a `context` argument (if accepted by the handler). This object is an instance of `exiv.server.app_core.TaskContext`.

### 2. Starting an Anchor
To define a new stage in your pipeline, use `start_anchor(name: str, steps: int = 1)`. 
- `name`: A descriptive name for the stage.
- `steps`: The relative weight of this stage (how many units of progress it consumes). By default, 1 step corresponds to the initialized `step_size`.

### 3. Reporting Progress
Within that stage, you call `progress(percent: float, status: str, stage: Optional[str] = None)`.
- `percent`: The completion of the *current stage* (a float between 0.0 and 1.0).
- `status`: A short message describing what is happening right now.
- `stage`: (Optional) Overrides the name of the stage.

The `TaskContext` will automatically translate your `percent` (e.g., 0.5 for 50% through the current anchor) into the correct global progress value before sending it to the client.

## Code Examples

### Example: Basic App with Progress

Here is how you might track progress across a multi-step generation task:

```python
from exiv.server.app_core import TaskContext

def my_app_handler(prompt: str, context: TaskContext):
    # Step 1: Preprocessing (takes 1 step)
    context.start_anchor("Preprocessing", steps=1)
    context.progress(0.0, "Tokenizing prompt...")
    # ... do some work ...
    context.progress(0.5, "Loading embeddings...")
    # ... do more work ...
    context.progress(1.0, "Preprocessing complete.")
    
    # Step 2: Generation (takes 4 steps, i.e., it's a longer process)
    context.start_anchor("Generation", steps=4)
    total_steps = 20
    for i in range(total_steps):
        # ... generate ...
        
        # Local percent goes from 0.0 to 1.0
        local_percent = (i + 1) / total_steps
        context.progress(local_percent, f"Generating step {i+1}/{total_steps}")
        
    # Step 3: Postprocessing
    context.start_anchor("Postprocessing", steps=1)
    context.progress(0.5, "Saving output...")
    # ... save ...
    context.progress(1.0, "Done!")
    
    return "output.png"
```

### Example: Passing Progress Callbacks to Models

Many underlying model components (like samplers or encoders) accept a `progress_callback` function. You can wrap `context.progress` to pass it down.

```python
def generate_image(prompt: str, context: TaskContext):
    context.start_anchor("Sampling", steps=10)
    
    # Define a callback that the sampler can call
    def my_callback(percent: float, message: str):
        context.progress(percent, message)
        
    # Pass the callback to the component
    my_sampler.sample(prompt, progress_callback=my_callback)
    
    return "result.png"
```
