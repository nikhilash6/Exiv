# Operational Modes

Exiv is designed to be flexible, supporting various integration patterns from simple command-line execution to full-scale API serving.

Select how you intend to use Exiv:

<!-- install-grid -->
<!-- option: Interface = CLI | Server | Package -->

<!-- section: cli -->
## Command Line Interface (CLI)

The CLI provides the most direct way to interact with Exiv for experimentation and local runs.

### Standalone App Execution
You can execute pre-built applications in the `apps/` directory directly. These scripts parse command-line arguments and run logic locally.

```bash
python apps/wan_video.py --steps 20 --seed 42
```

### Global `exiv` Command
When installed in editable mode (`pip install -e .`), you can use the `exiv` command from anywhere:

```bash
exiv run wan_video --steps 20
```
<!-- /section -->

<!-- section: server -->
## Web Server

Launch Exiv as a backend server using FastAPI. This mode is ideal for building web applications or remote API services.

### Starting the Server
Run the following command to start the server:

```bash
exiv serve
```

By default, the server listens on `http://0.0.0.0:8000`.

### API Endpoints
Exiv provides a RESTful API and WebSocket support for managing tasks and resources:

- **`GET /api/apps`**: Lists all available applications and their input/output schemas.
- **`POST /api/apps/run`**: Queues a new task for execution.
- **`GET /status/{task_id}`**: Retrieves the current status and progress of a task.
- **`WS /ws/status/{task_id}`**: Real-time task progress updates via WebSocket.
- **`GET /api/extensions`**: Returns metadata for all registered extensions.
- **`GET /api/outputs`**: Lists files in the output directory.
- **`GET /api/outputs/{filename}`**: Downloads a specific output file.
- **`GET /api/apps/{app_name}/assets/{filename}`**: Serves frontend assets for custom app UIs.

### Use Cases
- **Frontend Integration**: Connect your React/Vue apps to the Exiv backend.
- **Task Management**: Handle long-running inference tasks asynchronously.
- **Remote Access**: Host your models on a GPU-enabled machine and access them via API.
<!-- /section -->

<!-- section: package -->
## Python Package

Integrate Exiv directly into your existing Python projects as a library. This provides the most granular control over model loading, patching, and execution.

### Basic Usage

```python
import torch
from exiv.components.models.wan.constructor import get_wan_instance
from exiv.model_utils.common_classes import ModelWrapper

# Initialize a model
wan_model = get_wan_instance("path/to/model.safetensors", "https://model-url.com")
model_wrapper = ModelWrapper(model=wan_model)

# Use Exiv's utilities for patching or inference
# ...
```

### Why use Package mode?
- **Custom Pipelines**: Build specialized AI workflows that aren't covered by the built-in apps.
- **Deep Integration**: Embed Exiv's memory management and quantization features directly into your application logic.
<!-- /section -->
