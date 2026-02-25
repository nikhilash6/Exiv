# Quick Start

The fastest way to verify your installation is to run one of the pre-built test applications.

## Running an App Standalone
You can execute an inbuit app directly after installation. Remember this will download more than 20 GB of model weights the first time you run it.

```bash
exiv run wan_video --prompt "a dog running in the park"
```

**Expected Output:**
```text
/path/to/output.mp4
```

## What just happened?
Exiv applications wrap a core processing `handler` with structured `inputs` and `outputs`. When you run an app standalone, it initializes the necessary models and configurations and runs the logic locally. 

To learn more about tweaking VRAM usage before running heavier models, check out the [Configuration](../02_configuration/01_dynamic_offloading.md) section. To see what else you can run, head over to [Apps](../03_apps/01_overview.md).