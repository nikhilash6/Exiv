# Building a Custom Extension

Writing a custom extension allows you to integrate third-party tools, models, or custom behaviors into the Exiv ecosystem.

In this guide, we will walk through building the **Greeting Extension**, a simple extension that prints a random movie quote to the console every time the Exiv system starts up.

## 1. Directory Structure
A valid extension must live in its own directory and expose a specific entry point called `extension.py`. 

For the Greeting Extension, the structure looks like this:

```
extensions/greeting_extension/
├── data/
│   └── movie_greetings.json
├── extension.py
├── LICENSE
├── README.md
└── requirements.txt
```

## 2. The Entry Point (`extension.py`)
The most important file in your extension is `extension.py`. The Extension Registry will dynamically load this file and look for any class that inherits from the base `exiv.components.extensions.Extension` class.

Here is the complete code for our Greeting Extension:

```python
import os
import json
import random
from exiv.components.extensions import Extension
from exiv.utils.logging import app_logger

class GreetingExtension(Extension):
    # 1. Define metadata for your extension
    ID = "greeting_extension"
    DISPLAY_NAME = "Startup Greeter"
    VERSION = "0.0.1"
    
    def register(self):
        # 2. Return a list of capabilities
        # By returning a callable method, we are creating a "System Patch". 
        # The registry will execute this callable immediately during initialization.
        return [self.show_random_greeting]

    def show_random_greeting(self):
        # 3. Define the actual logic
        try:
            # We can use __file__ to resolve paths relative to our extension folder
            json_path = os.path.join(os.path.dirname(__file__), "data", "movie_greetings.json")
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    greetings = json.load(f)
                    msg = random.choice(greetings)
                    app_logger.info(f"{msg}")
        except Exception as e:
            app_logger.warning(f"Failed to greet: {e}")
```

### Breaking Down the Code:
1. **Metadata (`ID`, `DISPLAY_NAME`, `VERSION`)**: Every extension must declare these basic properties. The `ID` must be unique across all installed extensions.
2. **`register()` Method**: This method is called by the Extension Registry when the extension is loaded. 
   - If you return a callable (like `self.show_random_greeting`), Exiv will execute it immediately during initialization. This is useful for executing system hooks, monkey-patching core components, or performing startup tasks.
   - If your extension acts as a processing node (e.g., performing inference or processing media), you can return `[self]`, indicating that the extension itself is a tool that apps can call later via the `extension.process()` method.
3. **Internal Logic (`show_random_greeting`)**: You can define any custom Python logic. Note how we use `os.path.dirname(__file__)` to safely read internal files bundled with our extension (like the `movie_greetings.json` file).

## 3. Registering the Extension
To make the system aware of your new extension, run the following command in your terminal from your project's root:

```bash
exiv register extensions
```

This command will:
1. Automatically install any dependencies found in your `requirements.txt`.
2. Add the extension's path to your system's `exiv_config.json`.

Once registered, you can verify your extension was successfully added by running:
```bash
exiv list extensions
```

The next time your application runs, the `GreetingExtension` will be dynamically loaded and will print a greeting to the console!
