# Loading and Execution

The extension system in Exiv is designed to be dynamically loaded, allowing external code to patch core engine behaviors seamlessly. The entire lifecycle is managed by the `ExtensionRegistry` singleton.

Here is the step-by-step flow of how an extension goes from being registered to actually executing its logic.

## 1. Registration & Configuration
Extensions are typically registered into the system via the command-line interface:
```bash
exiv register <path-to-extension>
```
When this command runs:
1. It looks for a `requirements.txt` in the extension folder and installs any necessary dependencies via `pip`.
2. Upon successful installation, it adds the extension's path to the `exiv_config.json` file under the `"extensions"` list.

*Note: Built-in extensions (located in `src/exiv/extensions`) are automatically discovered and do not need to be manually registered.*

## 2. Discovery and Namespace Setup
During the system startup (or whenever `ExtensionRegistry.get_instance()` is called), the registry begins the initialization phase:
1. It reads the `exiv_config.json` to get the list of registered extension paths.
2. For each path, it checks for the presence of the required `extension.py` entrypoint file.
3. To prevent import conflicts and allow relative imports within the extension, the registry creates a consistent, isolated module namespace for each extension: `exiv.ext.<extension_folder_name>`.

You can also view the list of all registered extensions by running the following command in your terminal:
```bash
exiv list extensions
```

## 3. Manual Execution in Code
If an extension does not auto-execute via system patches and instead acts as a functional processor (e.g., a custom model or image transformation), it can be called manually inside your application code.

First, initialize the registry and retrieve the extension by its unique `ID`. Then, instantiate it and call its `process()` method with the required arguments.

Here is an example of how you might call an extension manually (e.g., `dwpose`):

```python
from exiv.components.extension_registry import ExtensionRegistry

# 1. Initialize the extension registry and load all extensions
registry = ExtensionRegistry.get_instance()

# 2. Access the extension by its ID and instantiate it
dwpose_ext = registry.extensions.get("dwpose")()

if dwpose_ext:
    # 3. Call the process method with the required arguments.
    # The arguments match what the extension defines in its process() method.
    out_tensor = dwpose_ext.process(
        image=input_frame, 
        detect_body=True, 
        detect_hand=True, 
        detect_face=False
    )
```