# Model Paths

Exiv uses a flexible, multi-root path resolution system to find, load, and organize your model weights and outputs. It allows you to use a standard directory structure out-of-the-box, or add custom directories to share models with other UIs (like ComfyUI or Automatic1111).

## Default Directory Structure

By default, the engine registers your current working directory (`.`) as the primary search root and expects models to be placed in specific folders based on their type:

```text
kirin/
├── models/
│   ├── checkpoints/      (Full models)
│   ├── unet/             (UNet models, diffusion models)
│   ├── diffusion_models/ 
│   ├── loras/            (LoRA weights)
│   ├── vae/              (VAEs)
│   ├── clip/             (Text encoders)
│   └── clip_vision/      (Vision encoders)
├── input/                (Input media)
└── output/               (Generated outputs)
```

## Adding Custom Directories

If you already have models downloaded elsewhere on your system (e.g., inside a ComfyUI installation), you can register that external directory as an additional search path without moving or duplicating your files.

You can do this programmatically using the `FilePaths.add_search_path()` method:

```python
from exiv.utils.file_path import FilePaths

# Add a generic search path using the default folder mapping
FilePaths.add_search_path("/path/to/ComfyUI")

# Or, add a search path with a custom folder mapping
FilePaths.add_search_path("/path/to/my_models", mapping={
    "checkpoint": ["Checkpoints", "Stable-diffusion"],
    "lora": ["LoRAs"],
    "vae": "VAE"
})
```

**How Path Resolution Works:**
When an App requests a model (e.g., `wan21_1_3B.safetensors` of type `checkpoint`), the engine searches in this order:
1. **Direct Path:** If you provide an absolute or explicit relative path (e.g., `./my_custom_model.safetensors`), it loads it immediately.
2. **Local Exact Match:** Searches all registered roots and their mapped folders for an exact filename match.
3. **Local Stem Match:** Searches for the filename without the extension.
4. **Auto-Download:** If not found locally, it checks the internal `DOWNLOAD_MAP` to see if it can fetch it from Hugging Face.

## Auto-Download Settings

Exiv includes a curated `DOWNLOAD_MAP` containing verified URLs for popular models. If a requested model is missing from your local paths, the engine will attempt to download it automatically.

*   Downloads are saved to the primary root (usually `./models/...`).
*   You can disable auto-downloading entirely via environment variables if you prefer to manage weights manually:

```bash
export auto_download=0
```
If `auto_download` is disabled, the engine will prompt you interactively in the CLI if a missing file is required.

## Custom Output Directory

All generated outputs (images, videos, etc.) are saved to the `output/` directory in the primary root by default. 

If you want to route outputs to a different location, you can override this path programmatically before running your App:

```python
from exiv.utils.file_path import FilePaths

FilePaths.OUTPUT_DIRECTORY = "/absolute/path/to/my/custom_output_folder"
```