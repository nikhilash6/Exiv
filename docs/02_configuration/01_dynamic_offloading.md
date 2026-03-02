# VRAM Management

Exiv is designed to run large models on consumer hardware by efficiently managing how and when weights are loaded into VRAM. The configuration can either be passed through the CLI arguments or set via environment variables.

## Loading Modes
You can control the VRAM strategy using environment variables or by passing configuration to the App. There are three primary modes (prioritized in this order):

1.  **`no_oom` (No Out Of Memory)**
    *   **Behavior**: Most aggressive offloading. Moves model layers back to CPU RAM immediately after execution.
    *   **Use Case**: For GPUs with very limited VRAM (e.g., 8GB or less), this can be quite slow.
2.  **`low_vram` (Default)**
    *   **Behavior**: Balances speed and VRAM by keeping frequently used components in VRAM while offloading others.
    *   **Use Case**: Standard usage on consumer GPUs (e.g., 8GB - 16GB VRAM).
3.  **`normal_load`**
    *   **Behavior**: Keeps all loaded models in VRAM if possible for maximum inference speed.
    *   **Use Case**: High-end GPUs or server environments with ample VRAM. This can lead to OOM errors if the model exceeds available VRAM.

## Setting the Mode via Environment Variables
```bash
export low_vram=1
export no_oom=0
python apps/your_app.py
```

## Activation Memory Estimation

In `low_vram` and `no_oom` modes, Exiv estimates peak activation memory to decide how many model layers can stay on the GPU simultaneously. This uses a simple formula:

```
peak_activation ≈ num_tokens × max(hidden_dim × attn_factor, ffn_dim × ffn_factor) × dtype_size
```

Each model defines `attn_factor` and `ffn_factor` via `get_memory_footprint_params()`. The `attn_factor` is a tunable, model-specific multiplier that absorbs all per-token activation costs — attention projections, score matrices, cross-attention, modulation, residuals, etc. It is intentionally high (~20–40) because it accounts for more than just attention. Adjust it based on hardware testing: higher values are more conservative (fewer layers on GPU), lower values are more aggressive.

