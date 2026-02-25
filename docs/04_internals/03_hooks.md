# Hooks

The `model_patching` system in Exiv heavily relies on Hooks to intercept and alter model behavior during execution. Most dynamic logic that alters the output (or anything else that is not a conditioning) can be implemented through hooks.

## The Hook Registry
Hooks are managed by the `hook_registry.py`. These hooks can be injected at different points in the inference process and even on a granular level (e.g., per layer or per timestep). Some of the pre-built hooks include:
*   Dynamic offloading hook - manages memory by offloading and reloading model components as needed
*   Caching hook - caches intermediate results to speed up repeated inferences
*   Sliding Contexts - extends the context window by stitching together multiple forward passes
*   Debugging states - provides insights into the internal states of the model for debugging and visualization purposes

## Why Hooks?
By using hooks instead of hardcoding changes into the model's forward pass, you can easily turn features like `sliding_context` on or off without altering the base model definition.

> **Note:** Please note that custom third-party hooks are not yet supported but will be soon.
