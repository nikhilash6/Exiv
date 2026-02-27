# Model Mixin

The core of Exiv relies on a `ModelMixin` architecture to manage model state and utilities.

## What is a Model Mixin?
Mixins provide shared functionalities across various model components (like Text Encoders, Vision Encoders, and VAEs) without enforcing a deep inheritance hierarchy. 

Key functionalities usually provided by `model_mixin.py`:
*   Loading model weights dynamically based on `AppConfig`.
*   Moving tensors between `cpu` and `cuda` efficiently based on the active `loading_mode` (e.g., `no_oom`, `low_vram`).
*   Injecting LoRAs and handling quantizations seamlessly.