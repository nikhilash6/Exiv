# Quantization

Exiv supports popular quantization methods and it can be easily extended to include new ones.

## Supported Quantizers
The `src/exiv/quantizers/` directory provides multiple methods for reducing model footprint:

1.  **BNB (bitsandbytes)**: 8-bit and 4-bit loading. Great for general-purpose VRAM reduction on consumer hardware.
2.  **FP8 Scaled**: Optimizes weights to use the `float8` format, balancing speed and model quality natively on newer NVIDIA GPUs.

## Base Classes

To implement a new quantizer or understand existing ones, you primarily interact with two abstract base classes defined in `src/exiv/quantizers/base.py`:

- **`QuantizationConfig`**: The base class for defining configuration parameters for a specific quantization method. It enforces type validation on initialization.
- **`Quantizer`**: The base class that houses the actual quantization logic. It requires methods like `process_model_before_weight_loading`, `create_quantized_param`, and `check_quantized_param_shape` to be implemented.

## Defining Configuration

Configuration is typically defined by subclassing `QuantizationConfig` as a dataclass. For example, `BNBQuantizerConfig` defines parameters like `load_in_8bit`, `load_in_4bit`, and `bnb_4bit_quant_type`. 

This configuration can be provided in two ways:
1. **Explicitly**: Passed when instantiating a model (by setting `quant_type` and `quant_config`).
2. **Auto-detected**: The framework uses functions like `detect_quantization_type(model_path)` and `load_quant_config(model_path)` to extract the quantization scheme and its configuration embedded directly within the `safetensors` metadata at runtime.

## Application in Code

Quantization is integrated deeply into the model loading process via the `ModelMixin` class (`src/exiv/model_utils/model_mixin.py`). The application happens in these key stages:

1. **Initialization**: The specific `Quantizer` object is retrieved using `get_quantizer(quant_type, quant_config)`. This happens either early on during the `ModuleMeta` initialization phase or is auto-detected later in the `load_model()` method.
2. **Pre-processing**: Inside the `_set_quantization()` method, *before* any weights are loaded from the file, `quantizer.process_model_before_weight_loading(model)` is called. This step typically iterates through the model's structure and replaces standard layers (e.g., `nn.Linear`) with their quantized counterparts (e.g., `Linear8bitLt` or `Linear4bit`).
3. **Weight Loading**: During `load_model()`, as the `state_dict` is iterated over, the system checks if each parameter belongs to a quantized module. If `quantizer.check_if_quantized_param()` returns true, the framework defers the parameter creation and loading to `quantizer.create_quantized_param()`. This method correctly formats, quantizes (if necessary), and places the weights into the previously pre-processed layers.