# LoRA Mixin & Hook

The `LoraMixin` provides the internal state and logic for applying Low-Rank Adaptations (LoRAs) to models efficiently. Instead of permanently patching the model's weights and taking up heavy VRAM or RAM footprint, Exiv dynamically reads and applies LoRA weights during the forward pass.

It works closely with `LoraDefinition` to encapsulate scheduling/parameters, and `enable_lora_hook` to interface with the main Model Patching/Hook system.

## The LoRA Flow

The application of a LoRA in Exiv generally follows a four-step lifecycle:

1. **Definition**: A `LoraDefinition` object is created containing the path to the LoRA `safetensors` file, optional download URLs, and a `base_strength` (which can be a constant float or a dynamic list of floats per-timestep).
2. **Registration**: The `enable_lora_hook` entrypoint is called on the target model. This translates to `model.add_lora(lora_def)` under the hood, appending it to the model's internal list of definitions.
3. **Preparation**: Before the inference loop starts, `model.prepare_loras_for_inference(total_steps)` is called. This maps the entire LoRA file into memory (`mmap`) without fully loading it into RAM, standardizes the internal keys, matching them to the model's layers, and caches the references in `self.lora_weights_cache_dict`.
4. **Execution**: During the forward pass, `EfficientModuleLoaderHook` (from the efficient loading system) intercepts layer executions. It fetches the matched LoRA inputs (Up, Down, and Scale weights), calculates the delta via `lora_def.get_output()`, and adds it to the layer's base output seamlessly.

## Key Classes and Methods

### `LoraDefinition`
A dataclass defining a specific LoRA instance.

- **`from_json(data)`**: Class method that securely initializes a LoRA definition from a dictionary or JSON config, ensuring the path/model exists.
- **`expand_strength_schedule(total_steps)`**: Converts a static float strength into a scheduled array corresponding to the `total_steps` of the inference, allowing multi-step variations in LoRA strength.
- **`get_output(input, w_up, w_down, scale, timestep)`**: Calculates the actual delta to add to the base layer. Casts the LoRA up/down weights to the specific `input.device` and `input.dtype`, performs the dot product, scales it based on the alpha calculation, and applies the strength of the current timestep.

### `LoraMixin`
The core mixin inherited by models supporting LoRAs (typically baked into `ModelMixin`).

- **`add_lora(lora_def: LoraDefinition)`**: Appends a new LoRA to the model's registry.
- **`prepare_loras_for_inference(total_steps, device)`**: The core setup function. Expands the schedule for all registered LoRAs, creates the memory maps, iterates through the model's sub-modules, resolves keys, and populates `LORA_WEIGHTS_CACHE_DICT`.
- **`standardize_dict_keys(lora_state_dict_keys)`**: Translates external and custom LoRA keys into standard Exiv prefixes (e.g., standardizing text encoder specific keys).
- **`create_model_lora_key_map(lora_state_dict_keys, model_type)`**: Matches the model's actual layer names (like `blocks.0.cross_attn.k`) with the custom layer names saved inside the LoRA file (like `lora.blocks_0_cross_attn_k.down`), ensuring each layer gets the correct LoRA weights.
- **`get_delta_from_mmap(...)`**: Extracts only the requested `.up` and `.down` byte chunks from the `mmap` object, moving them to the offload device without fully un-pickling the whole safetensor binary.

## How to Add LoRA to a Model

To use LoRA with a compatible Exiv model, you simply define the LoRA, apply the hook, prepare it for inference, and then track the inference steps so the scheduling works out. 

```python
import torch
from exiv.model_patching.lora_hook import enable_lora_hook
from exiv.model_utils.lora_mixin import LoraDefinition

# 1. Initialize your model (assuming it inherits ModelMixin/LoraMixin)
model = MyDiffusionModel()
model.load_model("path/to/base_model.safetensors")

# 2. Define the LoRA parameters
lora_def = LoraDefinition(
    path="path/to/my_lora.safetensors",
    base_strength=0.8 # Can also be a list for dynamic scheduling: [0.0, 0.5, 0.8, 1.0]
)

# 3. Apply the hook to register it onto the model
enable_lora_hook(model, lora_def)

# 4. Prepare it right before inference
total_inference_steps = 20
model.prepare_loras_for_inference(total_steps=total_inference_steps)

# 5. During the inference loop, inform the model of the current step
dummy_input = torch.randn(1, 1024).to(model.gpu_device)

for step in range(total_inference_steps):
    # This property tracks where in the schedule array the definition should look
    model.current_time_step = step 
    
    # Run the model normally. The EfficientModuleLoaderHook automatically 
    # computes the LoRA delta and adds it to the layer's outputs.
    output = model(dummy_input)
```

By leveraging `mmap` and the existing Efficient Loading system, multiple LoRAs can be stacked with dynamically shifting strengths per timestep, and it introduces zero permanent VRAM penalty when offloading is managed.