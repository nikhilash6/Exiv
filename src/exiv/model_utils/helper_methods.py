import torch
from torch import nn

import os
from typing import Union, Optional
import safetensors

from ..config import BYTES_IN_MB, global_config
from ..utils.logging import app_logger
from ..utils.device import is_same_device

def move_model(model, device):
    # handling device movement through our custom logic
    for name, module in model.named_modules():
        move_module(
            model,
            module,
            name, 
            target_device=device,
        )

    return model


def _iter_tensor_attributes(module: nn.Module):
    """
    Centralized logic to yield all tensors (params, buffers, attributes, state) 
    associated with a module
    Yields: (key_name, parent_object, attribute_name)
    """
    # params and their attributes
    for name, param in module.named_parameters(recurse=False):
        yield name, param, "data"
        
        # __dict__ for generic tensors
        if hasattr(param, "__dict__"):
            for attr, val in param.__dict__.items():
                if torch.is_tensor(val): yield name, param, attr
        
        # explicit BNB/Quant attributes (if not in __dict__)
        for attr in ["CB", "SCB", "quant_state", "absmax"]:
            val = getattr(param, attr, None)
            if val is not None and torch.is_tensor(val): yield name, param, attr
            
    # buffers
    for name, buf in module.named_buffers(recurse=False):
        yield name, buf, "data"
        
    # module state (BNB specific)
    if hasattr(module, "state"):
        for attr, val in module.state.__dict__.items():
            if torch.is_tensor(val): yield "state", module.state, attr

def move_immediate_params(module, device, non_blocking=False):
    """
    Moves all immediate tensors to device, handles BNB/Quant attributes automatically
    """
    for _, obj, attr in _iter_tensor_attributes(module):
        val = getattr(obj, attr)
        if val.device != torch.device(device):
            # setattr for attributes, direct assignment for .data
            if attr == "data" and isinstance(obj, (nn.Parameter, torch.Tensor)):
                obj.data = val.to(device, non_blocking=non_blocking)
                # not needed rn but keeping for completeness
                if isinstance(obj, nn.Parameter) and obj.grad is not None:
                     obj.grad.data = obj.grad.data.to(device, non_blocking=non_blocking)
            else:
                setattr(obj, attr, val.to(device, non_blocking=non_blocking))

c = 0
# TODO: dtype and non_blocking params are not handled as of now
def move_module(model, module, module_name, target_device=None):
    if module is None: return
    
    global c
    app_logger.debug(f"Loading the current module: {c}")
    c += 1
    
    target_device = target_device or model.gpu_device
    app_logger.debug(f"Moving {module_name} to {target_device}")
    
    if any(p.device.type == "meta" for p in module.parameters(recurse=True)):
        module.to_empty(device=target_device)
    else:
        # call .to() for children/submodules if strictly necessary, 
        # but for leaf modules, move_immediate_params does the heavy lifting
        module.to(target_device)

    # handles CB, SCB, state, etc. automatically
    move_immediate_params(module, target_device)

# lots of checks that can be skipped
def set_module_tensor_to_device(
    module: nn.Module,
    tensor_name: str,
    device: Union[int, str, torch.device],
    value: Optional[torch.Tensor] = None,
    dtype: Optional[Union[str, torch.dtype]] = None,
    non_blocking: bool = False,
):
    # traverse the nested modules using the '.' in the tensor name (e.g., 'encoder.layer.0.weight')
    if "." in tensor_name:
        splits = tensor_name.split(".")
        for split in splits[:-1]:
            new_module = getattr(module, split)
            if new_module is None:
                raise ValueError(f"{module} has no attribute {split}.")
            module = new_module
        tensor_name = splits[-1]

    # ensure the tensor name corresponds to an existing parameter or buffer
    if tensor_name not in module._parameters and tensor_name not in module._buffers:
        raise ValueError(f"{module} does not have a parameter or a buffer named {tensor_name}.")
    is_buffer = tensor_name in module._buffers
    old_value = getattr(module, tensor_name)

    if old_value.device == torch.device("meta") and device not in ["meta", torch.device("meta")] and value is None:
        raise ValueError(f"{tensor_name} is on the meta device, we need a `value` to put in on {device}.")

    param = module._parameters[tensor_name] if tensor_name in module._parameters else None
    param_cls = type(param)
    if value is not None:
        if dtype is None:
            # For compatibility with PyTorch load_state_dict which converts state dict dtype to existing dtype in model
            value = value.to(old_value.dtype, non_blocking=non_blocking)
        elif not str(value.dtype).startswith(("torch.uint", "torch.int", "torch.bool")):
            value = value.to(dtype, non_blocking=non_blocking)

    device_quantization = None
    with torch.no_grad():
        # temporarily set the device to 'cpu' to handle quantization correctly before the final move.
        # if it's currently not on gpu then it needs to processed first before moving to gpu (if its the target)
        if (
            param is not None
            and param.device.type not in ("cuda", "xpu")
            and torch.device(device).type in ("cuda", "xpu")
            and param_cls.__name__ in ["Int8Params", "FP4Params", "Params4bit"]
        ):
            device_quantization = device 
            device = "cpu"

        if isinstance(value, torch.Tensor):
            new_value = value.to(device, non_blocking=non_blocking)
        else:
            new_value = torch.tensor(value, device=device)

        # revert the target device to the original GPU
        if device_quantization is not None:
            device = device_quantization

        # --- final assignment
        # simple assignment for the buffer
        if is_buffer:
            module._buffers[tensor_name] = new_value
        
        # update if a new value was provided OR if the device actually changed.
        elif value is not None or not is_same_device(torch.device(device), module._parameters[tensor_name].device):
            param_cls = type(module._parameters[tensor_name])
            kwargs = module._parameters[tensor_name].__dict__
            
            # special handling for low-precision/quantized parameter classes (e.g., bitsandbytes)
            if param_cls.__name__ in ["Int8Params", "FP4Params", "Params4bit"]:
                # cast to fp16 for 8-bit serialization/compatibility if needed
                if param_cls.__name__ == "Int8Params" and new_value.dtype == torch.float32:
                    new_value = new_value.to(torch.float16, non_blocking=non_blocking)
                
                # quantize the weights on the GPU first then move to the CPU
                if device == "cpu" and param_cls.__name__ == "Int8Params":
                    new_value = param_cls(new_value, requires_grad=old_value.requires_grad, **kwargs).to(0).to("cpu")
                    new_value.CB = new_value.CB.to("cpu")
                    new_value.SCB = new_value.SCB.to("cpu")
                else:
                    # re-wrap the tensor using its specialized class and move to the final device
                    new_value = param_cls(new_value, requires_grad=old_value.requires_grad, **kwargs).to(
                        device, non_blocking=non_blocking
                    )

            # other known quantized tensor types (affine one is from torchao)
            elif param_cls.__name__ in ["QTensor", "QBitsTensor", "AffineQuantizedTensor"]:
                new_value = torch.nn.Parameter(new_value, requires_grad=old_value.requires_grad).to(
                    device, non_blocking=non_blocking
                )
            
            # default
            else:
                new_value = param_cls(new_value, requires_grad=old_value.requires_grad).to(
                    device, non_blocking=non_blocking
                )

            module._parameters[tensor_name] = new_value

    # freeing old_value (safety check)
    # MemoryManager.clear_memory()

# TODO: merge with the code inside LoraMixin
def clean_state_dict(state_dict, model_type=None):
    if not model_type: return state_dict
    if model_type == "checkpoint":
        prefixes_to_strip = ["model.", "diffusion_model."]
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key
            for prefix in prefixes_to_strip:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
            new_state_dict[new_key] = value
        state_dict = new_state_dict

    return state_dict

# TODO: replace this with custom loading logic (very buggy on windows)
# code adapted from ComfyUI
def get_state_dict(model_path, model_type=None, device=torch.device("cpu")):
    if isinstance(device, str):
        device = torch.device(device)
    
    file_extension = os.path.basename(model_path).split(".")[-1]
    if file_extension in ["safetensors", "sft"]:
        try:
            # safetensor's zero copy loading (pt - pytorch)
            kwargs = {"framework": "pt"}
            # safetensors only support cpu and cuda, and doesn't take cpu as param
            if device.type == "cuda": kwargs["device"] = device.type
            with safetensors.safe_open(model_path, **kwargs) as f:
                sd = {}
                for k in f.keys():
                    tensor = f.get_tensor(k)    # loading one key at a time; low mem pressure
                    if global_config.disable_mmap:
                        # moving to device (no zero copying)
                        tensor = tensor.to(device=device, copy=True)
                    sd[k] = tensor
        except Exception as e:
            app_logger.error(str(e))
            raise e
    else:   # ckpt, pth, pt
        torch_args = {}
        # using simple flags rn, will fix later
        if not global_config.disable_mmap: torch_args["mmap"] = True
        if global_config.always_safe_load: torch_args["weights_only"] = True
        
        sd = torch.load(model_path, map_location=device, **torch_args)
        if "state_dict" in sd:  
            sd = sd["state_dict"]   # loading state_dict if available
        elif len(sd) == 1:          # loading the first key (if it's a dict)
            val = next(iter(sd.values()))
            sd = val if isinstance(val, dict) else sd
    
    sd = clean_state_dict(sd, model_type)
    return sd

# TODO: create a proper dataclass for the memory_config
def estimate_peak_activation_size(memory_footprint_config = None, target_shape = None):
    """
    A rough estimation of what peak mem use could be. Mostly focused
    on attn + rope part.
    """
    default_mem_estimate = 0
    if not target_shape: return default_mem_estimate

    if memory_footprint_config is not None:
        # NOTE: update this normalization logic as more models are added
        # ------------------------------
        safe_shape = list(target_shape)
        # image (B, C, H, W) -> insert time=1 -> (B, C, 1, H, W)
        if len(safe_shape) == 4:
            safe_shape.insert(2, 1)
        # text/latents (B, L, D) -> pad spatial dims with 1 -> (B, L, D, 1, 1)
        while len(safe_shape) < 5:
            safe_shape.append(1)
        target_shape = safe_shape
        # --------------------------------
        
        params = memory_footprint_config
        if not params: return default_mem_estimate
        
        # target_shape is (B, C, T, H, W)
        t, h, w = target_shape[2], target_shape[3], target_shape[4]
        patch_t, patch_h, patch_w = params.get("patch_size", (1, 1, 1))
        
        # round up dimensions to nearest patch multiple (padding logic)
        t_tokens = (t + patch_t - 1) // patch_t
        h_tokens = (h + patch_h - 1) // patch_h
        w_tokens = (w + patch_w - 1) // patch_w
        # total tokens / chunk the transformer processes at a time
        num_tokens = target_shape[0] * t_tokens * h_tokens * w_tokens
        
        # attn calculation
        # q, k, v, o, modulation vectors, skip connections
        attn_peak = params["hidden_dim"] * params.get("attn_factor", 2.5)
        ffn_peak  = params["ffn_dim"]    * params.get("ffn_factor", 1.0)
        peak_width = max(attn_peak, ffn_peak)
        
        dtype_size = params.get("dtype_size", 2)
        # basically - (how many tokens) * (peak floats per token) * (bytes per float)
        total_bytes = num_tokens * peak_width * dtype_size
        
        return total_bytes / BYTES_IN_MB
    else:
        return default_mem_estimate

# PONDER: is this better placed inside the ModelMixin
def get_mem_usage(model, shape):
    # gives the approximate mem usage of running a particular model
    # with a given input shape
    from ..model_patching.common import get_effective_shape
    from ..model_patching.efficient_loading_hook import split_model_for_loading
    
    effective_shape = get_effective_shape(model, shape)
    _, loaded_model_mem = split_model_for_loading(model, effective_shape)
    return loaded_model_mem