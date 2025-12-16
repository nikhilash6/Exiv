import torch
from torch import nn

import os
from typing import Union, Optional
import safetensors

from ..config import global_config
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

def move_immediate_params(module, device, non_blocking=False):
    """
    Moves only the immediate parameters of the module
    """
    for param in module.parameters(recurse=False):
        if param is not None:
            param.data = param.data.to(device, non_blocking=non_blocking)
            if param.grad is not None:
                param.grad.data = param.grad.data.to(device, non_blocking=non_blocking)

    for name, buf in module.named_buffers(recurse=False):
        if buf is not None:
            module._buffers[name] = buf.to(device, non_blocking=non_blocking)

c = 0
# TODO: dtype and non_blocking params are not handled as of now
def move_module(model, module, module_name, target_device=None):
    """
    This contains the centralized logic for moving different module types 
    between devices
    """
    if module is None: return   # m_ref can turn out to be None
    
    global c
    app_logger.debug(f"Loading the current module: {c}")
    c += 1
    
    target_device = target_device or model.gpu_device
    app_logger.debug(f"Moving {module_name} to {target_device}")
    
    module_class_name = module.__class__.__name__
    is_bnb_module = module_class_name in ["Linear8bitLt", "Linear4bit"]

    if any(p.device.type == "meta" for p in module.parameters(recurse=True)):
        module.to_empty(device=target_device)
    
    elif is_bnb_module:
        device_index = torch.device(target_device).index
        if device_index is None:
             device_index = torch.cuda.current_device() # Get default index if "cuda"
        
        # .cuda(device_index) / to is overridden by bnb
        module.to(target_device)
        
        # handling the movement of linear8bit
        # after the first forward, quant weights are stored in the state
        if hasattr(module, "weight") and getattr(module.weight, "CB", None) is not None:
            module.weight.CB = module.weight.CB.to(target_device)
            
        if hasattr(module, "weight") and getattr(module.weight, "SCB", None) is not None:
            module.weight.SCB = module.weight.SCB.to(target_device)
        
        if hasattr(module, "state") and getattr(module.state, "CB", None) is not None:
            module.state.CB = module.state.CB.to(target_device)
            
        if hasattr(module, "state") and getattr(module.state, "SCB", None) is not None:
            module.state.SCB = module.state.SCB.to(target_device)

        
    else:
        # standard .to() for all other regular modules
        module.to(device=target_device)
        
    move_immediate_params(module, target_device)

    # app_logger.debug(f"modules rn: {[m.__class__.__name__ for mn, m in model.named_modules() if m != model]}")
    # MemoryManager.clear_memory()


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

# code adapted from ComfyUI
def get_state_dict(model_path, device=torch.device("cpu")):
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
            
    return sd