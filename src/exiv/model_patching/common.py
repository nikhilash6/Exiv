import torch
from torch import nn
from .hook_registry import HookRegistry, HookType

# TODO: will generalize this as more hooks are added
def get_effective_shape(model, shape):
    # hooks applied on the model may change the effective shape of 
    # the input going in, affecting the mem calc + other processing (e.g. sliding context hook)
    registry = HookRegistry.get_hook_registry(model)
    if registry and shape:
        if (sliding_ctx_hook:=registry.hooks_lookup.get(HookType.SLIDING_CONTEXT.value, None)) != None:
            # (B, C, T, H, W), clipping sliding context hook num of latent frames
            target_shape_list = list(shape)
            target_shape_list[2] = sliding_ctx_hook.config.ctx_len
            shape = torch.Size(target_shape_list)
            
    return shape

def prepare_and_cache_cpu_state(module: nn.Module):
    """
    Scans and caches references to all CPU tensors.
    """
    cache = {"params": {}, "buffers": {}, "state": {}}
    
    # params
    for name, param in module.named_parameters(recurse=False):
        p_attrs = {"data": param.data}
        # standard attributes
        if hasattr(param, "__dict__"):
            for attr, val in param.__dict__.items():
                if torch.is_tensor(val):
                    p_attrs[attr] = val
        
        # explicitly capture common BNB attributes (in case they aren't in __dict__)
        for attr in ["CB", "SCB", "quant_state"]:
            val = getattr(param, attr, None)
            if val is not None and torch.is_tensor(val):
                p_attrs[attr] = val
        cache["params"][name] = p_attrs
        
    # buffers
    for name, buf in module.named_buffers(recurse=False):
        cache["buffers"][name] = buf.data
        
    # module state (BNB specific)
    if hasattr(module, "state"):
        for attr, val in module.state.__dict__.items():
            if torch.is_tensor(val):
                cache["state"][attr] = val
                
    return cache

def restore_cpu_state(module: nn.Module, cache: dict):
    """
    Restores cached CPU pointers and FORCE moves any remaining GPU tensors to CPU.
    This prevents leaks from lazy initialization in quantization layers.
    """
    # params
    for name, p_attrs in cache["params"].items():
        if hasattr(module, name):
            param = getattr(module, name)
            for attr, val in p_attrs.items():
                if attr == "data":
                    param.data = val
                else:
                    setattr(param, attr, val)
    
    # buffers
    for name, val in cache["buffers"].items():
        if name in module._buffers:
            module._buffers[name].data = val
    
    # state
    if "state" in cache and hasattr(module, "state"):
        for attr, val in cache["state"].items():
            setattr(module.state, attr, val)

    # ----------
    # SAFETY NET: scanning for any leftovers on GPU
    # (e.g., new attributes created during forward pass)
    # params
    for param in module.parameters(recurse=False):
        if param.device.type != "cpu":
            param.data = param.data.to("cpu")
            
        # attributes
        attrs_to_check = list(param.__dict__.keys()) if hasattr(param, "__dict__") else []
        attrs_to_check.extend(["CB", "SCB", "quant_state"])
        
        for attr in attrs_to_check:
            val = getattr(param, attr, None)
            if torch.is_tensor(val) and val.device.type != "cpu":
                setattr(param, attr, val.to("cpu"))

    # buffers
    for buf in module.buffers(recurse=False):
        if buf.device.type != "cpu":
            buf.data = buf.data.to("cpu")

    # state
    if hasattr(module, "state"):
        for attr, val in module.state.__dict__.items():
            if torch.is_tensor(val) and val.device.type != "cpu":
                setattr(module.state, attr, val.to("cpu"))