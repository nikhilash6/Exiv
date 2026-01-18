import torch
from torch import nn

from exiv.utils.device import OFFLOAD_DEVICE

from .hook_registry import HookRegistry, HookType
from ..model_utils.helper_methods import _iter_tensor_attributes

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
    Scans and caches references to all CPU tensors using the shared iterator
    """
    cache = {}
    for key, obj, attr in _iter_tensor_attributes(module):
        val = getattr(obj, attr)
        # only cache if it's currently on CPU
        if val.device.type == OFFLOAD_DEVICE:
            if key not in cache: cache[key] = {}
            cache[key][attr] = val
    return cache

def restore_cpu_state(module: nn.Module, cache: dict):
    """
    Restores cached CPU pointers (Fast Path) or moves leftovers to CPU (Safety Net)
    """
    for key, obj, attr in _iter_tensor_attributes(module):
        # FAST PATH: restore from cache
        if key in cache and attr in cache[key]:
            val = cache[key][attr]
            if attr == "data": 
                obj.data = val
            else: 
                setattr(obj, attr, val)
        # SAFETY NET: force move to CPU if missed by cache
        elif getattr(obj, attr).device.type != OFFLOAD_DEVICE:
            curr = getattr(obj, attr)
            if attr == "data":
                obj.data = curr.to(OFFLOAD_DEVICE)
            else:
                setattr(obj, attr, curr.to(OFFLOAD_DEVICE))