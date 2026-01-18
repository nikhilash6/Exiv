import torch
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