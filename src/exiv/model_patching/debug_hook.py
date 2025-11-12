import torch
from torch import Tensor

from typing import Any

from .hook_registry import HookRegistry, HookType
from ..utils.logging import app_logger

class NANCheckHook:
    def __init__(self):
        super().__init__()
        self.hook_type = HookType.NAN_CHECK.value
        
    def pre_forward(self, module: torch.nn.Module, *args, **kwargs):
        return args, kwargs
    
    def post_forward(self, module: torch.nn.Module, output: Any):
        outputs_to_check = []
        if isinstance(output, Tensor):
            outputs_to_check.append(output)
        elif isinstance(output, (list, tuple)):
            for o in output:
                if isinstance(o, Tensor):
                    outputs_to_check.append(o)

        for i, out in enumerate(outputs_to_check):
            if torch.isnan(out).any():
                app_logger.error(f"!!! NaN detected in output {i} of: {module.__class__.__name__}")
                # import pdb; pdb.set_trace()
                # raise RuntimeError(f"NaNs detected in {module.__class__.__name__}")
            
        return output

# NOTE: this used mainly for test / dev
def add_debug_hooks(model):
    hook_handle_list = [NANCheckHook]
    
    for m_name, module in model.named_modules():
        if module is model or not model.is_leaf_module(module):
            continue
        
        for hook_cls in hook_handle_list:
            module_hook = hook_cls()
            HookRegistry.apply_hook_to_module(module, module_hook)