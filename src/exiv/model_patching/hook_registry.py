import torch
from torch import nn

import gc
import functools

from typing import Callable, Tuple, Any, Dict, Optional

from ..utils.enum import ExtendedEnum
from ..utils.logging import app_logger

# this creates a lookup table coupled with a doubly linked list, that has O(1) lookup and update

class HookType(ExtendedEnum):
    GENERIC = "generic"
    
    # loading hooks
    EFFICIENT_MODEL_LOADER = "efficient_model_loader"
    EFFICIENT_MODULE_LOADER = "efficient_module_loader"
    
    # caching hooks
    CACHE_STEPS_HOOK = "caching_steps_hook"
    
    # pre-processing hooks
    INPAINT_HOOK = "inpaint_hook"
    
    # debug hooks
    NAN_CHECK = "nan_check"

class ModelHook:
    # we are creating a chain of hooks, being applied one after the other
    def __init__(self, *args, **kwargs):
        self.hook_type = HookType.GENERIC.value
        self.next_hook: ModelHook = None
        self.prev_hook: ModelHook = None
    
    # PONDER: can the structure be improved
    def call_wrapper(self, module: torch.nn.Module, og_call: Callable, *args, **kwargs):
        return og_call(*args, **kwargs)

    def pre_forward(self, module: torch.nn.Module, *args, **kwargs):
        return args, kwargs

    def post_forward(self, module: torch.nn.Module, output: Any):
        return output
    
    # NOTE: this replaces the forward completely and thus needs to be applied
    # first or else it will overwrite all other hooks applied before it
    def new_forward(self, module: torch.nn.Module,*args, **kwargs):
        return module.forward(*args, **kwargs)
        # raise NotImplementedError("Base new_forward should not be called directly.")


class HookRegistry:
    
    def __init__(self, module_ref: torch.nn.Module) -> None:
        self.hooks_lookup: Dict[str, ModelHook] = {}
        self._module_ref = module_ref
        
        self.head = ModelHook()     # dummy head and tail
        self.tail = ModelHook()
        self.head.next_hook = self.tail
        self.tail.prev_hook = self.head
        
    def _has_new_forward(self, hook: ModelHook) -> bool:
        return getattr(hook.__class__, "new_forward", None) is not ModelHook.new_forward

    def remove_hook(self, hook_type: str, recurse: bool = True) -> None:
        hook = self.hooks_lookup.get(hook_type, None)
        if not hook: return
        
        prev_hook_node = hook.prev_hook
        next_hook_node = hook.next_hook
        prev_hook_node.next_hook = next_hook_node
        next_hook_node.prev_hook = prev_hook_node
        
        del self.hooks_lookup[hook_type]
        self._cached_forward = None

        if recurse:
            for module_name, module in self._module_ref.named_modules():
                if module_name == "":
                    continue
                if hasattr(module, "hook_registry"):
                    module.hook_registry.remove_hook(hook_type, recurse=False)

    def _insert_hook_after(self, prev_node: ModelHook, new_hook: ModelHook):
        next_node = prev_node.next_hook
        new_hook.prev_hook = prev_node
        new_hook.next_hook = next_node
        prev_node.next_hook = new_hook
        next_node.prev_hook = new_hook
        self.hooks_lookup[new_hook.hook_type] = new_hook

    def register_hook(self, hook: ModelHook) -> None:
        if hook.hook_type in self.hooks_lookup.keys():
            raise ValueError(f"{hook.hook_type} already exists")
        
        is_nf = self._has_new_forward(hook)
        # hooks with new_forward go to the absolute start
        curr = self.head.next_hook
        last_nf = None
        while curr != self.tail and self._has_new_forward(curr):
            last_nf = curr
            curr = curr.next_hook
        
        if is_nf and last_nf is not None:
            app_logger.warning(f"{hook.hook_type} will overwrite all previous new_forward hooks")
        
        # always applying after the last_nf node (or head)
        apply_after = last_nf or self.head
        self._insert_hook_after(apply_after, hook)
        
        self._cached_forward = None
        self._cached_call = None
    
    def get_modified_forward(self):
        if getattr(self, "_cached_forward", None) is not None:
            return self._cached_forward
    
        cur_forward = self._module_ref._original_forward if \
            getattr(self._module_ref, "_original_forward", None) else self._module_ref.forward
        cur_hook = self.head.next_hook
        
        def create_new_forward(hook, og_forward):
            def new_forward(*args, **kwargs):
                args, kwargs = hook.pre_forward(self._module_ref, *args, **kwargs)
                if self._has_new_forward(hook):
                    output = hook.new_forward(self._module_ref, *args, **kwargs)
                else:
                    output = og_forward(*args, **kwargs)
                return hook.post_forward(self._module_ref, output)

            return new_forward
        
        while cur_hook != self.tail:
            cur_forward = create_new_forward(hook=cur_hook, og_forward=cur_forward)
            cur_hook = cur_hook.next_hook
        
        self._cached_forward = cur_forward
        return cur_forward
    
    # NOTE: hackish sol for now, instead of overwriting __call__ (which can't be done reliably)
    # this is directly called from the ModelMixin's __call__ method
    def get_modified_call(self, og_call):
        if getattr(self, '_cached_call', None):
            return self._cached_call
    
        cur_call = og_call
        cur_hook = self.head.next_hook
        
        def create_new_call(hook, og_call):
            def new_call(*args, **kwargs):
                return hook.call_wrapper(self._module_ref, og_call, *args, **kwargs)
            return new_call
        
        while cur_hook != self.tail:
            cur_call = create_new_call(hook=cur_hook, og_call=cur_call)
            cur_hook = cur_hook.next_hook
        
        self._cached_call = cur_call
        return cur_call
    
    @classmethod
    def get_hook_registry(cls, module):
        if not hasattr(module, "hook_registry"):
            module.hook_registry = cls(module)
        
        return module.hook_registry
    
    @staticmethod
    def apply_hook_to_module(module, hook):
        # saves the original forward method, so it can be reverted later
        if not hasattr(module, "_original_forward"):
            module._original_forward = module.forward
            
        registry = HookRegistry.get_hook_registry(module)
        registry.register_hook(hook)
        module.forward = registry.get_modified_forward()
        
    @staticmethod
    def remove_hook_from_module(module, hook_type: str):
        registry = HookRegistry.get_hook_registry(module)
        registry.remove_hook(hook_type)
        module.forward = registry.get_modified_forward()

    def get_hook(self, hook_type: str) -> Optional[ModelHook]:
        return self.hooks_lookup.get(hook_type, None)

    def __repr__(self) -> str:
        if not self.hooks_lookup:
            return "HookRegistry(empty)"
        
        parts = []
        for hook_type, hook in self.hooks_lookup.items():
            parts.append(f"  {hook_type} - {hook.__class__.__name__}")
        
        return f"HookRegistry(\n" + "\n".join(parts) + "\n)"

def clean_and_restore(module_ref):
    # restores the original forward and deletes the registry
    if hasattr(module_ref, "_original_forward"):
        module_ref.forward = module_ref._original_forward
        del module_ref._original_forward

    if hasattr(module_ref, "hook_registry"):
        del module_ref.hook_registry
            
def clear_hook_registry(model: nn.Module):
    # completely deletes the hook registry and all
    # associated data with it
    if model is None:
        return

    for module in model.modules():
        clean_and_restore(module)