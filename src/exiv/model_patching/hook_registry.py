import torch
from torch import nn, Tensor

from typing import Callable, Any, Dict, List, Optional

from ..utils.enum import ExtendedEnum
from ..utils.logging import app_logger

# this creates a lookup table coupled with a doubly linked list, that has O(1) lookup and update

class HookLocation(ExtendedEnum):
    FORWARD = "forward"                # module forward
    SAMPLER_STEP = "sampler_step"      # around compute_batched_output
    INNER_SAMPLER_STEP = "inner_sampler_step"   # inside compute_batched_output
    MODEL_RUN = "model_run"            # __call__ / the actual model call

class HookType(ExtendedEnum):
    GENERIC = "generic"
    
    # loading hooks
    EFFICIENT_MODEL_LOADER = "efficient_model_loader"
    EFFICIENT_MODULE_LOADER = "efficient_module_loader"
    
    # caching hooks
    TAYLOR_SEER_MODULE_HOOK = "taylor_seer_module_hook"
    TAYLOR_SEER_MODEL_HOOK = "taylor_seer_model_hook"
    
    TAYLOR_SEER_LITE_MODEL_HOOK = "taylor_seer_lite_model_hook"
    
    # sampler level hooks
    SLIDING_CONTEXT = "sliding_context"
    INPAINT_HOOK = "inpaint_hook"
    
    # debug hooks
    NAN_CHECK = "nan_check"

class ModelHook:
    # we are creating a chain of hooks, being applied one after the other
    def __init__(self, *args, **kwargs):
        self.hook_type = HookType.GENERIC.value
        self.hook_location = HookLocation.FORWARD.value
        
        self.next_hook: ModelHook = None
        self.prev_hook: ModelHook = None
        
    def execute(self, module: nn.Module, original_fn: Callable, *args, **kwargs):
        # single wrapper for any location
        return original_fn(*args, **kwargs)


class HookRegistry:
    
    def __init__(self, module_ref: torch.nn.Module) -> None:
        self.hooks_lookup: Dict[str, ModelHook] = {}
        self._module_ref = module_ref
        
        self.head = ModelHook()     # dummy head and tail
        self.tail = ModelHook()
        self.head.next_hook = self.tail
        self.tail.prev_hook = self.head
        
        self._cached_wrappers = {}     # cache by location
        
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
        
        self._insert_hook_after(self.tail.prev_hook, hook)
        self._cached_wrappers = {}

    def get_sorted_hooks(self, location: Optional[str] = None, hook_order: List[str] = None):
        # hook_order defines in what order the hooks must be applied
        # for e.g. the first ele / hook should be applied at the topmost level
        hook_order = hook_order or []
        all_hooks = []
        curr = self.head.next_hook
        while curr != self.tail:
            if location is None or getattr(curr, 'hook_location', None) == location:
                all_hooks.append(curr)
            curr = curr.next_hook

        # 'hook_order' should be ['sliding', 'inpainting'] (desired outermost -> innermost)
        priority_hooks = [h for h in all_hooks if h.hook_type in hook_order]
        rest_hooks = [h for h in all_hooks if h.hook_type not in hook_order]

        # sort priority hooks to match hook_order
        priority_hooks.sort(key=lambda h: hook_order.index(h.hook_type))

        # final list: 'rest' applied first (inner), priority applied last (outer)
        # execution: sliding(inpainting(Rest(model)))
        sorted_hooks = rest_hooks + list(reversed(priority_hooks))
        return sorted_hooks
    
    def get_wrapped_fn(self, original_fn: Callable, location: str, hook_order: List[str] = None) -> Callable:
        cache_key = f"{location}_{id(original_fn)}"
        if cache_key in self._cached_wrappers:
            return self._cached_wrappers[cache_key]
        
        sorted_hooks = self.get_sorted_hooks(location, hook_order)
        
        wrapped_fn = original_fn
        def create_new_wrap(hook, og_fn):
            def new_call(*args, **kwargs):
                return hook.execute(self._module_ref, og_fn, *args, **kwargs)
            return new_call
        
        for hook in sorted_hooks:
            wrapped_fn = create_new_wrap(hook=hook, og_fn=wrapped_fn)
        
        self._cached_wrappers[cache_key] = wrapped_fn
        return wrapped_fn
    
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
        module.forward = registry.get_wrapped_fn(module._original_forward, HookLocation.FORWARD.value)
        
    @staticmethod
    def remove_hook_from_module(module, hook_type: str):
        registry = HookRegistry.get_hook_registry(module)
        registry.remove_hook(hook_type)
        if hasattr(module, "_original_forward"):
            module.forward = registry.get_wrapped_fn(module._original_forward, HookLocation.FORWARD.value)

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