import torch

import functools

from typing import Tuple, Any, Dict, Optional

from ..utils.enum import ExtendedEnum

# this creates a lookup table coupled with a doubly linked list, that has O(1) lookup and update

class HookType(ExtendedEnum):
    GENERIC = "generc"
    
    EFFICIENT_MODEL_LOADER = "efficient_model_loader"
    EFFICIENT_MODULE_LOADER = "efficient_module_loader"

class ModelHook:
    # we are creating a chain of hooks, being applied one after the other
    def __init__(self, *args, **kwargs):
        self.hook_type = HookType.GENERIC.value
        self.next_hook: ModelHook = None
        self.prev_hook: ModelHook = None

    def pre_forward(self, module: torch.nn.Module, *args, **kwargs):
        return args, kwargs

    def post_forward(self, module: torch.nn.Module, output: Any):
        return output


class HookRegistry:
    
    def __init__(self, module_ref: torch.nn.Module) -> None:
        self.hooks_lookup: Dict[str, ModelHook] = {}
        self._module_ref = module_ref
        
        self.head = ModelHook()     # dummy head and tail
        self.tail = ModelHook()
        self.head.next_hook = self.tail
        self.tail.prev_hook = self.head

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

    def _add_to_front(self, hook: ModelHook):
        # adds the new hook to the front (will be processed first)
        hook.next_hook = self.head.next_hook
        hook.prev_hook = self.head
        self.head.next_hook.prev_hook = hook
        self.head.next_hook = hook
        
        self.hooks_lookup[hook.hook_type] = hook

    def register_hook(self, hook: ModelHook) -> None:
        if hook.hook_type in self.hooks_lookup.keys():
            raise ValueError(f"{hook.hook_type} already exists")
        
        self._add_to_front(hook)
        self._cached_forward = None
    
    def get_modified_forward(self):
        if self._cached_forward:
            return self._cached_forward
    
        cur_forward = self._module_ref.forward
        cur_hook = self.head.next_hook
        
        def new_forward(hook, og_forward, *args, **kwargs):
            args, kwargs = hook.pre_forward(self._module_ref, *args, **kwargs)
            output = og_forward(*args, **kwargs)
            return hook.post_forward(self._module_ref, output)
        
        while cur_hook != self.tail:
            cur_forward = functools.partial(new_forward, hook=cur_hook, og_forward=cur_forward)
            cur_hook = cur_hook.next_hook
        
        self._cached_forward = cur_forward
        return cur_forward
    
    @classmethod
    def get_hook_registry(cls, module):
        if not hasattr(module, "hook_registry"):
            module.hook_registry = cls(module)
        
        return module.hook_registry
    
    @staticmethod
    def apply_hook_to_module(module, hook):
        registry = HookRegistry.get_hook_registry(module)
        registry.register_hook(hook)

    def get_hook(self, hook_type: str) -> Optional[ModelHook]:
        return self.hooks_lookup.get(hook_type, None)

    def __repr__(self) -> str:
        if not self.hooks_lookup:
            return "HookRegistry(empty)"
        
        parts = []
        for hook_type, hook in self.hooks_lookup.items():
            parts.append(f"  {hook_type} - {hook.__class__.__name__}")
        
        return f"HookRegistry(\n" + "\n".join(parts) + "\n)"