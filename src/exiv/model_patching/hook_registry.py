import torch

import functools

from typing import Tuple, Any, Dict, Optional

# this creates a lookup table coupled with a doubly linked list, that has O(1) lookup and update

class ModelHook:
    # we are creating a chain of hooks, being applied one after the other
    def __init__(self):
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


    def remove_hook(self, name: str, recurse: bool = True) -> None:
        hook = self.hooks_lookup.get(name, None)
        if not hook: return
        
        prev_hook_node = hook.prev_hook
        next_hook_node = hook.next_hook
        prev_hook_node.next_hook = next_hook_node
        next_hook_node.prev_hook = prev_hook_node
        
        del self.hooks_lookup[name]

        if recurse:
            for module_name, module in self._module_ref.named_modules():
                if module_name == "":
                    continue
                if hasattr(module, "hook_registry"):
                    module.hook_registry.remove_hook(name, recurse=False)


    def _add_to_front(self, hook: ModelHook, name: str):
        # adds the new hook to the front (will be processed first)
        hook.next_hook = self.head.next_hook
        hook.prev_hook = self.head
        self.head.next_hook.prev_hook = hook
        self.head.next_hook = hook
        
        self.hooks_lookup[name] = hook


    def register_hook(self, hook: ModelHook, name: str) -> None:
        if name in self.hooks_lookup.keys():
            raise ValueError(f"{name} already exists")
        
        self._add_to_front(hook, name)

    
    def get_modified_forward(self):
        cur_forward = self._module_ref.forward
        cur_hook = self.head.next_hook
        
        def new_forward(hook, og_forward, *args, **kwargs):
            args, kwargs = hook.pre_forward(self._module_ref, *args, **kwargs)
            output = og_forward(*args, **kwargs)
            return hook.post_forward(self._module_ref, output)
        
        while cur_hook != self.tail:
            cur_forward = functools.partial(new_forward, hook=cur_hook, og_forward=cur_forward)
            cur_hook = cur_hook.next_hook
            
        return cur_forward


    def get_hook(self, name: str) -> Optional[ModelHook]:
        return self.hooks_lookup.get(name, None)


    def __repr__(self) -> str:
        if not self.hooks_lookup:
            return "HookRegistry(empty)"
        
        parts = []
        for name, hook in self.hooks_lookup.items():
            parts.append(f"  {name} - {hook.__class__.__name__}")
        
        return f"HookRegistry(\n" + "\n".join(parts) + "\n)"
