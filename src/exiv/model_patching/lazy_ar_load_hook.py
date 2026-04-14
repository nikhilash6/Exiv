import torch
from torch import nn
from typing import Callable

from .hook_registry import HookRegistry, HookLocation, HookType, ModelHook

class LazyARLoadHook(ModelHook):
    """
    Hook that lazily moves an AR model to its target device on first execution.
    """

    hook_type = HookType.LAZY_AR_LOADER.value
    hook_location = HookLocation.AR_GENERATE.value

    def __init__(self):
        super().__init__()
        self.hook_type = HookType.LAZY_AR_LOADER.value
        self.hook_location = HookLocation.AR_GENERATE.value

    @classmethod
    def lazy_load(cls, model: nn.Module):
        if not getattr(model, "_lazy_loaded", False):
            target_device = getattr(model, "gpu_device", None)
            if target_device is not None:
                model.to(target_device)
            model._lazy_loaded = True

    def execute(self, model: nn.Module, original_fn: Callable, *args, **kwargs):
        self.lazy_load(model)
        return original_fn(*args, **kwargs)

def enable_lazy_ar_loading(model: nn.Module):
    # TODO: update the logic here
    registry = getattr(model, "hook_registry", None)
    if registry and registry.get_hook(HookType.LAZY_AR_LOADER.value):
        return

    lazy_hook = LazyARLoadHook()
    HookRegistry.apply_hook_to_module(
        model, lazy_hook, method_name="generate"
    )

def remove_lazy_ar_loading(model: nn.Module):
    """Remove the lazy loading hook and restore original methods."""
    HookRegistry.remove_hook_from_module(
        model, HookType.LAZY_AR_LOADER.value, method_name="generate"
    )
    if hasattr(model, "_lazy_loaded"):
        delattr(model, "_lazy_loaded")
