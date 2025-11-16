import unittest
import torch
import torch.nn as nn

from typing import Callable

from exiv.model_patching.hook_registry import HookRegistry, HookType, ModelHook
from exiv.utils.device import VRAM_DEVICE
from tests.test_utils.common import SimpleModel

class DummyCallHook(ModelHook):
    """
    A simple hook to test the call_wrapper chain.
    It sets a flag when called and adds 100.0 to the output.
    """
    def __init__(self):
        super().__init__()
        self.hook_type = HookType.GENERIC.value

    def call_wrapper(
        self, 
        module: nn.Module, 
        og_call: Callable,
        *args, 
        **kwargs
    ):
        # flag to check this hook was called
        module.called = True
        output = og_call(*args, **kwargs)
        # modify the output
        return output + 100.0
    
def add_debug_hooks(model):
    hook_handle_list = [DummyCallHook]
    
    for hook_cls in hook_handle_list:
        module_hook = hook_cls()
        HookRegistry.apply_hook_to_module(model, module_hook)


class HookTest(unittest.TestCase):
    
    def test_call_hook_wrapper(self):
        model = SimpleModel()
        input_tensor = torch.ones(1, 1024)
        original_output = model(input_tensor)
        
        add_debug_hooks(model)
        hook_output = model(input_tensor)

        # checking if the hook was called
        self.assertEqual(getattr(model, "called", False), True), "The hook's call_wrapper method was never called."
        # checking if the output was correctly modified
        expected_output = original_output + torch.tensor([100.0]).to(VRAM_DEVICE)
        self.assertTrue(
            torch.allclose(hook_output, expected_output),
            f"Output was {hook_output}, but expected {expected_output}."
        )

        