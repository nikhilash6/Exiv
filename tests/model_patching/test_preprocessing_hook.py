import torch
import torch.nn as nn

import unittest
from typing import Any, Callable

from exiv.model_patching.hook_registry import HookRegistry, HookType, ModelHook, clear_hook_registry
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


class CallWrapperTest(unittest.TestCase):
    
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

class PreModHook(ModelHook):
    def __init__(self):
        super().__init__()
        self.hook_type = "pre_mod_hook"
    
    def pre_forward(self, module, *args, **kwargs):
        # Add 1.0 to the first argument tensor
        new_args = list(args)
        if isinstance(new_args[0], torch.Tensor):
            new_args[0] = new_args[0] + 1.0
        return tuple(new_args), kwargs

class PostModHook(ModelHook):
    def __init__(self):
        super().__init__()
        self.hook_type = "post_mod_hook"

    def post_forward(self, module, output: Any):
        # Multiply output by 2
        return output * 2.0

class ReplacementHook(ModelHook):
    def __init__(self):
        super().__init__()
        self.hook_type = "replacement_hook"

    def new_forward(self, module, *args, **kwargs):
        # Ignore original model logic
        return torch.full_like(args[0], 5.0)

class ReplacementHook_1(ModelHook):
    def __init__(self):
        super().__init__()
        self.hook_type = "replacement_hook_1"

    def new_forward(self, module, *args, **kwargs):
        # Ignore original model logic
        return torch.full_like(args[0], 21.0)

class ReplacementHook_2(ModelHook):
    def __init__(self):
        super().__init__()
        self.hook_type = "replacement_hook_2"

    def new_forward(self, module, *args, **kwargs):
        # Ignore original model logic
        return torch.full_like(args[0], 31.0)

class HybridHook(ModelHook):
    def __init__(self):
        super().__init__()
        self.hook_type = "hybrid_hook"

    def pre_forward(self, module, *args, **kwargs):
        # Add 10 to input
        new_args = list(args)
        new_args[0] = new_args[0] + 10.0
        return tuple(new_args), kwargs

    def post_forward(self, module, output):
        # Add 10 to output
        return output + 10.0

class HookTest(unittest.TestCase):
    
    def setUp(self):
        self.model = SimpleModel()
        self.input_tensor = torch.ones(1, 1024) # Value: 1.0
        clear_hook_registry(self.model)

    def tearDown(self):
        clear_hook_registry(self.model)

    def test_pre_forward_modification(self):
        # pre hook doesn't have any effect here, will fix it later
        hook = PreModHook()
        original_output = self.model(self.input_tensor)
        HookRegistry.apply_hook_to_module(self.model, hook)
        
        output = self.model(self.input_tensor)
        self.assertTrue(torch.allclose(output, original_output))

    def test_post_forward_modification(self):
        # Logic: Input(1.0) -> Model(Identity) -> 1.0 -> PostHook(*2) -> Output(2.0)
        hook = PostModHook()
        original_output = self.model(self.input_tensor)
        HookRegistry.apply_hook_to_module(self.model, hook)
        
        output = self.model(self.input_tensor)
        self.assertTrue(torch.allclose(output, original_output * 2.0))

    def test_new_forward_replacement(self):
        # Logic: Input(1.0) -> ReplacementHook(Returns 5.0) -> Output(5.0)
        # Original model logic is skipped.
        hook = ReplacementHook()
        HookRegistry.apply_hook_to_module(self.model, hook)
        
        output = self.model(self.input_tensor)
        expected = torch.full_like(self.input_tensor, 5.0).to(VRAM_DEVICE)
        self.assertTrue(torch.allclose(output, expected))

    def test_new_forward_priority_logic(self):
        HookRegistry.apply_hook_to_module(self.model, PostModHook())
        HookRegistry.apply_hook_to_module(self.model, ReplacementHook())
        
        output = self.model(self.input_tensor)
        
        # new_forward -> 5, post_forward -> x2
        expected = torch.full_like(self.input_tensor, 10.0).to(VRAM_DEVICE)
        self.assertTrue(torch.allclose(output, expected))
    
    def test_new_forward_priority_logic_2(self):
        HookRegistry.apply_hook_to_module(self.model, PostModHook())
        HookRegistry.apply_hook_to_module(self.model, ReplacementHook())
        HookRegistry.apply_hook_to_module(self.model, ReplacementHook_1())
        HookRegistry.apply_hook_to_module(self.model, ReplacementHook_2())
        
        output = self.model(self.input_tensor)
        
        # last new_forward -> 31, post_forward -> x2
        expected = torch.full_like(self.input_tensor, 62.0).to(VRAM_DEVICE)
        self.assertTrue(torch.allclose(output, expected))

    def test_cleanup_restores_original(self):
        original_output = self.model(self.input_tensor)
        HookRegistry.apply_hook_to_module(self.model, ReplacementHook())
        clear_hook_registry(self.model)
        
        restored_output = self.model(self.input_tensor)
        self.assertTrue(torch.allclose(original_output, restored_output))