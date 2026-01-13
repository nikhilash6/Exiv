import torch
import torch.nn as nn
import unittest
from typing import Any, Callable

from exiv.model_patching.hook_registry import HookRegistry, HookType, ModelHook, HookLocation, clear_hook_registry
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
        self.location = HookLocation.FORWARD

    def execute(self, module: nn.Module, original_fn: Callable, *args, **kwargs):
        # flag to check this hook was called
        module.called = True
        output = original_fn(*args, **kwargs)
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
        input_tensor = torch.ones(1, 1024).to(VRAM_DEVICE)
        
        original_output = model(input_tensor)
        
        add_debug_hooks(model)
        hook_output = model(input_tensor)

        # checking if the hook was called
        self.assertEqual(getattr(model, "called", False), True), "The hook's execute method was never called."
        
        # checking if the output was correctly modified
        expected_output = original_output + 100.0
        self.assertTrue(
            torch.allclose(hook_output, expected_output),
            f"Output mismatch. \nActual: {hook_output[0,:5]} \nExpected: {expected_output[0,:5]}"
        )

class PreModHook(ModelHook):
    def __init__(self):
        super().__init__()
        self.hook_type = "pre_mod_hook"
        self.location = HookLocation.FORWARD
    
    def execute(self, module, original_fn, *args, **kwargs):
        # Add 1.0 to the first argument tensor
        new_args = list(args)
        if len(new_args) > 0 and isinstance(new_args[0], torch.Tensor):
            new_args[0] = new_args[0] + 1.0
        return original_fn(*tuple(new_args), **kwargs)

class PostModHook(ModelHook):
    def __init__(self):
        super().__init__()
        self.hook_type = "post_mod_hook"
        self.location = HookLocation.FORWARD

    def execute(self, module, original_fn, *args, **kwargs):
        # Multiply output by 2
        output = original_fn(*args, **kwargs)
        return output * 2.0

class ReplacementHook(ModelHook):
    def __init__(self):
        super().__init__()
        self.hook_type = "replacement_hook"
        self.location = HookLocation.FORWARD

    def execute(self, module, original_fn, *args, **kwargs):
        # Ignore original_fn, return constant 5.0 of input shape
        # NOTE: Model output shape is (1, 512), but this returns (1, 1024)
        # We must return the correct shape for the tests to make sense if chained, 
        # but for this specific test we expect 5.0
        return torch.full_like(args[0], 5.0)

class ReplacementHook_1(ModelHook):
    def __init__(self):
        super().__init__()
        self.hook_type = "replacement_hook_1"
        self.location = HookLocation.FORWARD

    def execute(self, module, original_fn, *args, **kwargs):
        return torch.full_like(args[0], 21.0)

class ReplacementHook_2(ModelHook):
    def __init__(self):
        super().__init__()
        self.hook_type = "replacement_hook_2"
        self.location = HookLocation.FORWARD

    def execute(self, module, original_fn, *args, **kwargs):
        return torch.full_like(args[0], 31.0)

class HookTest(unittest.TestCase):
    
    def setUp(self):
        self.model = SimpleModel()
        self.input_tensor = torch.ones(1, 1024).to(VRAM_DEVICE)
        clear_hook_registry(self.model)

    def tearDown(self):
        clear_hook_registry(self.model)

    def test_pre_forward_modification(self):
        # 1. Calculate Expected: Run model manually on (input + 1.0)
        expected_input = self.input_tensor + 1.0
        expected_output = self.model(expected_input)
        
        # 2. Apply Hook
        hook = PreModHook()
        HookRegistry.apply_hook_to_module(self.model, hook)
        
        # 3. Run model on original input (Hook adds +1.0)
        output = self.model(self.input_tensor)
        
        self.assertTrue(torch.allclose(output, expected_output))

    def test_post_forward_modification(self):
        # 1. Calculate Expected: Run model normally, then * 2.0
        original_output = self.model(self.input_tensor)
        expected_output = original_output * 2.0
        
        # 2. Apply Hook
        hook = PostModHook()
        HookRegistry.apply_hook_to_module(self.model, hook)
        
        # 3. Run model
        output = self.model(self.input_tensor)
        
        self.assertTrue(torch.allclose(output, expected_output))

    def test_new_forward_replacement(self):
        # Logic: ReplacementHook(Returns 5.0) -> Output(5.0)
        # Note: This hook returns shape (1, 1024) while model returns (1, 512).
        # This checks that the model was indeed bypassed.
        hook = ReplacementHook()
        HookRegistry.apply_hook_to_module(self.model, hook)
        
        output = self.model(self.input_tensor)
        expected = torch.full_like(self.input_tensor, 5.0).to(VRAM_DEVICE)
        self.assertTrue(torch.allclose(output, expected))

    def test_new_forward_priority_logic(self):
        # Stack: Post (Outer) -> Replacement (Inner)
        # Exec: Post( Replacement( Model ) )
        # 1. Post calls Replacement
        # 2. Replacement returns 5.0 (Model ignored)
        # 3. Post multiplies 5.0 * 2.0 = 10.0
        HookRegistry.apply_hook_to_module(self.model, ReplacementHook())
        HookRegistry.apply_hook_to_module(self.model, PostModHook())
        
        output = self.model(self.input_tensor)
        expected = torch.full_like(self.input_tensor, 10.0).to(VRAM_DEVICE)
        self.assertTrue(torch.allclose(output, expected))
    
    def test_new_forward_priority_logic_2(self):
        # Stack: Post (Outer) -> Rep2 -> Rep1 -> Rep (Inner)
        # Exec: Post( Rep2( ... ) )
        # Rep2 returns 31.0 (swallows Rep1, Rep, Model)
        # Post multiplies 31.0 * 2.0 = 62.0
        HookRegistry.apply_hook_to_module(self.model, ReplacementHook())
        HookRegistry.apply_hook_to_module(self.model, ReplacementHook_1())
        HookRegistry.apply_hook_to_module(self.model, ReplacementHook_2())
        HookRegistry.apply_hook_to_module(self.model, PostModHook())
        
        output = self.model(self.input_tensor)
        expected = torch.full_like(self.input_tensor, 62.0).to(VRAM_DEVICE)
        self.assertTrue(torch.allclose(output, expected))

    def test_cleanup_restores_original(self):
        original_output = self.model(self.input_tensor)
        HookRegistry.apply_hook_to_module(self.model, ReplacementHook())
        clear_hook_registry(self.model)
        
        restored_output = self.model(self.input_tensor)
        self.assertTrue(torch.allclose(original_output, restored_output))