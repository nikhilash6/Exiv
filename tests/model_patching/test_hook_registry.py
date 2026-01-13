import unittest
import torch.nn as nn
from exiv.model_patching.hook_registry import HookRegistry, ModelHook, HookLocation

class MockHook(ModelHook):
    def __init__(self, hook_type, location=HookLocation.FORWARD):
        super().__init__()
        self.hook_type = hook_type
        self.hook_location = location

class TestHookSorting(unittest.TestCase):
    def setUp(self):
        self.module = nn.Linear(1, 1)
        self.registry = HookRegistry(self.module)

    def test_get_sorted_hooks_ordering(self):
        """
        Verifies that:
        1. 'Rest' hooks (not in order list) come first (Inner).
        2. Priority hooks come last (Outer), respecting the requested order.
        """
        h_bg1 = MockHook("background_1")
        h_sliding = MockHook("sliding_context")
        h_bg2 = MockHook("background_2")
        h_inpaint = MockHook("inpaint_hook")
        
        # Register in mixed order
        self.registry.register_hook(h_bg1)
        self.registry.register_hook(h_sliding)
        self.registry.register_hook(h_bg2)
        self.registry.register_hook(h_inpaint)
        
        # We want 'sliding' to be the Outermost (Last in list)
        target_order = ["sliding_context", "inpaint_hook"]
        sorted_hooks = self.registry.get_sorted_hooks(hook_order=target_order)
        
        self.assertEqual(len(sorted_hooks), 4)
        
        # 1. Rest hooks (Backgrounds) are first (Inner layers)
        self.assertIn(sorted_hooks[0].hook_type, ["background_1", "background_2"])
        self.assertIn(sorted_hooks[1].hook_type, ["background_1", "background_2"])
        
        # 2. Priority hooks are last (Outer layers)
        # Order in list: [..., Inpaint, Sliding] -> Exec: Sliding(Inpaint(...))
        self.assertEqual(sorted_hooks[2].hook_type, "inpaint_hook")
        self.assertEqual(sorted_hooks[3].hook_type, "sliding_context")

    def test_get_sorted_hooks_missing_and_empty(self):
        """Checks behavior with missing hooks or empty order lists."""
        h1 = MockHook("existing_hook")
        self.registry.register_hook(h1)
        
        # Request non-existent hook -> Should be ignored
        sorted_hooks = self.registry.get_sorted_hooks(hook_order=["non_existent", "existing_hook"])
        self.assertEqual(len(sorted_hooks), 1)
        self.assertEqual(sorted_hooks[0].hook_type, "existing_hook")
        
        # Empty order -> Returns all hooks in default registration order
        sorted_hooks_empty = self.registry.get_sorted_hooks(hook_order=[])
        self.assertEqual(len(sorted_hooks_empty), 1)
        self.assertEqual(sorted_hooks_empty[0], h1)

    def test_get_sorted_hooks_location_filtering(self):
        """Verifies that get_sorted_hooks filters by HookLocation."""
        h_forward = MockHook("forward_hook", location=HookLocation.FORWARD)
        h_sampler = MockHook("sampler_hook", location=HookLocation.SAMPLER_STEP)
        
        self.registry.register_hook(h_forward)
        self.registry.register_hook(h_sampler)
        
        # Filter for FORWARD only
        forward_hooks = self.registry.get_sorted_hooks(location=HookLocation.FORWARD)
        self.assertEqual(len(forward_hooks), 1)
        self.assertEqual(forward_hooks[0].hook_type, "forward_hook")
        
        # Filter for SAMPLER_STEP only
        sampler_hooks = self.registry.get_sorted_hooks(location=HookLocation.SAMPLER_STEP)
        self.assertEqual(len(sampler_hooks), 1)
        self.assertEqual(sampler_hooks[0].hook_type, "sampler_hook")
        
        # No filter -> All hooks
        all_hooks = self.registry.get_sorted_hooks(location=None)
        self.assertEqual(len(all_hooks), 2)