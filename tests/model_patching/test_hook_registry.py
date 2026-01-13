import unittest
import torch.nn as nn
from src.exiv.model_patching.hook_registry import HookRegistry, ModelHook

class MockHook(ModelHook):
    def __init__(self, hook_type):
        super().__init__()
        self.hook_type = hook_type

class TestHookSorting(unittest.TestCase):
    def setUp(self):
        self.module = nn.Linear(1, 1)
        self.registry = HookRegistry(self.module)

    def test_get_sorted_hooks_ordering(self):
        h_bg1 = MockHook("background_1")
        h_sliding = MockHook("sliding_context")
        h_bg2 = MockHook("background_2")
        h_inpaint = MockHook("inpaint_hook")
        
        self.registry.register_hook(h_bg1)
        self.registry.register_hook(h_sliding)
        self.registry.register_hook(h_bg2)
        self.registry.register_hook(h_inpaint)
        
        target_order = ["sliding_context", "inpaint_hook"]
        sorted_hooks = self.registry.get_sorted_hooks(target_order)
        
        self.assertEqual(len(sorted_hooks), 4)
        self.assertIn(sorted_hooks[0].hook_type, ["background_1", "background_2"])
        self.assertIn(sorted_hooks[1].hook_type, ["background_1", "background_2"])
        self.assertEqual(sorted_hooks[2].hook_type, "inpaint_hook")
        self.assertEqual(sorted_hooks[3].hook_type, "sliding_context")

    def test_get_sorted_hooks_missing_and_empty(self):
        h1 = MockHook("existing_hook")
        self.registry.register_hook(h1)
        
        sorted_hooks = self.registry.get_sorted_hooks(["non_existent", "existing_hook"])
        self.assertEqual(len(sorted_hooks), 1)
        self.assertEqual(sorted_hooks[0].hook_type, "existing_hook")
        
        sorted_hooks_empty = self.registry.get_sorted_hooks([])
        self.assertEqual(len(sorted_hooks_empty), 1)
        self.assertEqual(sorted_hooks_empty[0], h1)