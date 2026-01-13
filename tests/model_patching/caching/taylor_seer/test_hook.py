import torch
import torch.nn as nn
import unittest
from unittest.mock import MagicMock

from exiv.model_patching.caching.taylor_seer.hook import TaylorSeerModuleHook
from exiv.model_patching.hook_registry import HookLocation

class TestTaylorSeerHook(unittest.TestCase):
    def test_hook_switching(self):
        # Warmup=1, Skip=2 (Compute at Step 0, Skip Step 1, Compute Step 2)
        hook = TaylorSeerModuleHook(max_warmup_steps=1, skip_interval_steps=2)
        hook.location = HookLocation.MODEL_RUN # Ensure location is set if used in logic
        
        # Mocking setup
        module = MagicMock()
        module.modulation = torch.zeros(10)
        
        original_fn = MagicMock(return_value=torch.tensor([500.0]))
        
        # Mock _fused_operation to track "optimized" computations
        hook._fused_operation = MagicMock(return_value=torch.tensor([100.0]))
        
        # Mock state.approximate to track approximations
        hook.seer_state.approximate = MagicMock(return_value=torch.tensor([200.0]))

        args = (torch.randn(1, 4, 16, 16), torch.randn(1, 6, 10), torch.randn(1, 10), torch.randn(1, 10, 10))
        kwargs = {}

        # Step 0: Warmup -> Should Compute (via _fused_operation)
        out_0 = hook.execute(module, original_fn, *args, **kwargs)
        self.assertEqual(hook._fused_operation.call_count, 1)
        self.assertTrue(torch.equal(out_0, torch.tensor([100.0])))

        # Step 1: Post-Warmup (Skip Interval) -> Should Approximate
        # Logic: (1 - 1 + 1) % 2 != 0 -> Approximate
        out_1 = hook.execute(module, original_fn, *args, **kwargs)
        self.assertEqual(hook._fused_operation.call_count, 1) # Count shouldn't increase
        self.assertTrue(torch.equal(out_1, torch.tensor([200.0]))) # Returns approximation

        # Step 2: Post-Warmup (Active Interval) -> Should Compute
        # Logic: (2 - 1 + 1) % 2 == 0 -> Compute
        out_2 = hook.execute(module, original_fn, *args, **kwargs)
        self.assertEqual(hook._fused_operation.call_count, 2) # Count increases
        self.assertTrue(torch.equal(out_2, torch.tensor([100.0])))