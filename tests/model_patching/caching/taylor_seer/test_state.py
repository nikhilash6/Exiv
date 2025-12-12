import torch

import unittest

from exiv.model_patching.caching.taylor_seer.state import TaylorSeerState

class BasicTaylorSeerStateTest(unittest.TestCase):
    
    def test_warmup_and_skip_logic(self):
        # Warmup=2, Skip=2
        # Steps 0, 1: Warmup (Compute)
        # Step 2: Skip (Approximate)
        # Step 3: Interval (Compute)
        state = TaylorSeerState(max_warmup_steps=2, skip_interval_steps=2)
        
        # Step 0
        state.mark_step_begin()
        assert state.should_compute() is True
        state.update(torch.tensor([0.0]))
        
        # Step 1
        state.mark_step_begin()
        assert state.should_compute() is True
        state.update(torch.tensor([1.0]))
        
        # Step 2
        state.mark_step_begin()
        assert state.should_compute() is False
        # Do not call update here, simulating approximation usage
        
        # Step 3
        state.mark_step_begin()
        assert state.should_compute() is True
        
    def test_linear_approximation(self):
        # If values increase linearly (y=x), 1st derivative is the constant slop, 
        # approximation should be exact.
        state = TaylorSeerState(n_derivatives=1, max_warmup_steps=2)
        
        # Step 0: y=0
        state.mark_step_begin()
        state.update(torch.tensor([0.0]))
        
        # Step 1: y=1 (Derivative calculates here: (1-0)/1 = 1)
        state.mark_step_begin()
        state.update(torch.tensor([1.0]))
        
        # Step 2: Should approximate y=2
        state.mark_step_begin()
        # Force approximation logic validation even if should_compute might be True depending on defaults
        # We manually check the approximation value based on previous state
        pred = state.approximate() 
        # elapsed = 2 - 1 = 1
        # pred = y_prev + y'_prev * 1 = 1.0 + 1.0 * 1 = 2.0
        assert torch.isclose(pred, torch.tensor([2.0]))