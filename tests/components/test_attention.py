import torch
import unittest
import torch.nn.functional as F
from unittest.mock import patch
from exiv.components.attention import create_attention_mask, optimized_attention, standard_attention

class TestAttentionMock(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)  # Ensure reproducibility
        self.bs, self.seq, self.dim = 2, 16, 64
        self.heads = 4
        self.q = torch.randn(self.bs, self.seq, self.dim)
        self.k = torch.randn(self.bs, self.seq, self.dim)
        self.v = torch.randn(self.bs, self.seq, self.dim)

    def get_reference_output(self):
        """Calculates expected output using standard_attention with manual reshaping."""
        dim_head = self.dim // self.heads
        # Reshape to (bs, heads, seq, dim_head) as expected by standard_attention
        q_in = self.q.view(self.bs, -1, self.heads, dim_head).transpose(1, 2)
        k_in = self.k.view(self.bs, -1, self.heads, dim_head).transpose(1, 2)
        v_in = self.v.view(self.bs, -1, self.heads, dim_head).transpose(1, 2)
        
        out = standard_attention(q_in, k_in, v_in)
        return out.transpose(1, 2).reshape(self.bs, self.seq, self.dim)

    def test_force_standard_attention(self):
        expected = self.get_reference_output()
        
        # Patch available_attn to return standard_attention
        with patch('exiv.components.attention.available_attn', return_value=standard_attention) as mock_method:
            out = optimized_attention(self.q, self.k, self.v, self.heads)
            
            # Verify usage and output correctness
            mock_method.assert_called()
            self.assertTrue(torch.allclose(out, expected, atol=1e-6), "Output mismatch for standard attention")

    def test_force_sdpa(self):
        expected = self.get_reference_output()
        
        # Patch available_attn to return torch's SDPA
        with patch('exiv.components.attention.available_attn', return_value=F.scaled_dot_product_attention):
            out = optimized_attention(self.q, self.k, self.v, self.heads)
            
            # SDPA and standard attention should be numerically close
            self.assertTrue(torch.allclose(out, expected, atol=1e-5), "Output mismatch for SDPA")

class TestAttentionMask(unittest.TestCase):
      """Tests for create_attention_mask function."""

      def test_causal_mask_shape(self):
          """Test that causal mask has correct shape (1, 1, query_len, kv_len)."""
          mask = create_attention_mask(query_len=4, kv_len=4, device=torch.device('cpu'))
          self.assertEqual(mask.shape, (1, 1, 4, 4))

      def test_causal_mask_values(self):
          """Test causal mask: lower triangle should be 0, upper should be -inf."""
          mask = create_attention_mask(query_len=4, kv_len=4, device=torch.device('cpu'))

          # Expected pattern:
          # [0,    -inf, -inf, -inf]
          # [0,    0,    -inf, -inf]
          # [0,    0,    0,    -inf]
          # [0,    0,    0,    0   ]

          expected = torch.tensor([[
              [0.0, float('-inf'), float('-inf'), float('-inf')],
              [0.0, 0.0, float('-inf'), float('-inf')],
              [0.0, 0.0, 0.0, float('-inf')],
              [0.0, 0.0, 0.0, 0.0]
          ]])

          self.assertTrue(torch.allclose(mask[0, 0], expected, equal_nan=False))

      def test_causal_mask_allows_past_only(self):
          """Test that each query position can only attend to previous key positions."""
          mask = create_attention_mask(query_len=5, kv_len=5, device=torch.device('cpu'))

          for q_pos in range(5):
              for k_pos in range(5):
                  value = mask[0, 0, q_pos, k_pos].item()
                  if k_pos <= q_pos:
                      self.assertEqual(value, 0.0, f"Position ({q_pos}, {k_pos}) should be allowed (0)")
                  else:
                      self.assertEqual(value, float('-inf'), f"Position ({q_pos}, {k_pos}) should be masked (-inf)")

      def test_sliding_window_mask(self):
          """Test sliding window: only attend to last N positions."""
          # Window size 2: can only see current and previous 1 position
          # (window=2 means 2 total: current + 1 previous)
          mask = create_attention_mask(query_len=5, kv_len=5, device=torch.device('cpu'), sliding_window=2)

          # Logic: kv_idx <= q_idx (causal) AND kv_idx > q_idx - 2 (window)
          # q=0: kv > -2 AND kv <= 0 -> kv in {0}
          # q=1: kv > -1 AND kv <= 1 -> kv in {0, 1}
          # q=2: kv > 0 AND kv <= 2 -> kv in {1, 2}
          # q=3: kv > 1 AND kv <= 3 -> kv in {2, 3}
          # q=4: kv > 2 AND kv <= 4 -> kv in {3, 4}

          expected = torch.tensor([[
              [0.0, float('-inf'), float('-inf'), float('-inf'), float('-inf')],
              [0.0, 0.0, float('-inf'), float('-inf'), float('-inf')],
              [float('-inf'), 0.0, 0.0, float('-inf'), float('-inf')],
              [float('-inf'), float('-inf'), 0.0, 0.0, float('-inf')],
              [float('-inf'), float('-inf'), float('-inf'), 0.0, 0.0]
          ]])

          self.assertTrue(torch.allclose(mask[0, 0], expected, equal_nan=False))

      def test_sliding_window_smaller_than_query(self):
          """Test sliding window when window is smaller than query length."""
          mask = create_attention_mask(query_len=6, kv_len=6, device=torch.device('cpu'), sliding_window=2)

          # Query position 5: kv > 3 AND kv <= 5 -> kv in {4, 5} (only 2 positions: current + 1 previous)
          self.assertEqual(mask[0, 0, 5, 0].item(), float('-inf'))  # Too far
          self.assertEqual(mask[0, 0, 5, 1].item(), float('-inf'))  # Too far
          self.assertEqual(mask[0, 0, 5, 2].item(), float('-inf'))  # Too far
          self.assertEqual(mask[0, 0, 5, 3].item(), float('-inf'))  # Too far (5-2=3, so kv must be > 3)
          self.assertEqual(mask[0, 0, 5, 4].item(), 0.0)  # Within window (4 > 3 and 4 <= 5)
          self.assertEqual(mask[0, 0, 5, 5].item(), 0.0)  # Current position

      def test_different_query_kv_lengths(self):
          """Test mask creation when query and kv lengths differ (for generation)."""
          # Common in generation: query_len=1 (new token at position N), kv_len=N (cached)
          # When query_len == 1, the implementation treats this as a generation step with
          # KV cache, so the single query token should attend to ALL cached positions.
          mask = create_attention_mask(query_len=1, kv_len=10, device=torch.device('cpu'))

          self.assertEqual(mask.shape, (1, 1, 1, 10))
          # All positions allowed because query_len==1 triggers the KV-cache fast path
          self.assertEqual(mask[0, 0, 0, 0].item(), 0.0)  # Allowed
          self.assertEqual(mask[0, 0, 0, 1].item(), 0.0)  # Allowed (KV cache path)
          self.assertTrue(torch.all(mask == 0.0))

      def test_device_placement(self):
          """Test that mask is created on the correct device."""
          # Skip if CUDA not available
          if not torch.cuda.is_available():
              self.skipTest("CUDA not available")

          mask = create_attention_mask(query_len=4, kv_len=4, device=torch.device('cuda'))
          self.assertEqual(mask.device.type, 'cuda')