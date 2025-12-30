import torch
import unittest
import torch.nn.functional as F
from unittest.mock import patch
from exiv.components.attention import optimized_attention, standard_attention

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

