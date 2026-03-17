"""Standalone test for Qwen3TTSSubtalker - no HF dependencies."""
import sys
import os
import torch
import unittest

# Add kirin to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../")))

from src.exiv.components.models.qwen3_tts.core.config import Qwen3TTSTalkerConfig, Qwen3TTSTalkerCodePredictorConfig
from src.exiv.components.models.qwen3_tts.core.subtalker_base import Qwen3TTSTalkerCodePredictorModelForConditionalGeneration


class TestQwen3TTSSubtalkerStandalone(unittest.TestCase):
    """Standalone test for Subtalker (Code Predictor) model - no HF dependencies."""
    
    def setUp(self):
        """Initialize model with deterministic random weights."""
        self.device = "cpu"
        self.dtype = torch.float32
        torch.manual_seed(42)
        
        self.rope_scaling = {
            "mrope_section": [24, 20, 20],
            "rope_type": "default",
            "interleaved": True,
        }
        
        self.cp_kwargs = dict(
            vocab_size=3072, hidden_size=128, intermediate_size=256, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, rope_scaling=self.rope_scaling, num_code_groups=16
        )
        
        self.talker_kwargs = dict(
            vocab_size=3072, hidden_size=128, intermediate_size=256, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, text_hidden_size=128,
            text_vocab_size=152000, pad_token_id=0, rope_scaling=self.rope_scaling, num_code_groups=16
        )
        
        # Create configs
        cp_config = Qwen3TTSTalkerCodePredictorConfig(**self.cp_kwargs)
        talker_config = Qwen3TTSTalkerConfig(code_predictor_config=cp_config, **self.talker_kwargs)
        
        # Initialize model
        self.model = Qwen3TTSTalkerCodePredictorModelForConditionalGeneration(
            cp_config, talker_config=talker_config, dtype=self.dtype, device=self.device
        )
        self.model.to_empty(device=self.device)
        
        # Initialize with deterministic random weights
        for name, param in self.model.named_parameters():
            if param.is_meta:
                param.data = torch.randn_like(param, device=self.device, dtype=self.dtype)
        
        self.model.eval()

    def test_model_structure(self):
        """Test that model has expected structure."""
        self.assertTrue(hasattr(self.model, 'model'))
        self.assertTrue(hasattr(self.model, 'lm_head'))
        self.assertTrue(hasattr(self.model.model, 'layers'))
        self.assertTrue(hasattr(self.model.model, 'codec_embedding'))
        
    def test_forward_output_shape(self):
        """Test that forward pass produces correct output shape."""
        seq_len = 10
        batch_size = 1
        num_code_groups = self.cp_kwargs['num_code_groups']
        
        inputs_embeds = torch.randn(batch_size, num_code_groups, self.cp_kwargs['hidden_size'], dtype=self.dtype)
        
        outputs = self.model(inputs_embeds=inputs_embeds)
        
        expected_shape = (batch_size, num_code_groups, self.cp_kwargs['vocab_size'])
        self.assertEqual(outputs.logits.shape, expected_shape)
        
    def test_forward_runs_without_error(self):
        """Test that forward pass completes without errors."""
        seq_len = 10
        batch_size = 1
        num_code_groups = self.cp_kwargs['num_code_groups']
        
        inputs_embeds = torch.randn(batch_size, num_code_groups, self.cp_kwargs['hidden_size'], dtype=self.dtype)
        
        outputs = self.model(inputs_embeds=inputs_embeds)
        
        # Just check that output was produced with correct shape
        self.assertEqual(outputs.logits.shape, (batch_size, num_code_groups, self.cp_kwargs['vocab_size']))
        
    def test_codec_embedding_shape(self):
        """Test that codec_embedding has correct weight shape."""
        # codec_embedding is a ModuleList of Embeddings (one per code group)
        self.assertEqual(len(self.model.model.codec_embedding), self.cp_kwargs['num_code_groups'] - 1)
        for embed in self.model.model.codec_embedding:
            expected_shape = (self.cp_kwargs['vocab_size'], self.cp_kwargs['hidden_size'])
            self.assertEqual(embed.weight.shape, expected_shape)

    def test_num_layers(self):
        """Test that model has correct number of layers."""
        num_layers = len(self.model.model.layers)
        self.assertEqual(num_layers, self.cp_kwargs['num_hidden_layers'])

    def test_attention_heads(self):
        """Test attention head configuration."""
        first_layer = self.model.model.layers[0]
        self.assertEqual(first_layer.self_attn.config.num_attention_heads, self.cp_kwargs['num_attention_heads'])
        self.assertEqual(first_layer.self_attn.config.num_key_value_heads, self.cp_kwargs['num_key_value_heads'])


def run_test():
    """Run tests with verbose output."""
    print("=== Starting Subtalker Standalone Test ===")
    
    # Run tests
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestQwen3TTSSubtalkerStandalone)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    if result.wasSuccessful():
        print("\n🚀 All standalone tests passed!")
        return True
    else:
        print("\n⚠️ Some tests failed.")
        return False


if __name__ == "__main__":
    success = run_test()
    sys.exit(0 if success else 1)
