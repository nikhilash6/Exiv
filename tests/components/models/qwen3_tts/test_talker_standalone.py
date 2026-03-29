import sys
import torch
import unittest

from src.exiv.components.models.qwen3_tts.core.config import Qwen3TTSTalkerConfig, Qwen3TTSTalkerCodePredictorConfig
from src.exiv.components.models.qwen3_tts.core.talker_base import Qwen3TTSTalker


ROPE_SCALING = {
    "mrope_section": [24, 20, 20],
    "rope_type": "default",
    "interleaved": True,
}

CP_KWARGS = dict(
    vocab_size=3072, hidden_size=128, intermediate_size=256, num_hidden_layers=2,
    num_attention_heads=4, num_key_value_heads=2, rope_scaling=ROPE_SCALING, num_code_groups=16
)

TALKER_KWARGS = dict(
    vocab_size=3072, hidden_size=128, intermediate_size=256, num_hidden_layers=2,
    num_attention_heads=4, num_key_value_heads=2, text_hidden_size=128,
    text_vocab_size=152000, pad_token_id=0, rope_scaling=ROPE_SCALING, num_code_groups=16
)


class TestQwen3TTSTalkerStandalone(unittest.TestCase):
    
    def setUp(self):
        self.device = "cpu"
        self.dtype = torch.float32
        torch.manual_seed(42)
        
        # Create configs
        cp_config = Qwen3TTSTalkerCodePredictorConfig(**CP_KWARGS)
        config = Qwen3TTSTalkerConfig(code_predictor_config=cp_config, **TALKER_KWARGS)
        
        # Initialize model
        self.model = Qwen3TTSTalker(config, dtype=self.dtype, device=self.device)
        self.model.to_empty(device=self.device)
        
        # Initialize with deterministic random weights
        for name, param in self.model.named_parameters():
            if param.is_meta:
                param.data = torch.randn_like(param, device=self.device, dtype=self.dtype)
        
        self.model.eval()

    def test_model_structure(self):
        """Test that model has expected structure."""
        self.assertTrue(hasattr(self.model, 'model'))
        self.assertTrue(hasattr(self.model, 'codec_head'))
        self.assertTrue(hasattr(self.model, 'code_predictor'))
        self.assertTrue(hasattr(self.model.model, 'layers'))
        self.assertTrue(hasattr(self.model.model, 'codec_embedding'))
        self.assertTrue(hasattr(self.model.model, 'text_embedding'))
        
    def test_forward_output_shape(self):
        """Test that forward pass produces correct output shape."""
        seq_len = 10
        batch_size = 1
        
        inputs_embeds = torch.randn(batch_size, seq_len, TALKER_KWARGS['hidden_size'], dtype=self.dtype)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
        position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
        
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids
        )
        
        expected_shape = (batch_size, seq_len, TALKER_KWARGS['vocab_size'])
        self.assertEqual(outputs.logits.shape, expected_shape)
        
    def test_forward_runs_without_error(self):
        """Test that forward pass completes without errors."""
        seq_len = 10
        batch_size = 1
        
        inputs_embeds = torch.randn(batch_size, seq_len, TALKER_KWARGS['hidden_size'], dtype=self.dtype)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
        position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
        
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids
        )
        
        # Just check that output was produced with correct shape
        self.assertEqual(outputs.logits.shape, (batch_size, seq_len, TALKER_KWARGS['vocab_size']))
        
    def test_codec_embedding_shape(self):
        """Test that codec_embedding has correct weight shape."""
        embed_shape = self.model.model.codec_embedding.weight.shape
        expected_shape = (TALKER_KWARGS['vocab_size'], TALKER_KWARGS['hidden_size'])
        self.assertEqual(embed_shape, expected_shape)

    def test_num_layers(self):
        """Test that model has correct number of layers."""
        num_layers = len(self.model.model.layers)
        self.assertEqual(num_layers, TALKER_KWARGS['num_hidden_layers'])

    def test_attention_heads(self):
        """Test attention head configuration."""
        first_layer = self.model.model.layers[0]
        self.assertEqual(first_layer.self_attn.config.num_attention_heads, TALKER_KWARGS['num_attention_heads'])
        self.assertEqual(first_layer.self_attn.config.num_key_value_heads, TALKER_KWARGS['num_key_value_heads'])


def run_test():
    """Run tests with verbose output."""
    print("=== Starting Talker Standalone Test ===")
    
    # Run tests
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestQwen3TTSTalkerStandalone)
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
