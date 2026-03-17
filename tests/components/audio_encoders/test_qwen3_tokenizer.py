import torch
import unittest
import warnings
from exiv.components.audio_encoders.qwen3_tts.tokenizer_config import Qwen3TTSTokenizerConfig
from exiv.components.audio_encoders.qwen3_tts.tokenizer_base import Qwen3TTSTokenizerModel
from exiv.utils.file_path import FilePaths
from exiv.utils.file import ensure_model_availability
from exiv.config import global_config

warnings.filterwarnings("ignore")


class Qwen3TokenizerTest(unittest.TestCase):
    """Test Qwen3 TTS Tokenizer with pre-trained weights."""
    
    @classmethod
    def setUpClass(cls):
        """Load model with pre-trained weights once for all tests."""
        # Enable auto-download for CI environments
        global_config.auto_download = True
        
        # Encoder config from Qwen3-TTS-Tokenizer-12Hz
        encoder_config = {
            "audio_channels": 1,
            "num_filters": 64,
            "kernel_size": 7,
            "upsampling_ratios": [8, 6, 5, 4],
            "num_residual_layers": 1,
            "dilation_growth_rate": 2,
            "hidden_size": 512,
            "num_hidden_layers": 8,
            "num_attention_heads": 8,
            "num_key_value_heads": 8,
            "num_quantizers": 32,
            "num_semantic_quantizers": 1,
            "codebook_size": 2048,
            "codebook_dim": 256,
            "frame_rate": 12.5,
            "upsample_groups": 512,
            "use_causal_conv": True,
            "pad_mode": "constant",
            "norm_eps": 1e-05,
            "rope_theta": 10000.0,
            "sliding_window": 250,
            "attention_bias": False,
            "attention_dropout": 0.0,
            "hidden_act": "gelu",
            "intermediate_size": 2048,
            "layer_scale_initial_scale": 0.01,
            "max_position_embeddings": 8000,
            "use_conv_shortcut": False,
            "trim_right_ratio": 1.0,
            "vector_quantization_hidden_dimension": 256,
        }
        
        decoder_config = {
            "codebook_size": 2048,
            "codebook_dim": 512,
            "hidden_size": 512,
            "head_dim": 64,
            "latent_dim": 1024,
            "max_position_embeddings": 8000,
            "rope_theta": 10000,
            "num_attention_heads": 16,
            "num_key_value_heads": 16,
            "attention_bias": False,
            "sliding_window": 72,
            "intermediate_size": 1024,
            "hidden_act": "silu",
            "layer_scale_initial_scale": 0.01,
            "rms_norm_eps": 1e-5,
            "num_hidden_layers": 8,
            "num_quantizers": 16,
            "upsample_rates": [8, 5, 4, 3],
            "upsampling_ratios": [2, 2],
            "decoder_dim": 1536,
            "attention_dropout": 0.0,
        }
        
        config = Qwen3TTSTokenizerConfig(
            encoder_config=encoder_config,
            decoder_config=decoder_config,
            encoder_valid_num_quantizers=16,
            input_sample_rate=24000,
            output_sample_rate=24000,
            decode_upsample_rate=1920,
            encode_downsample_rate=1920,
        )
        
        cls.model = Qwen3TTSTokenizerModel(config)
        
        # Load pre-trained weights
        file_data = FilePaths.get_path("qwen3_tts_tokenizer_12hz.safetensors", file_type="audio_encoder")
        weight_path = ensure_model_availability(
            file_data.path if file_data.path else "qwen3_tts_tokenizer_12hz.safetensors",
            download_url=file_data.url
        )
        cls.model.load_model(weight_path)
        cls.model.eval()

    def test_encoder_fidelity(self):
        """Verify encoder produces expected codes for known input."""
        torch.manual_seed(42)
        fake_waveform = torch.ones(1, 24000) * 0.5
        padding_mask = torch.ones(1, 24000, dtype=torch.long)
        
        # Expected values from running original HF code
        expected_shape = [13, 16]
        expected_slice = torch.tensor([1995, 2032, 1094, 456, 912, 633, 210, 1561], dtype=torch.long)
        
        output = self.model.encode(fake_waveform, padding_mask=padding_mask)
        actual_codes = output.audio_codes[0]
        
        self.assertEqual(list(actual_codes.shape), expected_shape)
        torch.testing.assert_close(actual_codes[0, :8], expected_slice)

    def test_decoder_fidelity(self):
        """Verify decoder produces expected audio for known codes."""
        torch.manual_seed(42)
        fake_codes = torch.ones((1, 100, 16), dtype=torch.long) * 42
        
        # Expected values from running original HF code
        expected_shape = [192000]
        expected_slice = torch.tensor(
            [0.02388, 0.023743, 0.003053, -0.010241, -0.004708, -0.001865, -0.008493, -0.015281],
            dtype=torch.float32
        )
        
        output = self.model.decode(fake_codes)
        actual_audio = output.audio_values[0]
        
        self.assertEqual(list(actual_audio.shape), expected_shape)
        torch.testing.assert_close(actual_audio[:8], expected_slice, rtol=1e-4, atol=2e-4)

    def test_model_structure(self):
        """Ensure all expected submodules are present."""
        self.assertTrue(hasattr(self.model, "encoder"))
        self.assertTrue(hasattr(self.model, "decoder"))
        
        # Encoder should have dropped unused decoder parts
        self.assertIsNone(self.model.encoder.upsample)
        self.assertIsNone(self.model.encoder.decoder_transformer)
        self.assertIsNone(self.model.encoder.decoder)


if __name__ == "__main__":
    unittest.main()
