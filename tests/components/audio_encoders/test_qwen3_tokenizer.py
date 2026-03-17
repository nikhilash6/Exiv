import torch
import unittest

from exiv.components.audio_encoders.qwen3_tts.tokenizer_config import Qwen3TTSTokenizerConfig
from exiv.components.audio_encoders.qwen3_tts.tokenizer_base import Qwen3TTSTokenizerModel
from exiv.utils.file_path import FilePaths
from exiv.utils.file import ensure_model_availability
from exiv.config import global_config


class Qwen3TokenizerTest(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        global_config.auto_download = True
        config = Qwen3TTSTokenizerConfig.from_12hz()
        
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
