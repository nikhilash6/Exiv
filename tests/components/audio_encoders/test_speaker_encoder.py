import torch
import unittest
import warnings
from exiv.components.models.qwen3_tts.core.configuration_qwen3_tts import Qwen3TTSSpeakerEncoderConfig
from exiv.components.audio_encoders.qwen3_speaker_encoder import Qwen3TTSSpeakerEncoder
from exiv.components.audio_encoders.qwen3_speaker_encoder import mel_spectrogram

warnings.filterwarnings("ignore")

class Qwen3SpeakerEncoderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # We don't load external weights here, we test the deterministic output 
        # of the un-trained architecture to ensure the math/forward pass hasn't broken.
        cls.device = torch.device("cpu")
        config = Qwen3TTSSpeakerEncoderConfig()
        
        # Initialize model with a fixed seed
        torch.manual_seed(42)
        cls.model = Qwen3TTSSpeakerEncoder(config)
        cls.model.eval()
        cls.model.to(cls.device)

    def test_mel_spectrogram_fidelity(self):
        """Verify the DSP function produces the expected Golden Slice."""
        torch.manual_seed(100)
        fake_waveform = torch.randn(1, 24000).squeeze(0).to(self.device)

        # found by running the original code
        expected_shape = [1, 128, 93]
        expected_slice = torch.tensor([
            [0.1646, 0.1607, -0.3979, -1.1671, 0.1068], 
            [0.3424, -0.0730, -0.4444, -0.4062, 0.1785], 
            [-0.4565, 0.0895, 0.0566, -0.6672, 0.2544], 
            [-0.8070, -0.0614, -0.1604, -0.3023, 0.2475], 
            [-0.8387, -0.6024, -0.9617, -0.9294, -0.2026]
        ], dtype=torch.float32)

        mel = mel_spectrogram(
            fake_waveform.unsqueeze(0), 
            n_fft=1024, 
            num_mels=128, 
            sampling_rate=24000,
            hop_size=256, 
            win_size=1024, 
            fmin=0, 
            fmax=12000
        )

        actual_slice = mel[0, :5, :5]
        
        self.assertEqual(list(mel.shape), expected_shape)
        torch.testing.assert_close(actual_slice, expected_slice, rtol=1e-3, atol=1e-3)

    def test_encoder_forward_fidelity(self):
        """Verify the neural network produces the expected embedding Golden Slice."""
        torch.manual_seed(100)
        fake_waveform = torch.randn(1, 24000).squeeze(0).to(self.device)
        
        mel = mel_spectrogram(
            fake_waveform.unsqueeze(0), 
            n_fft=1024, 
            num_mels=128, 
            sampling_rate=24000,
            hop_size=256, 
            win_size=1024, 
            fmin=0, 
            fmax=12000
        )
        input_mel = mel.transpose(1, 2)
        # found by running the original code
        expected_shape = [1, 1024]
        expected_slice = torch.tensor(
            [0.0252, -0.0015, 0.0007, -0.0621, -0.0105, -0.0088, -0.0004, 0.0370, 0.0330, -0.0145],
            dtype=torch.float32
        )

        with torch.no_grad():
            embedding = self.model(input_mel)

        actual_slice = embedding[0, :10]

        self.assertEqual(list(embedding.shape), expected_shape)
        torch.testing.assert_close(actual_slice, expected_slice, rtol=1e-3, atol=1e-3)

    def test_model_structure(self):
        """Ensure all expected submodules are present."""
        self.assertTrue(hasattr(self.model, "blocks"))
        self.assertTrue(hasattr(self.model, "mfa"))
        self.assertTrue(hasattr(self.model, "asp"))
        self.assertTrue(hasattr(self.model, "fc"))

if __name__ == "__main__":
    unittest.main()
