"""
Phase 5: Simplest Happy Flow Test
Just verify text-to-voice works end-to-end without crashing.
"""
import torch
import unittest
import numpy as np
import os

from exiv.components.models.qwen3_tts.constructor import get_qwen3_tts_instance
from exiv.components.models.qwen3_tts.inference.qwen3_tts_model import Qwen3TTSModel
from exiv.config import global_config


def save_audio(wav, sr, filename):
    """Save audio waveform to file."""
    try:
        import soundfile as sf
        sf.write(filename, wav, sr)
        print(f"    Saved audio to: {filename}")
    except ImportError:
        try:
            from scipy.io import wavfile
            # Convert to int16 for WAV format
            wav_int16 = (wav * 32767).astype(np.int16)
            wavfile.write(filename, sr, wav_int16)
            print(f"    Saved audio to: {filename}")
        except ImportError:
            print(f"    Warning: Could not save audio - no soundfile or scipy available")
            # Save as raw numpy array as fallback
            np.save(filename.replace('.wav', '.npy'), wav)
            print(f"    Saved raw numpy array to: {filename.replace('.wav', '.npy')}")


class TestQwen3TTSHappyFlow(unittest.TestCase):
    """Minimal test: Can we generate audio from text?"""
    
    @classmethod
    def setUpClass(cls):
        """Load model once for all tests."""
        print("\n=== Loading Qwen3 TTS Model ===")
        
        # Enable auto-download for testing
        global_config.auto_download = True
        
        # Auto-detect device: CUDA > CPU (MPS disabled due to dtype issues)
        if torch.cuda.is_available():
            cls.device = "cuda"
        else:
            cls.device = "cpu"
        print(f"Using device: {cls.device}")
        
        # Load model (this may download on first run)
        # Using custom voice model for better quality
        raw_model, text_tokenizer, _ = get_qwen3_tts_instance(
            model_path="qwen3_tts_12hz_custom_voice_1_7b.safetensors",
            # MPS works better with float32 for some ops, but fp16 saves memory
            # Use fp16 for CUDA, fp32 for MPS/CPU to avoid potential issues
            force_dtype=torch.float16 if cls.device == "cuda" else torch.float32
        )
        
        cls.model = Qwen3TTSModel(model=raw_model, processor=text_tokenizer)
        cls.model.model.to(cls.device)
        print(f"=== Model loaded on {cls.device} ===")
    
    def test_simple_generation_completes(self):
        """Test that 'Hello world' generation completes without error."""
        text = "Hello world."
        speaker = "Serena"  # Use predefined speaker for custom voice model
        
        print(f"\n>>> Generating: '{text}'")
        
        # This should not raise any exceptions
        wavs, sr = self.model.generate_custom_voice(
            text=text,
            speaker=speaker,
            language="english",
            max_new_tokens=100,  # Keep it short for testing
        )
        
        # Basic sanity checks
        self.assertEqual(len(wavs), 1, "Should return one waveform")
        self.assertIsInstance(wavs[0], np.ndarray, "Waveform should be numpy array")
        self.assertGreater(wavs[0].shape[0], 0, "Waveform should have samples")
        self.assertEqual(sr, 24000, "Sample rate should be 24kHz")
        
        # Save the audio file
        output_dir = os.path.join(os.path.dirname(__file__), "output_audio")
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "test_hello_world.wav")
        save_audio(wavs[0], sr, output_file)
        
        print(f"<<< Generated audio: {wavs[0].shape[0]} samples @ {sr}Hz")
        print(f"    Duration: {wavs[0].shape[0]/sr:.2f}s")
    
    def test_generation_produces_non_silence(self):
        """Verify the output isn't just silence."""
        text = "Hi."
        speaker = "Ryan"  # Male English speaker
        
        wavs, sr = self.model.generate_custom_voice(
            text=text,
            speaker=speaker,
            max_new_tokens=50,
        )
        
        audio = wavs[0]
        
        # Check output has correct properties
        self.assertIsInstance(audio, np.ndarray, "Audio should be numpy array")
        self.assertGreater(audio.shape[0], 0, "Audio should have samples")
        self.assertEqual(sr, 24000, "Sample rate should be 24kHz")
        
        # Save the audio file
        output_dir = os.path.join(os.path.dirname(__file__), "output_audio")
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "test_hi_male.wav")
        save_audio(audio, sr, output_file)
        
        # Check not all zeros (silence)
        self.assertGreater(np.abs(audio).max(), 0.001, "Audio shouldn't be silence")
        # Check has some variation (not constant DC offset)
        self.assertGreater(audio.std(), 0.001, "Audio should have variation")
        
        print(f"    Audio shape: {audio.shape}, sr={sr}Hz")
        print(f"    Audio stats: max={np.abs(audio).max():.4f}, std={audio.std():.4f}")
        print("    Audio has non-silence content!")


def run_quick_test():
    """Run just the basic happy flow test."""
    print("\n" + "="*60)
    print("QWEN3 TTS - SIMPLE HAPPY FLOW TEST")
    print("="*60)
    
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestQwen3TTSHappyFlow)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    if result.wasSuccessful():
        print("\n" + "="*60)
        print("✅ HAPPY FLOW PASSED - Basic TTS works!")
        print("="*60)
        return True
    else:
        print("\n" + "="*60)
        print("❌ HAPPY FLOW FAILED")
        print("="*60)
        return False


if __name__ == "__main__":
    import sys
    success = run_quick_test()
    sys.exit(0 if success else 1)
