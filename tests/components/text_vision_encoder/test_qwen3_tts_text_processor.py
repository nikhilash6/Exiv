import torch
import unittest
from transformers import AutoTokenizer

from src.exiv.components.models.qwen3_tts.core.text_prorcessor import Qwen3TTSTextProcessor

class TestQwen3TextProcessor(unittest.TestCase):
    def setUp(self):
        """Loads the original Qwen3 text tokenizer from HuggingFace and initializes the processor."""
        try:
            # We use the fast tokenizer from the official Qwen3-TTS repository
            self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign")
            self.processor = Qwen3TTSTextProcessor(tokenizer=self.tokenizer)
        except Exception as e:
            self.skipTest(f"Skipping tests because tokenizer could not be loaded: {e}")

    def test_build_assistant_text(self):
        text = "Hello, world!"
        expected = "<|im_start|>assistant\nHello, world!<|im_end|>\n<|im_start|>assistant\n"
        self.assertEqual(self.processor.build_assistant_text(text), expected)

    def test_build_ref_text(self):
        text = "This is a reference."
        expected = "<|im_start|>assistant\nThis is a reference.<|im_end|>\n"
        self.assertEqual(self.processor.build_ref_text(text), expected)

    def test_build_instruct_text(self):
        text = "Speak happily."
        expected = "<|im_start|>user\nSpeak happily.<|im_end|>\n"
        self.assertEqual(self.processor.build_instruct_text(text), expected)

    def test_processor_tokenization(self):
        """
        Tests that the processor correctly delegates text wrapping and tokenization
        to the underlying tokenizer and outputs the expected tensor format.
        """
        text = "Hello, how are you today?"
        
        # 1. Manually format the text as the model expects
        formatted_text = self.processor.build_assistant_text(text)
        
        # 2. Pass it through the processor
        output = self.processor(
            text=[formatted_text], 
            padding=True, 
            return_tensors="pt"
        )
        
        self.assertIn("input_ids", output)
        self.assertIn("attention_mask", output)
        self.assertIsInstance(output["input_ids"], torch.Tensor)
        
        # Assert specific shapes and known token indices for the first few tokens
        self.assertEqual(output["input_ids"].shape, (1, 15))
        
        expected_start_tokens = [151644, 77091, 198, 9707, 11] # <|im_start|>, assistant, \n, Hello, ','
        actual_start_tokens = output["input_ids"][0][:5].tolist()
        
        self.assertEqual(actual_start_tokens, expected_start_tokens, f"Expected start tokens {expected_start_tokens}, got {actual_start_tokens}")

    def test_processor_handles_list_input(self):
        """Tests that the processor can handle a batch of strings."""
        texts = [
            self.processor.build_assistant_text("First sentence."),
            self.processor.build_assistant_text("Second sentence is a little bit longer.")
        ]
        
        output = self.processor(text=texts, padding=True, return_tensors="pt")
        
        self.assertEqual(output["input_ids"].shape[0], 2)
        self.assertGreater(output["input_ids"].shape[1], 5) # The longer sentence determines the padded length

if __name__ == '__main__':
    unittest.main()
