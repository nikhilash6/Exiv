"""
Integration test for Qwen3-TTS full pipeline with three-sentence chunking.

This test verifies that:
1. Text is correctly split into 3 sentence chunks
2. Each chunk produces the expected input_ids
3. Model generation produces deterministic talker codes
4. Audio decoding produces deterministic output

The test uses a fixed torch.manual_seed(42) and lowered temperature
to ensure deterministic, reproducible results.
"""

import os
import sys
import unittest
import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
_APPS_QWEN3_TTS = os.path.join(_REPO_ROOT, "apps", "qwen3_tts")
_SRC_DIR = os.path.join(_REPO_ROOT, "src")

sys.path.insert(0, _APPS_QWEN3_TTS)
sys.path.insert(0, _SRC_DIR)

from exiv.utils.device import MemoryManager
from exiv.components.models.qwen3_tts.utils.inference_utils import (
    DEFAULT_QWEN3_CONFIG,
    get_voice_ref,
    tokenizer_decode,
)
from exiv.components.models.qwen3_tts import VoiceClonePromptItem
from exiv.components.models.qwen3_tts.constructor import get_qwen3_tts_instance
from exiv.utils.text_chunking import chunk_text_by_sentences
from wav_manager import get_manager as get_wav_manager


TEST_TEXT = (
    "The sun was setting over the mountains. "
    "Birds were flying back to their nests. "
    "It was a peaceful evening in the countryside."
)

# expected first few values captured with torch.manual_seed(42), temperature=0.1
def _chunk_input_ids_by_sentences(text, processor, max_token_chunk_size=30000, device=None):
    """Split text into sentence chunks and convert each to input_ids."""
    text_chunks = chunk_text_by_sentences(text, max_token_chunk_size)
    chunk_input_ids_list = []
    for chunk_text in text_chunks:
        assistant_text = processor.build_assistant_text(chunk_text)
        input_ids = processor(text=assistant_text, return_tensors="pt", padding=True)
        chunk_tokens = input_ids["input_ids"]
        chunk_tokens = chunk_tokens if chunk_tokens.dim() > 1 else chunk_tokens.unsqueeze(0)
        if device is not None:
            chunk_tokens = chunk_tokens.to(device)
        chunk_input_ids_list.append(chunk_tokens)
    return chunk_input_ids_list


EXPECTED_INPUT_IDS_FIRST_5 = [
    [151644, 77091, 198, 785, 7015],
    [151644, 77091, 198, 65270, 82],
    [151644, 77091, 198, 2132, 572],
]

EXPECTED_TALKER_CODES_FIRST_5 = [
    [1226, 1722, 355, 371, 1296, 1093, 625, 1814, 1511, 1144, 679, 1247, 896, 889, 586, 803],
    [1431, 1722, 355, 371, 1296, 1093, 625, 1814, 1511, 1144, 679, 1247, 896, 889, 1008, 803],
    [898, 1722, 355, 371, 1296, 1093, 625, 1814, 1511, 1144, 679, 1247, 896, 889, 586, 901],
    [1362, 1722, 355, 371, 1296, 1093, 625, 1814, 1511, 1144, 679, 1247, 1110, 889, 586, 901],
    [1687, 1722, 355, 371, 1296, 1093, 625, 1814, 1511, 1144, 679, 1247, 1110, 889, 586, 901],
]

EXPECTED_AUDIO_FIRST_5 = [
    1.043081283569336e-05,
    1.049041748046875e-05,
    1.0669231414794922e-05,
    1.150369644165039e-05,
    1.4066696166992188e-05,
]


class TestFullPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._orig_cwd = os.getcwd()
        os.chdir(_APPS_QWEN3_TTS)

        cls.device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.manual_seed(42)

        model_path = "models/checkpoints/qwen3_tts_12hz_base_1_7b.safetensors"
        cls.model, cls.text_tokenizer, _ = get_qwen3_tts_instance(
            model_path=model_path,
            force_dtype=torch.float16 if cls.device == "cuda" else torch.float32,
        )

        cls.chunked_input_ids = _chunk_input_ids_by_sentences(
            text=TEST_TEXT,
            processor=cls.text_tokenizer,
            max_token_chunk_size=5,
            device=cls.device,
        )

        wav_manager = get_wav_manager()
        ref_audio_path = wav_manager.ensure_available("calm_male")
        ref_text = wav_manager.get_text("calm_male")
        cls.voice_clone_prompt, cls.ref_ids = get_voice_ref(
            cls.model, cls.text_tokenizer, None, ref_audio_path, ref_text
        )
        voice_clone_prompt_dict = VoiceClonePromptItem.to_batched_dict(cls.voice_clone_prompt)

        # deterministic generation config
        gen_config = dict(DEFAULT_QWEN3_CONFIG)
        gen_config["max_new_tokens"] = 200
        gen_config["temperature"] = 0.1
        gen_config["subtalker_temperature"] = 0.1

        cls.talker_codes_list, _ = cls.model.generate(
            input_ids=cls.chunked_input_ids,
            ref_ids=cls.ref_ids,
            voice_clone_prompt=voice_clone_prompt_dict,
            instruct_ids=[None],
            languages=["English"],
            speakers=[None],
            non_streaming_mode=True,
            **gen_config,
        )

        cls.wavs, cls.sample_rate = tokenizer_decode(
            cls.model, cls.talker_codes_list, cls.voice_clone_prompt
        )

    def test_number_of_chunks(self):
        self.assertEqual(len(self.chunked_input_ids), 3)

    def test_input_ids_shapes(self):
        self.assertEqual(self.chunked_input_ids[0].shape, torch.Size([1, 16]))
        self.assertEqual(self.chunked_input_ids[1].shape, torch.Size([1, 17]))
        self.assertEqual(self.chunked_input_ids[2].shape, torch.Size([1, 17]))

    def test_input_ids_first_values(self):
        for i, expected in enumerate(EXPECTED_INPUT_IDS_FIRST_5):
            actual = self.chunked_input_ids[i].cpu().numpy().flatten()[:5].tolist()
            self.assertEqual(actual, expected)

    def test_talker_codes_shape_and_dtype(self):
        codes = self.talker_codes_list[0]
        self.assertEqual(codes.shape, torch.Size([94, 16]))
        self.assertEqual(codes.dtype, torch.int64)

    def test_talker_codes_first_values(self):
        actual = self.talker_codes_list[0].cpu().numpy()[:5].tolist()
        self.assertEqual(actual, EXPECTED_TALKER_CODES_FIRST_5)

    def test_decoded_audio_shape(self):
        audio = self.wavs[0]
        self.assertEqual(audio.shape, (180480,))

    def test_decoded_audio_sample_rate(self):
        self.assertEqual(self.sample_rate, 24000)

    def test_decoded_audio_first_values(self):
        actual = self.wavs[0][:5].tolist()
        self.assertEqual(actual, EXPECTED_AUDIO_FIRST_5)

    @classmethod
    def tearDownClass(cls):
        # Restore working directory first
        os.chdir(cls._orig_cwd)
        # Free large CUDA / CPU tensors
        del cls.model, cls.text_tokenizer, cls.chunked_input_ids
        del cls.voice_clone_prompt, cls.ref_ids, cls.talker_codes_list
        del cls.wavs
        MemoryManager.clear_memory()
        super().tearDownClass()


if __name__ == "__main__":
    unittest.main()
