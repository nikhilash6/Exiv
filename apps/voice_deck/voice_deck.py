import torch

from exiv.components.models.qwen3_tts.constructor import get_qwen3_tts_instance
from exiv.components.models.qwen3_tts.inference.qwen3_tts_model import Qwen3TTSPipeline

raw_model, text_tokenizer, _ = get_qwen3_tts_instance(
    model_path="qwen3_tts_12hz_custom_voice_1_7b.safetensors",
    force_dtype=torch.float16
)

qwen3_wrapper = Qwen3TTSPipeline(model=raw_model, processor=text_tokenizer)

supported_speakers = qwen3_wrapper.get_supported_speakers()
supported_languages = qwen3_wrapper.get_supported_languages()
print(f"\nSupported speakers: {supported_speakers}")
print(f"Supported languages: {supported_languages}")

