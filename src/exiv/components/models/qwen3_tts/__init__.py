"""
qwen_tts: Qwen-TTS package.
"""

from .utils import (
    VoiceClonePromptItem,
    create_voice_clone_prompt,
    get_voice_ref,
    tokenizer_decode,
    merge_generate_kwargs,
    load_audio_to_np,
    normalize_audio_inputs,
    tokenize_text,
    tokenize_texts,
    DEFAULT_QWEN3_CONFIG,
)
from .enums import (
    Qwen3TTSSpeaker,
    Qwen3TTSLanguage,
    SPEAKER_INFO,
    DEFAULT_SPEAKER_TEXTS,
    CROSS_LINGUAL_ENGLISH_TEXTS,
    get_speaker_info,
    get_speakers_by_language,
    get_speakers_by_dialect,
)

__all__ = [
    # Utils
    "VoiceClonePromptItem",
    "create_voice_clone_prompt",
    "get_voice_ref",
    "tokenizer_decode",
    "merge_generate_kwargs",
    "load_audio_to_np",
    "normalize_audio_inputs",
    "tokenize_text",
    "tokenize_texts",
    "DEFAULT_QWEN3_CONFIG",
    # Enums
    "Qwen3TTSSpeaker",
    "Qwen3TTSLanguage",
    "SPEAKER_INFO",
    "DEFAULT_SPEAKER_TEXTS",
    "CROSS_LINGUAL_ENGLISH_TEXTS",
    "get_speaker_info",
    "get_speakers_by_language",
    "get_speakers_by_dialect",
]
