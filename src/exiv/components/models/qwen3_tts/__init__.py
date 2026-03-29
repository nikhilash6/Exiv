"""
qwen_tts: Qwen-TTS package.
"""

from .inference.qwen3_tts_model import Qwen3TTSPipeline, VoiceClonePromptItem
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
    "Qwen3TTSPipeline",
    "VoiceClonePromptItem",
    "Qwen3TTSSpeaker",
    "Qwen3TTSLanguage",
    "SPEAKER_INFO",
    "DEFAULT_SPEAKER_TEXTS",
    "CROSS_LINGUAL_ENGLISH_TEXTS",
    "get_speaker_info",
    "get_speakers_by_language",
    "get_speakers_by_dialect",
]
