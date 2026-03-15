"""
qwen_tts: Qwen-TTS package.
"""

from .inference.qwen3_tts_model import Qwen3TTSModel, VoiceClonePromptItem

__all__ = ["Qwen3TTSModel", "VoiceClonePromptItem"]
