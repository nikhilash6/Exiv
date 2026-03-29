# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0

from enum import Enum
from typing import Dict, Optional


class Qwen3TTSSpeaker(Enum):
    VIVIAN = "vivian"
    SERENA = "serena"
    UNCLE_FU = "uncle_fu"
    DYLAN = "dylan"
    ERIC = "eric"
    RYAN = "ryan"
    AIDEN = "aiden"
    ONO_ANNA = "ono_anna"
    SOHEE = "sohee"
    
    def __str__(self) -> str:
        return self.value

    @classmethod
    def from_string(cls, name: str) -> "Qwen3TTSSpeaker":
        """Get speaker enum from string (case-insensitive)."""
        name = name.lower().replace(" ", "_")
        for speaker in cls:
            if speaker.value == name:
                return speaker
        raise ValueError(f"Unknown speaker: {name}. Supported: {[s.value for s in cls]}")


class Qwen3TTSLanguage(Enum):
    """Supported languages for Qwen3-TTS."""
    AUTO = "auto"
    CHINESE = "chinese"
    ENGLISH = "english"
    JAPANESE = "japanese"
    KOREAN = "korean"
    GERMAN = "german"
    FRENCH = "french"
    SPANISH = "spanish"
    ITALIAN = "italian"
    PORTUGUESE = "portuguese"
    RUSSIAN = "russian"

    def __str__(self) -> str:
        return self.value

    @classmethod
    def from_string(cls, name: str) -> "Qwen3TTSLanguage":
        """Get language enum from string (case-insensitive)."""
        name = name.lower()
        for lang in cls:
            if lang.value == name:
                return lang
        raise ValueError(f"Unknown language: {name}. Supported: {[l.value for l in cls]}")


# Speaker metadata: dialect and native language
SPEAKER_INFO: Dict[Qwen3TTSSpeaker, Dict[str, Optional[str]]] = {
    Qwen3TTSSpeaker.VIVIAN: {
        "name": "Vivian",
        "description": "Bright, slightly edgy young female voice",
        "native_language": "Chinese",
        "dialect": None,
        "gender": "female",
    },
    Qwen3TTSSpeaker.SERENA: {
        "name": "Serena",
        "description": "Warm, gentle young female voice",
        "native_language": "Chinese",
        "dialect": None,
        "gender": "female",
    },
    Qwen3TTSSpeaker.UNCLE_FU: {
        "name": "Uncle Fu",
        "description": "Seasoned male voice with a low, mellow timbre",
        "native_language": "Chinese",
        "dialect": None,
        "gender": "male",
    },
    Qwen3TTSSpeaker.DYLAN: {
        "name": "Dylan",
        "description": "Youthful Beijing male voice with clear, natural timbre",
        "native_language": "Chinese",
        "dialect": "beijing_dialect",  # 京腔/北京话
        "gender": "male",
    },
    Qwen3TTSSpeaker.ERIC: {
        "name": "Eric",
        "description": "Lively Chengdu male voice with slightly husky brightness",
        "native_language": "Chinese",
        "dialect": "sichuan_dialect",  # 川普/四川话
        "gender": "male",
    },
    Qwen3TTSSpeaker.RYAN: {
        "name": "Ryan",
        "description": "Dynamic male voice with strong rhythmic drive",
        "native_language": "English",
        "dialect": None,
        "gender": "male",
    },
    Qwen3TTSSpeaker.AIDEN: {
        "name": "Aiden",
        "description": "Sunny American male voice with a clear midrange",
        "native_language": "English",
        "dialect": None,
        "gender": "male",
    },
    Qwen3TTSSpeaker.ONO_ANNA: {
        "name": "Ono Anna",
        "description": "Playful Japanese female voice with a light, nimble timbre",
        "native_language": "Japanese",
        "dialect": None,
        "gender": "female",
    },
    Qwen3TTSSpeaker.SOHEE: {
        "name": "Sohee",
        "description": "Warm Korean female voice with rich emotion",
        "native_language": "Korean",
        "dialect": None,
        "gender": "female",
    },
}


def get_speaker_info(speaker: Qwen3TTSSpeaker) -> Dict[str, Optional[str]]:
    """Get metadata for a speaker."""
    return SPEAKER_INFO[speaker].copy()

def get_speakers_by_language(language: str) -> list[Qwen3TTSSpeaker]:
    """Get all speakers native to a given language."""
    language = language.lower()
    return [
        speaker for speaker, info in SPEAKER_INFO.items()
        if info["native_language"].lower() == language
    ]

def get_speakers_by_dialect() -> Dict[str, list[Qwen3TTSSpeaker]]:
    """Get speakers grouped by their dialect."""
    dialects: Dict[str, list[Qwen3TTSSpeaker]] = {
        "standard_mandarin": [],
        "beijing_dialect": [],
        "sichuan_dialect": [],
        "non_chinese": [],
    }
    for speaker, info in SPEAKER_INFO.items():
        dialect = info["dialect"]
        if dialect is None:
            if info["native_language"] == "Chinese":
                dialects["standard_mandarin"].append(speaker)
            else:
                dialects["non_chinese"].append(speaker)
        else:
            dialects[dialect].append(speaker)
    return dialects


# default sample texts for each speaker (for testing/demos)
DEFAULT_SPEAKER_TEXTS: Dict[Qwen3TTSSpeaker, Dict[str, str]] = {
    Qwen3TTSSpeaker.VIVIAN: {
        "Chinese": "你好，我是Vivian。我是一个性格开朗、略带锋芒的年轻女性。很高兴认识你！",
        "English": "Hello, this is Vivian speaking in English. I can speak multiple languages fluently.",
    },
    Qwen3TTSSpeaker.SERENA: {
        "Chinese": "你好，我是Serena。我的声音温暖而温柔，希望能为你带来舒适的体验。",
        "English": "Hello, this is Serena. Though my native language is Chinese, I can also speak English naturally.",
    },
    Qwen3TTSSpeaker.UNCLE_FU: {
        "Chinese": "你好，我是福叔。我是一个声音低沉、醇厚的成熟男性，有什么可以帮你的吗？",
        "English": "Hello, I'm Uncle Fu. Even as a Chinese native speaker, I can communicate with you in English.",
    },
    Qwen3TTSSpeaker.DYLAN: {
        "Chinese": "嘿，我是Dylan。我是一个北京小伙子，说话带着京味儿，声音清晰自然。",
        "English": "Hey there! Dylan here. I may be from Beijing, but I can totally speak English too!",
    },
    Qwen3TTSSpeaker.ERIC: {
        "Chinese": "你好噻，我是Eric。我是个成都小伙儿，声音有点沙哑但很亮堂，巴适得板！",
        "English": "Hi! Eric speaking. This is me speaking English with my unique voice from Chengdu.",
    },
    Qwen3TTSSpeaker.RYAN: {
        "English": "Hello, I'm Ryan. I have a dynamic voice with strong rhythmic drive. How can I help you today?",
    },
    Qwen3TTSSpeaker.AIDEN: {
        "English": "Hello there! My name is Aiden. I'm here to help you with any questions you might have.",
    },
    Qwen3TTSSpeaker.ONO_ANNA: {
        "Japanese": "こんにちは、小野アンナです。明るく軽やかな声が特徴の日本女性です。よろしくお願いします！",
        "English": "Hello! This is Ono Anna speaking English. Even though I'm Japanese, I can speak English well!",
    },
    Qwen3TTSSpeaker.SOHEE: {
        "Korean": "안녕하세요, 소희입니다. 따뜻하고 감성적인 한국어 여성 목소리입니다. 만나서 반갑습니다!",
        "English": "Hello, Sohee here. I'm a Korean speaker, but I can also communicate with you in English.",
    },
}


# cross-lingual texts (mainly for dev)
CROSS_LINGUAL_ENGLISH_TEXTS: Dict[Qwen3TTSSpeaker, str] = {
    Qwen3TTSSpeaker.VIVIAN: "Hello, this is Vivian speaking in English. I can speak multiple languages fluently.",
    Qwen3TTSSpeaker.SERENA: "Hello, this is Serena. Though my native language is Chinese, I can also speak English naturally.",
    Qwen3TTSSpeaker.UNCLE_FU: "Hello, I'm Uncle Fu. Even as a Chinese native speaker, I can communicate with you in English.",
    Qwen3TTSSpeaker.DYLAN: "Hey there! Dylan here. I may be from Beijing, but I can totally speak English too!",
    Qwen3TTSSpeaker.ERIC: "Hi! Eric speaking. This is me speaking English with my unique voice from Chengdu.",
    Qwen3TTSSpeaker.RYAN: "Hi, Ryan here. I'm a native English speaker with a dynamic, rhythmic voice.",
    Qwen3TTSSpeaker.AIDEN: "Hello! Aiden speaking. As a native English speaker, I'm here to assist you with anything you need.",
    Qwen3TTSSpeaker.ONO_ANNA: "Hello! This is Ono Anna speaking English. Even though I'm Japanese, I can speak English well!",
    Qwen3TTSSpeaker.SOHEE: "Hello, Sohee here. I'm a Korean speaker, but I can also communicate with you in English.",
}
