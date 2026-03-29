#!/usr/bin/env python3
"""Generate audio for all available Qwen3-TTS speakers and their dialects."""
import os
import sys
import torch
import numpy as np
import soundfile as sf

sys.path.insert(0, '/root/exiv-private/src')

from exiv.components.models.qwen3_tts.constructor import get_qwen3_tts_instance
from exiv.components.models.qwen3_tts.inference.qwen3_tts_model import DEFAULT_QWEN3_CONFIG, Qwen3TTSPipeline
from exiv.components.models.qwen3_tts import (
    Qwen3TTSSpeaker,
    Qwen3TTSLanguage,
    SPEAKER_INFO,
    DEFAULT_SPEAKER_TEXTS,
    CROSS_LINGUAL_ENGLISH_TEXTS,
)
from exiv.config import global_config

# Dialect demonstrations - showcasing regional Chinese dialects
# Using speaker enums for type safety
DIALECT_DEMOS = {
    "Standard_Mandarin": {
        "speaker": Qwen3TTSSpeaker.VIVIAN,
        "language": Qwen3TTSLanguage.CHINESE,
        "text": "大家好，这是标准的普通话。我们可以用清晰的标准中文进行交流。",
        "description": "Standard Mandarin Chinese"
    },
    "Beijing_Dialect": {
        "speaker": Qwen3TTSSpeaker.DYLAN,
        "language": Qwen3TTSLanguage.CHINESE,
        "text": "咱北京人啊，说话就这个味儿！您听听，这京腔京韵的多地道啊。",
        "description": "Beijing Dialect (Northern Chinese)"
    },
    "Sichuan_Dialect": {
        "speaker": Qwen3TTSSpeaker.ERIC,
        "language": Qwen3TTSLanguage.CHINESE,
        "text": "要得嘛，我们四川人说话就是这么巴适！你听我这川普，安逸得很噻！",
        "description": "Sichuan Dialect (Southwestern Chinese)"
    },
}

# Multi-language support demos - showing one speaker in many languages
MULTI_LANGUAGE_DEMOS = [
    {"language": Qwen3TTSLanguage.CHINESE, "text": "这是中文普通话测试。你好，世界！"},
    {"language": Qwen3TTSLanguage.ENGLISH, "text": "This is English language testing. Hello, world!"},
    {"language": Qwen3TTSLanguage.JAPANESE, "text": "これは日本語のテストです。こんにちは、世界！"},
    {"language": Qwen3TTSLanguage.KOREAN, "text": "이것은 한국어 테스트입니다. 안녕하세요, 세계!"},
    {"language": Qwen3TTSLanguage.GERMAN, "text": "Dies ist ein deutscher Test. Hallo Welt!"},
    {"language": Qwen3TTSLanguage.FRENCH, "text": "Ceci est un test en français. Bonjour le monde!"},
    {"language": Qwen3TTSLanguage.SPANISH, "text": "Esta es una prueba en español. ¡Hola mundo!"},
    {"language": Qwen3TTSLanguage.ITALIAN, "text": "Questo è un test in italiano. Ciao mondo!"},
    {"language": Qwen3TTSLanguage.PORTUGUESE, "text": "Este é um teste em português. Olá mundo!"},
    {"language": Qwen3TTSLanguage.RUSSIAN, "text": "Это тест на русском языке. Привет, мир!"},
]

global_config.auto_download = True

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

raw_model, text_tokenizer, _ = get_qwen3_tts_instance(
    model_path="qwen3_tts_12hz_custom_voice_1_7b.safetensors",
    force_dtype=torch.float16 if device == "cuda" else torch.float32
)

pipeline = Qwen3TTSPipeline(model=raw_model, processor=text_tokenizer)
pipeline.model.to(device)
if pipeline.model.speech_tokenizer is not None:
    pipeline.model.speech_tokenizer.model.to(device)
    pipeline.model.speech_tokenizer.device = device

# Get supported speakers and languages from the model
supported_speakers = pipeline.get_supported_speakers()
supported_languages = pipeline.get_supported_languages()
print(f"\nSupported speakers: {supported_speakers}")
print(f"Supported languages: {supported_languages}")


def generate_speaker(
    speaker: Qwen3TTSSpeaker,
    language: Qwen3TTSLanguage,
    text: str,
    output_file: str,
    description: str = ""
) -> bool:
    """Generate audio for a specific speaker."""
    speaker_info = SPEAKER_INFO[speaker]
    
    print(f"\n{'='*60}")
    print(f"Generating: {speaker_info['name']}")
    if description:
        print(f"Description: {description}")
    print(f"Language: {language.value}")
    print(f"Text: {text[:80]}...")
    print(f"Output: {output_file}")
    
    try:
        # tokenize text
        input = text_tokenizer(text=text, return_tensors="pt", padding=True)
        input_id = input["input_ids"].to(device)
        input_ids = input_id if input_id.dim() > 1 else input_id.unsqueeze(0)
        
        # no instruct ids
        instruct_ids = None
        talker_codes_list, _ = pipeline.model.generate(
            input_ids=[input_ids],
            instruct_ids=[instruct_ids],
            languages=[language.value],
            speakers=[speaker.value],
            non_streaming_mode=True,
            **DEFAULT_QWEN3_CONFIG,
        )
        wavs, sample_rate = pipeline.model.speech_tokenizer.decode([{"audio_codes": c} for c in talker_codes_list])
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        sf.write(output_file, wavs[0], sample_rate)
        
        duration = len(wavs[0]) / sample_rate
        max_amp = np.abs(wavs[0]).max()
        print(f"✓ Saved: {output_file}")
        print(f"  Duration: {duration:.2f}s")
        print(f"  Max amplitude: {max_amp:.4f}")
        return True
    except Exception as e:
        print(f"✗ Error generating {speaker_info['name']}: {e}")
        return False


def main():
    output_dir = "/root/exiv-private/output/all_speakers"
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("QWEN3-TTS ALL SPEAKERS GENERATOR")
    print("="*60)
    print(f"Total speakers: {len(Qwen3TTSSpeaker)}")
    
    results = []
    
    # Part 1: Generate native language samples for each speaker
    print("\n" + "="*60)
    print("PART 1: NATIVE LANGUAGE SAMPLES")
    print("="*60)
    
    for speaker in Qwen3TTSSpeaker:
        info = SPEAKER_INFO[speaker]
        native_lang = info["native_language"]
        texts = DEFAULT_SPEAKER_TEXTS[speaker]
        
        # Get text for native language (fallback to English if not available)
        text = texts.get(native_lang, texts.get("English", ""))
        if not text:
            print(f"Skipping {info['name']} - no sample text available")
            continue
            
        output_file = f"{output_dir}/{speaker.value}_native.wav"
        success = generate_speaker(
            speaker=speaker,
            language=Qwen3TTSLanguage.from_string(native_lang),
            text=text,
            output_file=output_file,
            description=info["description"]
        )
        results.append((info['name'], "native", native_lang, success))
    
    # Part 2: Generate English cross-lingual samples
    print("\n" + "="*60)
    print("PART 2: CROSS-LINGUAL SAMPLES (ENGLISH)")
    print("="*60)
    
    for speaker in Qwen3TTSSpeaker:
        info = SPEAKER_INFO[speaker]
        text = CROSS_LINGUAL_ENGLISH_TEXTS[speaker]
        
        output_file = f"{output_dir}/{speaker.value}_english.wav"
        success = generate_speaker(
            speaker=speaker,
            language=Qwen3TTSLanguage.ENGLISH,
            text=text,
            output_file=output_file,
            description="Speaking English"
        )
        results.append((info['name'], "cross-lingual", "English", success))
    
    # Part 3: Dialect demonstrations (Chinese dialects)
    print("\n" + "="*60)
    print("PART 3: CHINESE DIALECT DEMONSTRATIONS")
    print("="*60)
    
    for demo_name, demo_info in DIALECT_DEMOS.items():
        output_file = f"{output_dir}/dialect_{demo_name.lower()}.wav"
        success = generate_speaker(
            speaker=demo_info["speaker"],
            language=demo_info["language"],
            text=demo_info["text"],
            output_file=output_file,
            description=demo_info["description"]
        )
        results.append((
            SPEAKER_INFO[demo_info["speaker"]]["name"],
            "dialect",
            demo_info["language"].value,
            success
        ))
    
    # Part 4: Multi-language support (demonstrating one speaker in many languages)
    print("\n" + "="*60)
    print("PART 4: MULTI-LANGUAGE SUPPORT (VIVIAN SPEAKING 10 LANGUAGES)")
    print("="*60)
    
    for lang_demo in MULTI_LANGUAGE_DEMOS:
        lang = lang_demo["language"]
        text = lang_demo["text"]
        output_file = f"{output_dir}/multilang_vivian_{lang.value}.wav"
        success = generate_speaker(
            speaker=Qwen3TTSSpeaker.VIVIAN,
            language=lang,
            text=text,
            output_file=output_file,
            description=f"Vivian speaking {lang.value}"
        )
        results.append(("Vivian", "multilang", lang.value, success))
    
    # Summary
    print("\n" + "="*60)
    print("GENERATION SUMMARY")
    print("="*60)
    
    successful = sum(1 for _, _, _, s in results if s)
    total = len(results)
    
    print(f"\nTotal: {successful}/{total} successful")
    print(f"\nOutput directory: {output_dir}")
    print("\nGenerated files by category:")
    
    categories = {}
    for speaker, mode, lang, success in results:
        if mode not in categories:
            categories[mode] = []
        categories[mode].append((speaker, lang, success))
    
    for mode, items in categories.items():
        print(f"\n  {mode.upper()}:")
        for speaker, lang, success in items:
            status = "✓" if success else "✗"
            print(f"    {status} {speaker} ({lang})")
    
    print("\n" + "="*60)
    print("SPEAKER REFERENCE TABLE")
    print("="*60)
    print(f"\n{'Speaker':<15} {'Native Language':<20} {'Dialect':<20} {'Description'}")
    print("-" * 90)
    for speaker in Qwen3TTSSpeaker:
        info = SPEAKER_INFO[speaker]
        dialect = info["dialect"] or "Standard"
        print(f"{info['name']:<15} {info['native_language']:<20} {dialect:<20} {info['description']}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
