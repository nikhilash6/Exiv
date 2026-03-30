import os
import sys
import torch

from exiv.components.models.qwen3_tts.constructor import get_qwen3_tts_instance
from exiv.components.models.qwen3_tts.inference.qwen3_tts_model import DEFAULT_QWEN3_CONFIG, Qwen3TTSPipeline
from exiv.components.models.qwen3_tts import (
    Qwen3TTSSpeaker,
    Qwen3TTSLanguage,
    SPEAKER_INFO,
)
from exiv.utils.file import MediaProcessor

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

# Two speakers for cross-lingual demo
# Speaker 1: Native English, speaking non-native Chinese
# Speaker 2: Native Chinese, speaking non-native English

SPEAKERS = {
    "aiden": {
        "speaker": Qwen3TTSSpeaker.AIDEN,
        "native_lang": "English",
        "native_text": "Hello! I'm Aiden, a native English speaker from London. This is how I sound in my natural language.",
        "foreign_lang": Qwen3TTSLanguage.CHINESE,
        "foreign_text": "你好！我是迪伦。我正在学习中文，这可能听起来有些不自然，但我在努力改进！",
        "foreign_desc": "speaking Chinese (non-native)"
    },
    "vivian": {
        "speaker": Qwen3TTSSpeaker.VIVIAN,
        "native_lang": "Chinese",
        "native_text": "大家好！我是薇薇安，一个说标准普通话的中文母语者。这是我的自然声音。",
        "foreign_lang": Qwen3TTSLanguage.ENGLISH,
        "foreign_text": "Hello! I'm Vivian. I'm a native Chinese speaker trying to speak English. My accent might not be perfect, but I hope you can understand me!",
        "foreign_desc": "speaking English (non-native)"
    }
}

def generate_audio(speaker_id, language, text, filename_prefix):
    """Generate audio for a speaker."""
    from exiv.utils.file_path import FilePaths
    
    speaker_info = SPEAKER_INFO[speaker_id]
    
    print(f"\nGenerating: {speaker_info['name']} ({language.value})")
    print(f"Text: {text}")
    
    # build assistant text
    assistant_text = text_tokenizer.build_assistant_text(text)
    
    # tokenize
    input_ids = text_tokenizer(text=assistant_text, return_tensors="pt", padding=True)
    input_id = input_ids["input_ids"].to(device)
    input_id = input_id if input_id.dim() > 1 else input_id.unsqueeze(0)
    
    # generate
    talker_codes_list, _ = pipeline.model.generate(
        input_ids=[input_id],
        instruct_ids=[None],
        languages=[language.value],
        speakers=[speaker_id.value],
        non_streaming_mode=True,
        **DEFAULT_QWEN3_CONFIG,
    )
    
    # decode
    wavs, sample_rate = pipeline.model.speech_tokenizer.decode(
        [{"audio_codes": c} for c in talker_codes_list]
    )
    
    # (batch, channels, samples)
    audio_tensor = torch.from_numpy(wavs[0]).unsqueeze(0).unsqueeze(0)  # (1, 1, samples)
    output_paths = MediaProcessor.save_outputs(
        audio_tensor,
        metadata={"sample_rate": sample_rate, "speaker": speaker_info['name'], "language": language.value, "text": text},
        subfolder="voice_deck",
        media_type="audio"
    )
    
    duration = len(wavs[0]) / sample_rate
    print(f"✓ Saved: {output_paths[0]} ({duration:.2f}s)")
    
    return output_paths[0]


def main():
    print("\n" + "="*60)
    print("SIMPLE CROSS-LINGUAL TTS DEMO")
    print("="*60)
    
    for name, config in SPEAKERS.items():
        print(f"\n{'='*60}")
        print(f"Speaker: {SPEAKER_INFO[config['speaker']]['name']}")
        print(f"Description: {SPEAKER_INFO[config['speaker']]['description']}")
        print(f"Native Language: {config['native_lang']}")
        print("="*60)
        
        # Native language
        native_lang = Qwen3TTSLanguage.from_string(config['native_lang'])
        generate_audio(
            config['speaker'],
            native_lang,
            config['native_text'],
            f"{name}_native"
        )
        
        # Foreign language (cross-lingual)
        generate_audio(
            config['speaker'],
            config['foreign_lang'],
            config['foreign_text'],
            f"{name}_foreign"
        )
    
    from exiv.utils.file_path import FilePaths
    output_base = FilePaths.get_output_directory()
    voice_deck_dir = os.path.join(output_base, "voice_deck")
    
    print("\n" + "="*60)
    print("DONE! Generated files in:")
    print(f"  {voice_deck_dir}")
    print("="*60)
    
    for name, config in SPEAKERS.items():
        speaker_name = SPEAKER_INFO[config['speaker']]['name']
        print(f"\n{speaker_name}:")
        print(f"  Native:    voice_deck/output_audio_*.wav")
        print(f"  Foreign:   voice_deck/output_audio_*.wav")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
