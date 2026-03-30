"""
quick overview

- list: List all available voices in registry
- add: Add a new voice URL to registry
- delete: Remove a voice from registry
- clone: Clone a voice using reference audio from registry
- design: Design a voice from text description
- custom: Generate using preset speakers (CustomVoice model)

usage:
    python voice_deck.py --action list
    python voice_deck.py --action add --audio_id my_voice --url https://... --text "hello" --tags "male, calm"
    python voice_deck.py --action clone --ref_audio_id calm_male --text "Hello world"
    python voice_deck.py --action design --instruct "male, deep voice" --text "Hello world"
    python voice_deck.py --action custom --speaker vivian --text "Hello world"
"""

import os
import sys
import torch
from pathlib import Path
from typing import Dict, Any, Optional

from exiv.components.models.qwen3_tts.constructor import get_qwen3_tts_instance
from exiv.components.models.qwen3_tts.inference.qwen3_tts_model import DEFAULT_QWEN3_CONFIG, Qwen3TTSPipeline
from exiv.components.models.qwen3_tts import (
    Qwen3TTSSpeaker,
    Qwen3TTSLanguage,
    SPEAKER_INFO,
)
from exiv.utils.file import MediaProcessor

from wav_manager import get_manager as get_wav_manager
wav_manager = get_wav_manager()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

_pipeline = None
_text_tokenizer = None

def _get_pipeline(model_path=None):
    if not model_path: model_path = "qwen3_tts_12hz_base_1_7b.safetensors"
    global _pipeline, _text_tokenizer
    if _pipeline is None:
        raw_model, text_tokenizer, _ = get_qwen3_tts_instance(
            model_path=model_path,
            force_dtype=torch.float16 if device == "cuda" else torch.float32
        )
        _text_tokenizer = text_tokenizer
        _pipeline = Qwen3TTSPipeline(model=raw_model, processor=text_tokenizer)
        _pipeline.model.to(device)
        if _pipeline.model.speech_tokenizer is not None:
            _pipeline.model.speech_tokenizer.model.to(device)
            _pipeline.model.speech_tokenizer.device = device
    return _pipeline, _text_tokenizer


# =============================================================================
# ACTION HANDLERS
# =============================================================================

def handle_list(**params) -> Dict[str, Any]:
    """
    List all voices in registry.
    
    Optional params:
        filter: Filter by tags (case-insensitive substring match)
    """
    filter_tag = params.get("filter", "").lower()
    voices = wav_manager.list_voices()
    
    if filter_tag:
        voices = {
            k: v for k, v in voices.items() 
            if filter_tag in v.get("tags", "").lower()
        }
    
    # Format for display
    result = {
        "count": len(voices),
        "voices": {}
    }
    for audio_id, info in voices.items():
        result["voices"][audio_id] = {
            "text": info.get("text", "")[:100],
            "tags": info.get("tags", ""),
            "language": info.get("language", "English"),
            "url": info.get("url", "")[:50] if info.get("url") else "(local)",
        }
    
    print(f"\nFound {len(voices)} voice(s)")
    for aid, info in result["voices"].items():
        print(f"\n  {aid}:")
        print(f"    Tags: {info['tags']}")
        print(f"    Lang: {info['language']}")
    
    return result

def handle_add(**params) -> Dict[str, Any]:
    """
    Add a new voice to registry.
    
    Required params:
        audio_id: Unique identifier for the voice
        url: Remote URL to download the WAV
        text: Transcript of the audio
    
    Optional params:
        tags: Voice description tags
        language: Language code (default: English)
    """
    audio_id = params["audio_id"]
    url = params["url"]
    text = params["text"]
    tags = params.get("tags", "")
    language = params.get("language", "English")
    
    if wav_manager.has_voice(audio_id):
        print(f"⚠ Voice '{audio_id}' already exists. Use delete first to overwrite.")
        return {"status": "error", "message": "Voice already exists"}
    
    wav_manager.add_voice(audio_id, url, text, tags, language)
    
    print(f"✓ Added voice: {audio_id}")
    print(f"  URL: {url}")
    print(f"  Tags: {tags}")
    print(f"  Text: {text[:80]}...")
    
    return {"status": "added", "audio_id": audio_id}

def handle_delete(**params) -> Dict[str, Any]:
    """
    Delete a voice from registry.
    
    Required params:
        audio_id: Voice identifier to delete
    """
    audio_id = params["audio_id"]
    
    if not wav_manager.has_voice(audio_id):
        print(f"✗ Voice '{audio_id}' not found")
        return {"status": "error", "message": "Voice not found"}
    
    wav_manager.delete_voice(audio_id)
    print(f"✓ Deleted voice: {audio_id}")
    
    return {"status": "deleted", "audio_id": audio_id}

def handle_clone(**params) -> Dict[str, Any]:
    """
    Clone a voice using reference audio from registry.
    
    Required params:
        ref_audio_id: Voice ID from registry to use as reference
        text: Text to synthesize
    
    Optional params:
        language: Target language (default: English)
    """
    ref_audio_id = params["ref_audio_id"]
    text = params["text"]
    language = params.get("language", "English")
    
    if not wav_manager.has_voice(ref_audio_id):
        print(f"✗ Voice '{ref_audio_id}' not found in registry")
        return {"status": "error", "message": "Reference voice not found"}
    
    print(f"\nCloning voice: {ref_audio_id}")
    print(f"Text: {text}")
    
    # reference info
    ref_path = wav_manager.ensure_available(ref_audio_id)
    ref_text = wav_manager.get_text(ref_audio_id)
    ref_tags = wav_manager.get_tags(ref_audio_id)
    
    pipeline, text_tokenizer = _get_pipeline()
    
    # tokenize
    assistant_text = text_tokenizer.build_assistant_text(text)
    input_ids = text_tokenizer(text=assistant_text, return_tensors="pt", padding=True)
    input_id = input_ids["input_ids"].to(device)
    input_id = input_id if input_id.dim() > 1 else input_id.unsqueeze(0)
    
    # voice ref
    voice_clone_prompt_dict, ref_ids = pipeline.get_voice_ref(None, ref_path, ref_text)
    
    # generate
    talker_codes_list, _ = pipeline.model.generate(
        input_ids=[input_id],
        ref_ids=ref_ids,
        voice_clone_prompt=voice_clone_prompt_dict,
        instruct_ids=[None],
        languages=[language],
        speakers=[None],
        non_streaming_mode=True,
        **DEFAULT_QWEN3_CONFIG,
    )
    
    # decode
    wavs, sample_rate = pipeline.tokenizer_decode(talker_codes_list, voice_clone_prompt_dict)
    
    audio_tensor = torch.from_numpy(wavs[0]).unsqueeze(0).unsqueeze(0)
    output_paths = MediaProcessor.save_outputs(
        audio_tensor,
        metadata={"sample_rate": sample_rate, "language": language, "text": text, "cloned_from": ref_audio_id},
        subfolder="voice_deck",
        media_type="audio"
    )
    
    duration = len(wavs[0]) / sample_rate
    print(f"✓ Generated: {output_paths[0]} ({duration:.2f}s)")
    
    return {
        "status": "generated",
        "audio_path": output_paths[0],
        "duration": duration,
        "cloned_from": ref_audio_id
    }

def handle_custom(**params) -> Dict[str, Any]:
    """
    Generate audio using preset speakers (CustomVoice model style).
    
    Required params:
        speaker: Speaker name (vivian, serena, aiden, ryan, etc.)
        text: Text to synthesize
    
    Optional params:
        language: Target language (default: English)
    """
    speaker_name = params["speaker"].lower()
    text = params["text"]
    language = params.get("language", "English")
    
    speaker_map = {
        "vivian": Qwen3TTSSpeaker.VIVIAN,
        "serena": Qwen3TTSSpeaker.SERENA,
        "aiden": Qwen3TTSSpeaker.AIDEN,
        "ryan": Qwen3TTSSpeaker.RYAN,
        "uncle_fu": Qwen3TTSSpeaker.UNCLE_FU,
        "dylan": Qwen3TTSSpeaker.DYLAN,
        "eric": Qwen3TTSSpeaker.ERIC,
        "ono_anna": Qwen3TTSSpeaker.ONO_ANNA,
        "sohee": Qwen3TTSSpeaker.SOHEE,
    }
    
    if speaker_name not in speaker_map:
        print(f"✗ Unknown speaker: {speaker_name}")
        print(f"Available: {list(speaker_map.keys())}")
        return {"status": "error", "message": f"Unknown speaker: {speaker_name}"}
    
    speaker = speaker_map[speaker_name]
    speaker_info = SPEAKER_INFO[speaker]
    
    print(f"\nGenerating: {speaker_info['name']} ({language})")
    print(f"Text: {text}")
    
    pipeline, text_tokenizer = _get_pipeline(model_path="qwen3_tts_12hz_custom_voice_1_7b.safetensors")
    
    # tokenize
    assistant_text = text_tokenizer.build_assistant_text(text)
    input_ids = text_tokenizer(text=assistant_text, return_tensors="pt", padding=True)
    input_id = input_ids["input_ids"].to(device)
    input_id = input_id if input_id.dim() > 1 else input_id.unsqueeze(0)
    
    # generate
    talker_codes_list, _ = pipeline.model.generate(
        input_ids=[input_id],
        instruct_ids=[None],
        languages=[language],
        speakers=[speaker.value],
        non_streaming_mode=True,
        **DEFAULT_QWEN3_CONFIG,
    )
    
    # decode
    wavs, sample_rate = pipeline.tokenizer_decode(talker_codes_list)
    
    # save
    audio_tensor = torch.from_numpy(wavs[0]).unsqueeze(0).unsqueeze(0)
    output_paths = MediaProcessor.save_outputs(
        audio_tensor,
        metadata={"sample_rate": sample_rate, "speaker": speaker_info['name'], "language": language, "text": text},
        subfolder="voice_deck",
        media_type="audio"
    )
    
    duration = len(wavs[0]) / sample_rate
    print(f"✓ Generated: {output_paths[0]} ({duration:.2f}s)")
    
    return {
        "status": "generated",
        "audio_path": output_paths[0],
        "duration": duration,
        "speaker": speaker_info['name']
    }


ACTION_HANDLERS = {
    "list": handle_list,
    "add": handle_add,
    "delete": handle_delete,
    "clone": handle_clone,
    # "design": handle_design,
    "custom": handle_custom,
}


def main(**params) -> Dict[str, Any]:
    """
    Main entry point - routes to appropriate action handler.
    
    Required params:
        action: One of [list, add, delete, clone, design, custom]
    """
    action = params.get("action", "list")
    
    handler = ACTION_HANDLERS.get(action)
    if not handler:
        available = ", ".join(ACTION_HANDLERS.keys())
        print(f"✗ Unknown action: '{action}'")
        print(f"Available actions: {available}")
        return {"status": "error", "message": f"Unknown action: {action}"}
    
    print(f"\nAction: {action}")
    print("-" * 60)
    
    try:
        result = handler(**params)
        return result
    except KeyError as e:
        print(f"✗ Missing required parameter: {e}")
        return {"status": "error", "message": f"Missing parameter: {e}"}
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Voice Deck - Voice TTS Application",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            # List all voices
            python voice_deck.py --action list
            
            # Add a new voice
            python voice_deck.py --action add --audio_id my_voice --url https://... --text "hello"
            
            # Clone a voice
            python voice_deck.py --action clone --ref_audio_id calm_male --text "Hello world"
            
            # Use preset speaker
            python voice_deck.py --action custom --speaker vivian --text "Hello world"
        """
    )
    
    parser.add_argument("--action", type=str, default="list",
                       choices=list(ACTION_HANDLERS.keys()),
                       help="Action to perform")
    
    # common params
    parser.add_argument("--text", type=str, help="Text to synthesize")
    parser.add_argument("--language", type=str, default="English",
                       help="Target language")
    
    # list voices
    parser.add_argument("--filter", type=str, help="Filter voices by tags")
    
    # add voices
    parser.add_argument("--audio_id", type=str, help="Voice identifier")
    parser.add_argument("--url", type=str, help="Remote URL for voice")
    parser.add_argument("--tags", type=str, help="Voice description tags")
    
    # clone voices
    parser.add_argument("--ref_audio_id", type=str, 
                       help="Reference voice ID from registry")
    
    # custom params
    parser.add_argument("--speaker", type=str,
                       choices=["vivian", "serena", "aiden", "ryan", 
                               "uncle_fu", "dylan", "eric", "ono_anna", "sohee"],
                       help="Preset speaker name")
    
    args = parser.parse_args()
    params = {k: v for k, v in vars(args).items() if v is not None}
    
    result = main(**params)
    
    print("\n" + "=" * 60)
    print("Result:")
    print("=" * 60)
    import json
    print(json.dumps(result, indent=2, default=str))
