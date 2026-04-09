"""
Qwen3 TTS Application

Two main modes:
1. upload: Upload an audio file to the voice registry
2. generate: Select reference audio (uploaded or from registry) and generate speech from text

Usage:
    # Upload a new voice
    python app.py --mode upload --audio_id my_voice --audio_path /path/to/audio.wav --text "transcript"
    
    # Generate using a registry voice
    python app.py --mode generate --ref_audio_id calm_male --text "Hello world"
    
    # Generate using uploaded file directly
    python app.py --mode generate --audio_path /path/to/uploaded.wav --text "Hello world"
"""

import os
import sys
import torch
import shutil
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, Union

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from exiv.server.app_core import App, AppOutputType, Input, Output
from exiv.utils.enum import ExtendedEnum

from exiv.components.models.qwen3_tts.constructor import get_qwen3_tts_instance
from exiv.components.models.qwen3_tts import (
    VoiceClonePromptItem,
    get_voice_ref,
    tokenizer_decode,
    DEFAULT_QWEN3_CONFIG,
    Qwen3TTSSpeaker,
    Qwen3TTSLanguage,
    SPEAKER_INFO,
)
from exiv.utils.file import MediaProcessor

from wav_manager import get_manager as get_wav_manager
wav_manager = get_wav_manager()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

_model = None
_text_tokenizer = None


def _get_model(model_path=None):
    """Get or initialize the TTS model."""
    if not model_path:
        model_path = "qwen3_tts_12hz_base_1_7b.safetensors"
    global _model, _text_tokenizer
    if _model is None:
        raw_model, text_tokenizer, _ = get_qwen3_tts_instance(
            model_path=model_path,
            force_dtype=torch.float16 if device == "cuda" else torch.float32
        )
        _text_tokenizer = text_tokenizer
        _model = raw_model
        _model.to(device)
        if _model.speech_tokenizer is not None:
            _model.speech_tokenizer.model.to(device)
            _model.speech_tokenizer.device = device
    return _model, _text_tokenizer


# =============================================================================
# MODE 1: UPLOAD - Upload an audio file to registry
# =============================================================================

def handle_upload(
    audio_id: str,
    audio_path: str,
    text: str,
    tags: str = "",
    language: str = "English",
    **kwargs
) -> Dict[str, Any]:
    """
    Upload an audio file to the voice registry.
    
    Args:
        audio_id: Unique identifier for the voice
        audio_path: Path to the audio file to upload
        text: Transcript of the audio
        tags: Voice description tags (optional)
        language: Language code (default: English)
    
    Returns:
        Dict with status and voice info
    """
    audio_path = Path(audio_path)
    
    if not audio_path.exists():
        print(f"✗ Audio file not found: {audio_path}")
        return {"status": "error", "message": f"File not found: {audio_path}"}
    
    if wav_manager.has_voice(audio_id):
        print(f"⚠ Voice '{audio_id}' already exists. Use a different ID or delete first.")
        return {"status": "error", "message": "Voice already exists"}
    
    # Copy to wavs directory and register
    import soundfile as sf
    data, sr = sf.read(str(audio_path), dtype="float32")
    
    saved_path = wav_manager.save_wav(
        audio_id=audio_id,
        audio_array=data,
        sample_rate=sr,
        text=text,
        tags=tags,
        language=language,
        url="",  # Local upload, no URL
        copy_from=str(audio_path)
    )
    
    print(f"✓ Uploaded voice: {audio_id}")
    print(f"  Path: {saved_path}")
    print(f"  Tags: {tags}")
    print(f"  Text: {text[:80]}...")
    
    return {
        "status": "uploaded",
        "audio_id": audio_id,
        "path": saved_path,
        "tags": tags,
        "language": language
    }


# =============================================================================
# MODE 2: GENERATE - Generate speech using reference audio
# =============================================================================

def handle_generate(
    text: str,
    ref_audio_id: Optional[str] = None,
    audio_path: Optional[str] = None,
    language: str = "English",
    output_name: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Generate speech using reference audio.
    
    Can use either:
    - A registered voice ID (ref_audio_id)
    - A direct audio file path (audio_path)
    
    Args:
        text: Text to synthesize
        ref_audio_id: Voice ID from registry (optional)
        audio_path: Direct path to audio file (optional)
        language: Target language (default: English)
        output_name: Custom name for output file (optional)
    
    Returns:
        Dict with status and generated audio path
    """
    # Reload registry to get fresh data (in case voices were added)
    wav_manager._registry = wav_manager._load_registry()
    
    # Validate inputs
    if not ref_audio_id and not audio_path:
        print("✗ Must provide either ref_audio_id or audio_path")
        raise ValueError("Missing reference audio")
    
    # Determine reference audio path and text
    if ref_audio_id:
        # Use registry voice
        if not wav_manager.has_voice(ref_audio_id):
            print(f"✗ Voice '{ref_audio_id}' not found in registry")
            raise ValueError(f"Voice '{ref_audio_id}' not found in registry")
        
        ref_path = wav_manager.ensure_available(ref_audio_id)
        # allow empty text for x-vector mode
        try:
            ref_text = wav_manager.get_text(ref_audio_id)
        except ValueError:
            ref_text = ""
            print(f"⚠ No reference text found for '{ref_audio_id}' - using x-vector mode")
        source_info = f"registry:{ref_audio_id}"
        print(f"\nUsing registry voice: {ref_audio_id}")
    else:
        # Use direct file path
        ref_path = Path(audio_path)
        if not ref_path.exists():
            print(f"✗ Audio file not found: {ref_path}")
            raise ValueError(f"File not found: {ref_path}")
        
        ref_path = str(ref_path)
        ref_text = kwargs.get("ref_text", "")
        if not ref_text:
            print("⚠ Warning: No reference text provided for audio file")
        source_info = f"file:{ref_path}"
        print(f"\nUsing audio file: {ref_path}")
    
    print(f"Text: {text}")
    
    model, text_tokenizer = _get_model()
    
    # Tokenize input text
    assistant_text = text_tokenizer.build_assistant_text(text)
    input_ids = text_tokenizer(text=assistant_text, return_tensors="pt", padding=True)
    input_id = input_ids["input_ids"].to(device)
    input_id = input_id if input_id.dim() > 1 else input_id.unsqueeze(0)
    
    # Get voice reference
    voice_clone_prompt, ref_ids = get_voice_ref(model, text_tokenizer, None, ref_path, ref_text)
    voice_clone_prompt_dict = VoiceClonePromptItem.to_batched_dict(voice_clone_prompt)
    
    # formula: words / 2.5 words/sec * 12.5 tokens/sec * 1.5 margin
    word_count = len(text.split())
    estimated_audio_tokens = int(word_count / 2.5 * 12.5 * 1.5)
    calculated_max_tokens = max(500, min(8192, estimated_audio_tokens))
    generation_config = DEFAULT_QWEN3_CONFIG.copy()
    generation_config['max_new_tokens'] = calculated_max_tokens
    
    # Generate
    talker_codes_list, _ = model.generate(
        input_ids=[input_id],
        ref_ids=ref_ids,
        voice_clone_prompt=voice_clone_prompt_dict,
        instruct_ids=[None],
        languages=[language],
        speakers=[None],
        non_streaming_mode=True,
        enable_chunking=True,
        **generation_config,
    )
    
    # Decode
    wavs, sample_rate = tokenizer_decode(model, talker_codes_list, voice_clone_prompt_dict)
    
    # Save output
    audio_tensor = torch.from_numpy(wavs[0]).unsqueeze(0).unsqueeze(0)
    
    metadata = {
        "sample_rate": sample_rate,
        "language": language,
        "text": text,
        "source": source_info
    }
    
    if output_name:
        metadata["custom_name"] = output_name
    
    output_paths = MediaProcessor.save_outputs(
        audio_tensor,
        metadata=metadata,
        subfolder="qwen3-tts",
        media_type="audio"
    )
    
    duration = len(wavs[0]) / sample_rate
    print(f"✓ Generated: {output_paths[0]} ({duration:.2f}s)")
    
    # Return in format expected by Exiv: {output_id: value}
    return {"1": output_paths[0]}


# =============================================================================
# UTILITY: List available voices in registry
# =============================================================================

def handle_list(
    filter_tag: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    List all voices in the registry.
    
    Args:
        filter_tag: Filter voices by tag (optional)
    
    Returns:
        Dict with voice list
    """
    voices = wav_manager.list_voices()
    
    if filter_tag:
        voices = {
            k: v for k, v in voices.items() 
            if filter_tag.lower() in v.get("tags", "").lower()
        }
    
    print(f"\nFound {len(voices)} voice(s) in registry:")
    print("-" * 60)
    
    for audio_id, info in voices.items():
        print(f"\n  {audio_id}:")
        print(f"    Tags: {info.get('tags', '')}")
        print(f"    Lang: {info.get('language', 'English')}")
        print(f"    Text: {info.get('text', '')[:60]}...")
    
    # Return voices in a flat format compatible with the frontend
    # The frontend expects {voice_id: {text, tags, language, url, filepath}}
    simplified_voices = {}
    for k, v in voices.items():
        simplified_voices[k] = {
            "text": v.get("text", ""),
            "tags": v.get("tags", ""),
            "language": v.get("language", "English"),
            "url": v.get("url", ""),
            "filepath": v.get("filepath", "")
        }
    
    return {
        "status": "success",
        "voices": simplified_voices
    }


# =============================================================================
# ADDITIONAL HANDLERS (for app interface)
# =============================================================================

class Qwen3TTSMode(ExtendedEnum):
    UPLOAD = "upload"
    GENERATE = "generate"
    LIST = "list"
    GET_VOICE_AUDIO = "get_voice_audio"


def handle_get_voice_audio(audio_id: str, **kwargs) -> Dict[str, Any]:
    """Get voice audio data as base64 for playback."""
    if not wav_manager.has_voice(audio_id):
        raise ValueError(f"Voice '{audio_id}' not found in registry")
    
    # Ensure voice is downloaded
    filepath = wav_manager.ensure_available(audio_id)
    
    # Read file and encode as base64
    import base64
    with open(filepath, 'rb') as f:
        audio_data = f.read()
    
    audio_base64 = base64.b64encode(audio_data).decode('utf-8')
    
    return {
        "status": "success",
        "audio_id": audio_id,
        "filepath": filepath,
        "audio_base64": audio_base64,
        "mime_type": "audio/wav"
    }


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

MODE_HANDLERS = {
    "upload": handle_upload,
    "generate": handle_generate,
    "list": handle_list,
    "get_voice_audio": handle_get_voice_audio,
}


def main() -> Dict[str, Any]:
    """Main entry point for the Qwen3 TTS application."""
    parser = argparse.ArgumentParser(
        description="Qwen3 TTS - Voice Cloning Application",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upload a new voice
  python app.py upload --audio_id my_voice --audio_path ./voice.wav --text "hello"
  
  # Generate using registry voice
  python app.py generate --ref_audio_id calm_male --text "Hello world"
  
  # Generate using uploaded file
  python app.py generate --audio_path ./my_voice.wav --ref_text "reference text" --text "Hello"
  
  # List all voices
  python app.py list
        """
    )
    
    subparsers = parser.add_subparsers(dest="mode", help="Available modes")
    
    # Upload mode
    upload_parser = subparsers.add_parser("upload", help="Upload an audio file to registry")
    upload_parser.add_argument("--audio_id", required=True, help="Unique identifier for the voice")
    upload_parser.add_argument("--audio_path", required=True, help="Path to audio file")
    upload_parser.add_argument("--text", required=True, help="Transcript of the audio")
    upload_parser.add_argument("--tags", default="", help="Voice description tags")
    upload_parser.add_argument("--language", default="English", help="Language code")
    
    # Generate mode
    gen_parser = subparsers.add_parser("generate", help="Generate speech from text")
    gen_parser.add_argument("--text", required=True, help="Text to synthesize")
    gen_parser.add_argument("--ref_audio_id", help="Voice ID from registry")
    gen_parser.add_argument("--audio_path", help="Direct path to audio file")
    gen_parser.add_argument("--ref_text", help="Reference text (required when using audio_path)")
    gen_parser.add_argument("--language", default="English", help="Target language")
    gen_parser.add_argument("--output_name", help="Custom name for output file")
    
    # List mode
    list_parser = subparsers.add_parser("list", help="List available voices")
    list_parser.add_argument("--filter", dest="filter_tag", help="Filter by tag")
    
    args = parser.parse_args()
    
    if not args.mode:
        parser.print_help()
        return {"status": "error", "message": "No mode specified"}
    
    handler = MODE_HANDLERS.get(args.mode)
    if not handler:
        return {"status": "error", "message": f"Unknown mode: {args.mode}"}
    
    print(f"\nMode: {args.mode}")
    print("=" * 60)
    
    try:
        params = {k: v for k, v in vars(args).items() if v is not None and k != "mode"}
        result = handler(**params)
        return result
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    result = main()
    
    print("\n" + "=" * 60)
    print("Result:")
    print("=" * 60)
    import json
    print(json.dumps(result, indent=2, default=str))


# Exiv App Definition
# Create an App wrapper for the generate mode

def app_handler(**params):
    """Handler for Exiv App interface."""
    mode = params.get("mode", "generate")
    
    if mode == "generate":
        return handle_generate(**params)
    elif mode == "upload":
        return handle_upload(**params)
    elif mode == "list":
        return handle_list(**params)
    elif mode == "get_voice_audio":
        return handle_get_voice_audio(**params)
    else:
        return {"status": "error", "message": f"Unknown mode: {mode}"}

# Define the Exiv App
app = App(
    name="Qwen3 TTS",
    description="Qwen3 Text-to-Speech with Voice Cloning",
    inputs={
        'mode': Input(label="Mode", type="select", options=["generate", "upload", "list", "get_voice_audio"], default="generate"),
        'text': Input(label="Text to Synthesize", type="str", default="", resizable=True),
        'ref_audio_id': Input(label="Reference Voice ID", type="str", default="calm_male.wav"),
        'audio_path': Input(label="Audio File Path", type="str", default=""),
        'ref_text': Input(label="Reference Text", type="str", default="", resizable=True),
        'language': Input(label="Language", type="select", options=[
            "English", "Chinese", "Spanish", "French", "German", 
            "Japanese", "Korean", "Italian", "Portuguese", "Russian", "Arabic", "Hindi"
        ], default="English"),
        'output_name': Input(label="Output Name", type="str", default=""),
        'audio_id': Input(label="Voice ID (for upload)", type="str", default=""),
        'tags': Input(label="Voice Tags (for upload)", type="str", default=""),
    },
    outputs=[Output(id=1, type=AppOutputType.AUDIO.value)],
    extra_metadata={
        'frontend_assets': {
            'js': 'index.js',
            'css': 'style.css'
        }
    },
    handler=app_handler
)
