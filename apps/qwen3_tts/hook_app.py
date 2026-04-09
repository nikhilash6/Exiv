"""
Qwen3 TTS Application with AudioContextHook for voice continuity

This version uses the AudioContextHook to maintain voice consistency
across chunks by passing audio context between them.

Usage:
    # Generate with automatic chunking and voice continuity
    python hook_app.py --mode generate --ref_audio_id calm_male --text "Very long text..."
"""

import os
import sys
import torch
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

from exiv.utils.text_chunking import chunk_text_by_sentences

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

# Import the audio context hook
from exiv.model_patching.audio_context_hook import enable_audio_context, remove_audio_context

from wav_manager import get_manager as get_wav_manager
wav_manager = get_wav_manager()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

default_text = "The last Moon-Tender of Aethyria climbed the glass mountain at twilight, her ladder of braided starlight slung across one shoulder. Lyra had inherited the position when her grandmother's hands grew too spotted and trembling to polish the lunar surface—a duty passed down through generations that the modern world had forgotten, then dismissed as fairy tale. Tonight was the Waning, the most dangerous phase. She reached the summit where the moon sat lodged in a crater of its own making, no longer distant and celestial but close enough to touch, pale and pitted like ancient bone. It hummed, as it always did, a frequency that vibrated in her teeth and behind her eyes. The moon was not a rock, she had learned as a child, but a *repository*. Every dream deferred, every secret whispered at midnight, every spell cast in desperation—the moon absorbed them all, storing them in its hollows like rainwater in stone crevices. But repositories can overflow. Lyra unrolled her ladder and began to climb the curved face. The silver dust there coated her fingers immediately, cold and slightly greasy, the residue of ten thousand unspoken confessions. She reached the first fissure—a dark line bisecting the Sea of Tranquility—and dipped her brush into the jar at her hip. The ink within was distilled from her own breath and morning dew, the only solvent gentle enough for this work. She painted along the crack, whispering the old words: *Memory to mist, Burden to breeze, Let the silver sieve Grant its release.* The fissure glowed, then exhaled. A shape emerged: the ghost of a woman's regret, coiling upward like smoke, dissipating into the stars. Lyra worked methodically, moving from crack to crater, emptying the moon of its accumulated sorrows. Each release left her trembling, for she experienced them all—the heartbreaks, the grief, the quiet desperations of sleepers she would never meet. By midnight she reached the deepest hollow, the Mare Crisium, where the darkest magic..."

_model = None
_text_tokenizer = None


def _get_model(model_path=None, enable_voice_continuity=True, chunk_size=None):
    """
    Get or initialize the TTS model.
    
    Args:
        model_path: Path to model weights
        enable_voice_continuity: Whether to enable AudioContextHook for voice continuity
        chunk_size: Max tokens per chunk (default: 700, or 50 for testing)
    """
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
        
        # Enable AudioContextHook for voice continuity across chunks
        if enable_voice_continuity and hasattr(_model, 'talker'):
            max_chunk = chunk_size if chunk_size is not None else 700
            enable_audio_context(
                _model,
                max_chunk_size=max_chunk,  # Text tokens per chunk
            )
            print(f"✓ AudioContextHook enabled for voice continuity (chunk_size={max_chunk})")
    
    return _model, _text_tokenizer


def handle_generate(
    text: str,
    ref_audio_id: Optional[str] = None,
    audio_path: Optional[str] = None,
    language: str = "English",
    output_name: Optional[str] = None,
    enable_chunking: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Generate speech using reference audio with voice continuity.
    
    Args:
        text: Text to synthesize
        ref_audio_id: Voice ID from registry (optional)
        audio_path: Direct path to audio file (optional)
        language: Target language (default: English)
        output_name: Custom name for output file (optional)
        enable_chunking: Enable automatic chunking for long texts
        
    Returns:
        Dict with status and generated audio path
    """
    # Reload registry to get fresh data
    wav_manager._registry = wav_manager._load_registry()
    
    # Validate inputs
    if not ref_audio_id and not audio_path:
        print("✗ Must provide either ref_audio_id or audio_path")
        raise ValueError("Missing reference audio")
    
    # Determine reference audio path and text
    if ref_audio_id:
        if not wav_manager.has_voice(ref_audio_id):
            print(f"✗ Voice '{ref_audio_id}' not found in registry")
            raise ValueError(f"Voice '{ref_audio_id}' not found in registry")
        
        ref_path = wav_manager.ensure_available(ref_audio_id)
        try:
            ref_text = wav_manager.get_text(ref_audio_id)
        except ValueError:
            ref_text = ""
            print(f"⚠ No reference text found for '{ref_audio_id}' - using x-vector mode")
        source_info = f"registry:{ref_audio_id}"
        print(f"\nUsing registry voice: {ref_audio_id}")
    else:
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
    
    print(f"Text: {text[:100]}{'...' if len(text) > 100 else ''}")
    
    # Get model with hook enabled
    chunk_size = kwargs.get('chunk_size', None)
    model, text_tokenizer = _get_model(enable_voice_continuity=enable_chunking, chunk_size=chunk_size)
    
    # Tokenize input text
    chunked_text = chunk_text_by_sentences(text, chunk_size)
    input_id_list = []
    for t in chunked_text:
        assistant_text = text_tokenizer.build_assistant_text(t)
        input_ids = text_tokenizer(text=assistant_text, return_tensors="pt", padding=True)
        input_id = input_ids["input_ids"].to(device)
        input_id = input_id if input_id.dim() > 1 else input_id.unsqueeze(0)
        input_id_list.append(input_id)
    
    # Get voice reference
    voice_clone_prompt, ref_ids = get_voice_ref(model, text_tokenizer, None, ref_path, ref_text)
    voice_clone_prompt_dict = VoiceClonePromptItem.to_batched_dict(voice_clone_prompt)
    
    # Calculate max tokens based on text length
    word_count = len(text.split())
    estimated_audio_tokens = int(word_count / 2.5 * 12.5 * 1.5)
    calculated_max_tokens = max(500, min(8192, estimated_audio_tokens))
    generation_config = DEFAULT_QWEN3_CONFIG.copy()
    generation_config['max_new_tokens'] = calculated_max_tokens
    
    # Generate with hook handling chunking automatically
    # The hook (applied to self.talker) will split long texts and maintain voice continuity
    talker_codes_list, _ = model.generate(
        input_ids=[input_id_list],
        ref_ids=ref_ids,
        voice_clone_prompt=voice_clone_prompt_dict,
        instruct_ids=[None],
        languages=[language],
        speakers=[None],
        non_streaming_mode=True,
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
        "source": source_info,
        "voice_continuity": enable_chunking
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
    
    return {"1": output_paths[0]}


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Qwen3 TTS with AudioContextHook - Voice Continuity",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python hook_app.py --ref_audio_id calm_male --text "Hello world"
  python hook_app.py --audio_path ./voice.wav --ref_text "reference" --text "Hello"
        """
    )
    
    parser.add_argument("--text", default=default_text, help="Text to synthesize")
    parser.add_argument("--ref_audio_id", help="Voice ID from registry")
    parser.add_argument("--audio_path", help="Direct path to audio file")
    parser.add_argument("--ref_text", help="Reference text (required when using audio_path)")
    parser.add_argument("--language", default="English", help="Target language")
    parser.add_argument("--output_name", help="Custom name for output file")
    parser.add_argument("--no_chunking", action="store_true", help="Disable automatic chunking")
    
    args = parser.parse_args()
    
    try:
        result = handle_generate(
            text=args.text,
            ref_audio_id=args.ref_audio_id,
            audio_path=args.audio_path,
            ref_text=args.ref_text,
            language=args.language,
            output_name=args.output_name,
            enable_chunking=not args.no_chunking
        )
        print(f"\nResult: {result}")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
