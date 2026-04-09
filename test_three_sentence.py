"""
Test script for AudioContextHook with three sentence paragraph.

This script:
1. Takes a three sentence paragraph
2. Splits it into sentence chunks and creates input_ids for each chunk
3. Each chunk contains complete sentences
4. Respects max_token_chunk_size limit
"""

import os
import sys

os.chdir('/root/exiv-private/apps/qwen3_tts')
sys.path.insert(0, '/root/exiv-private/apps/qwen3_tts')
sys.path.insert(0, '/root/exiv-private/src')

import torch

from exiv.components.models.qwen3_tts.utils.inference_utils import DEFAULT_QWEN3_CONFIG, get_voice_ref, tokenizer_decode
from exiv.components.models.qwen3_tts import VoiceClonePromptItem

import numpy as np
from exiv.components.models.qwen3_tts.constructor import get_qwen3_tts_instance
from exiv.utils.text_chunking import chunk_text_by_sentences
from wav_manager import get_manager as get_wav_manager

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

TEST_TEXT = "The sun was setting over the mountains. Birds were flying back to their nests. It was a peaceful evening in the countryside."
TEST_TEXT_1 = """The last Moon-Tender of Aethyria climbed the glass mountain at twilight, her ladder of braided starlight slung across one shoulder. Lyra had inherited the position when her grandmother's hands grew too spotted and trembling to polish the lunar surface—a duty passed down through generations that the modern world had forgotten, then dismissed as fairy tale.

Tonight was the Waning, the most dangerous phase.

She reached the summit where the moon sat lodged in a crater of its own making, no longer distant and celestial but close enough to touch, pale and pitted like ancient bone. It hummed, as it always did, a frequency that vibrated in her teeth and behind her eyes. The moon was not a rock, she had learned as a child, but a *repository*. Every dream deferred, every secret whispered at midnight, every spell cast in desperation—the moon absorbed them all, storing them in its hollows like rainwater in stone crevices.

But repositories can overflow.

Lyra unrolled her ladder and began to climb the curved face. The silver dust there coated her fingers immediately, cold and slightly greasy, the residue of ten thousand unspoken confessions. She reached the first fissure—a dark line bisecting the Sea of Tranquility—and dipped her brush into the jar at her hip. The ink within was distilled from her own breath and morning dew, the only solvent gentle enough for this work.

She painted along the crack, whispering the old words:

*"Memory to mist,  
Burden to breeze,  
Let the silver sieve  
Grant its release."*

The fissure glowed, then exhaled.

A shape emerged: the ghost of a woman's regret, coiling upward like smoke, dissipating into the stars. Lyra worked methodically, moving from crack to crater, emptying the moon of its accumulated sorrows. Each release left her trembling, for she experienced them all—the heartbreaks, the grief, the quiet desperations of sleepers she would never meet.

By midnight she reached the deepest hollow, the Mare Crisium, where the darkest magic pooled. Here, centuries ago, a sorceress had tried to trap the moon entirely, to freeze the world in endless night so her lover's death might be postponed. The residue of that spell still clung, stubborn and tar-black.

Lyra hesitated. To clear this would require a trade.

She thought of her grandmother, now blind and forgetting names in a cottage below. She thought of the world that no longer believed, that sailed ships by electric light and never wondered why the tides answered to a distant rock. If she failed, the moon would crack open. Dreams would rain down like shrapnel. Madness would follow.

Lyra pressed her palm to the black stain.

*"Take the memory of my name,"* she offered.

The moon drank.

The stain lifted, swirling into the vacuum, leaving only clean silver. And Lyra—well, she would still be Lyra, but when she descended to the village tomorrow, no one would know to greet her. The ledgers would show no birth record. Her cottage would appear empty to all who passed.

But the moon would wax full and bright, unburdened. The tides would turn. Dreamers would wake refreshed, unknowing that someone had paid their debt with her own reflection in the world's memory.

She coiled her ladder and began the long climb down, already feeling the hollowness where her name had lived. Above her, the moon pulsed once—a heartbeat of thanks—and in its pale light, she began to glow softly herself, becoming something new. Not remembered, but eternal.

A star, perhaps. Or a story told to children who might one day believe enough to look up and wonder who keeps the night-light burning."""


def chunk_input_ids_by_sentences(text, processor, max_token_chunk_size=30000, device="cuda"):
    """
    Split text into sentence chunks and convert each to input_ids.
    
    Args:
        text: text to split
        processor: text processor with build_assistant_text
        max_token_chunk_size: max tokens per chunk (not strictly enforced, ensures complete sentences)
        device: device
        
    Returns:
        list of input_ids tensors [1, seq_len]
    """
    # Split text into sentence chunks
    text_chunks = chunk_text_by_sentences(text, max_token_chunk_size)
    
    print(f"Text split into {len(text_chunks)} chunks:")
    for i, chunk in enumerate(text_chunks):
        print(f"  Chunk {i+1}: '{chunk}'")
    
    chunk_input_ids_list = []
    
    for i, chunk_text in enumerate(text_chunks):
        # Build assistant text format
        assistant_text = processor.build_assistant_text(chunk_text)
        
        # Tokenize
        input_ids = processor(text=assistant_text, return_tensors="pt", padding=True)
        chunk_tokens = input_ids["input_ids"].to(device)
        chunk_tokens = chunk_tokens if chunk_tokens.dim() > 1 else chunk_tokens.unsqueeze(0)
        
        print(f"  Chunk {i+1} input_ids: shape={chunk_tokens.shape}, tokens={chunk_tokens.shape[1]}")
        chunk_input_ids_list.append(chunk_tokens)
    
    return chunk_input_ids_list


def test_three_sentence():
    print("=" * 60)
    print("Testing Input_ids Chunking - 3 Sentences")
    print("=" * 60)
    print(f"\nTest text: {TEST_TEXT}")
    
    model_path = "models/checkpoints/qwen3_tts_12hz_base_1_7b.safetensors"
    
    print(f"\nLoading model...")
    model, text_tokenizer, _ = get_qwen3_tts_instance(
        model_path=model_path,
        force_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    model.to(device)
    if model.speech_tokenizer is not None:
        model.speech_tokenizer.model.to(device)
        model.speech_tokenizer.device = device
    
    # Create chunked input_ids
    print("\n--- Chunking Text into input_ids ---")
    chunked_input_ids = chunk_input_ids_by_sentences(
        text=TEST_TEXT,
        processor=text_tokenizer,
        max_token_chunk_size=5,
        device=device
    )
    
    print(f"\nCreated {len(chunked_input_ids)} input_id chunks")
    for i, chunk in enumerate(chunked_input_ids):
        print(f"  Chunk {i+1}: {chunk.shape} -> {chunk.tolist()}")
    
    wav_manager = get_wav_manager()
    ref_audio_path = wav_manager.ensure_available("calm_male")
    ref_text = wav_manager.get_text("calm_male")
    voice_clone_prompt, ref_ids = get_voice_ref(model, text_tokenizer, None, ref_audio_path, ref_text)
    voice_clone_prompt_dict = VoiceClonePromptItem.to_batched_dict(voice_clone_prompt)
    talker_codes_list, _ = model.generate(
        input_ids=chunked_input_ids,
        ref_ids=ref_ids,
        voice_clone_prompt=voice_clone_prompt_dict,
        instruct_ids=[None],
        languages=["English"],
        speakers=[None],
        non_streaming_mode=True,
        **DEFAULT_QWEN3_CONFIG,
    )
    
    # Decode
    print("\nDecoding audio...")
    wavs, sample_rate = tokenizer_decode(model, talker_codes_list, voice_clone_prompt)
    
    audio = wavs[0]
    print(f"\n{'=' * 60}")
    print(f"Audio stats:")
    print(f"  Shape: {audio.shape}")
    print(f"  Sample rate: {sample_rate}")
    print(f"  Duration: {audio.shape[0]/sample_rate:.2f}s")
    print(f"  Max amplitude: {np.abs(audio).max():.4f}")
    print(f"  Std dev: {audio.std():.4f}")
    
    # Save to output folder in repo
    import soundfile as sf
    output_dir = "/root/exiv-private/output"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "working_three_sentence.wav")
    sf.write(output_file, audio, sample_rate)
    print(f"\nSaved to: {output_file}")
    print(f"{'=' * 60}")
    
    return output_file


if __name__ == "__main__":
    try:
        chunks = test_three_sentence()
        print(f"\n✅ SUCCESS! Created {len(chunks)} chunks")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
