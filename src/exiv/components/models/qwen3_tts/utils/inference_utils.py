# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import base64
import io
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse

import librosa
import numpy as np
import soundfile as sf
import torch

AudioLike = Union[
    str,                     # wav path, URL, base64
    np.ndarray,              # waveform (requires sr)
    Tuple[np.ndarray, int],  # (waveform, sr)
]

MaybeList = Union[Any, List[Any]]


DEFAULT_QWEN3_CONFIG = dict(
    do_sample=True,
    top_k=50,
    top_p=1.0,
    temperature=0.9,
    repetition_penalty=1.05,
    subtalker_dosample=True,
    subtalker_top_k=50,
    subtalker_top_p=1.0,
    subtalker_temperature=0.9,
    max_new_tokens=8192,
)


@dataclass
class VoiceClonePromptItem:
    """
    Container for one sample's voice-clone prompt information.
    
    This holds all the information needed for voice cloning a single sample.
    It supports two distinct voice cloning modes:
    
    1. **X-Vector Only Mode** (`x_vector_only_mode=True`):
       - Uses only the speaker embedding (ref_spk_embedding) to clone voice
       - Ignores ref_code and ref_text
       - Simpler but less accurate voice cloning
       - Useful when you don't have reference text or want faster inference
    
    2. **ICL Mode - In-Context Learning** (`x_vector_only_mode=False`, `icl_mode=True`):
       - Uses both reference codec codes (ref_code) AND speaker embedding
       - Requires reference text (ref_text) to be tokenized and passed separately
       - More accurate voice cloning as it learns from actual speech patterns
       - The model continues/conditions on the reference text + speech codes
    
    Fields:
        ref_code: Reference codec tensor. Shape: (T, Q) where T=time steps, 
                  Q=num quantizers (codebooks). Contains discrete speech tokens 
                  extracted from reference audio. Only used when icl_mode=True.
        
        ref_spk_embedding: Speaker embedding tensor. Shape: (D,) where D=speaker 
                          embedding dimension (typically 256 or 512). Contains 
                          continuous speaker characteristics extracted by speaker_encoder.
                          Used in both x-vector and ICL modes.
        
        x_vector_only_mode: If True, only speaker embedding is used. The reference 
                           codec codes are ignored.
        
        icl_mode: If True, ICL (in-context learning) is enabled. The model conditions 
                 on both ref_code and ref_spk_embedding. Note: icl_mode and 
                 x_vector_only_mode are mutually exclusive.
        
        ref_text: The reference transcript text. Required for ICL mode (when 
                 icl_mode=True), ignored in x-vector only mode.
    
    Example:
        >>> # Single sample with ICL mode
        >>> item = VoiceClonePromptItem(
        ...     ref_code=ref_codes_tensor,           # (T, 16) codec tokens
        ...     ref_spk_embedding=spk_emb_tensor,    # (256,) speaker vector
        ...     x_vector_only_mode=False,
        ...     icl_mode=True,
        ...     ref_text="Hello world"
        ... )
        >>> 
        >>> # X-vector only mode
        >>> item = VoiceClonePromptItem(
        ...     ref_code=None,
        ...     ref_spk_embedding=spk_emb_tensor,
        ...     x_vector_only_mode=True,
        ...     icl_mode=False,
        ...     ref_text=None
        ... )
    """
    ref_code: Optional[torch.Tensor]                 # (T, Q) or (T,) depending on tokenizer 25Hz/12Hz
    ref_spk_embedding: torch.Tensor                  # (D,)
    x_vector_only_mode: bool
    icl_mode: bool
    ref_text: Optional[str] = None
    
    @staticmethod
    def to_batched_dict(items: List["VoiceClonePromptItem"]) -> Dict[str, List]:
        """
        Convert a list of VoiceClonePromptItem to a batched dict format.
        
        Without to_batched_dict (individual items):
            # 3 voice samples we want to clone
            item1 = VoiceClonePromptItem(
                ref_code=tensor_a,           # reference audio codec tokens for sample 1
                ref_spk_embedding=vec_a,     # speaker embedding for sample 1
                x_vector_only_mode=False,
                icl_mode=True,
            )

            item2 = VoiceClonePromptItem(
                ref_code=tensor_b,           # sample 2
                ref_spk_embedding=vec_b,
                x_vector_only_mode=True,     # x-vector only, no ref_code
                icl_mode=False,
            )

            item3 = VoiceClonePromptItem(
                ref_code=tensor_c,           # sample 3
                ref_spk_embedding=vec_c,
                x_vector_only_mode=False,
                icl_mode=True,
            )

            items = [item1, item2, item3]  # List of 3 items
        
        With to_batched_dict (grouped by field):
            batched_dict = VoiceClonePromptItem.to_batched_dict(items)

            # Result:
            {
                "ref_code":           [tensor_a,  None,      tensor_c],  # all ref_codes grouped
                "ref_spk_embedding":  [vec_a,     vec_b,     vec_c],     # all speaker embeddings grouped
                "x_vector_only_mode": [False,     True,      False],     # all flags grouped
                "icl_mode":           [True,      False,     True],      # all flags grouped
            }
        """
        return {
            "ref_code": [it.ref_code for it in items],
            "ref_spk_embedding": [it.ref_spk_embedding for it in items],
            "x_vector_only_mode": [it.x_vector_only_mode for it in items],
            "icl_mode": [it.icl_mode for it in items],
        }


def _is_probably_base64(s: str) -> bool:
    """Check if a string is likely a base64 encoded audio."""
    if s.startswith("data:audio"):
        return True
    if ("/" not in s and "\\" not in s) and len(s) > 256:
        return True
    return False


def _is_url(s: str) -> bool:
    """Check if a string is a URL."""
    try:
        u = urlparse(s)
        return u.scheme in ("http", "https") and bool(u.netloc)
    except Exception:
        return False


def _decode_base64_to_wav_bytes(b64: str) -> bytes:
    """Decode base64 string to wav bytes."""
    if "," in b64 and b64.strip().startswith("data:"):
        b64 = b64.split(",", 1)[1]
    return base64.b64decode(b64)


def load_audio_to_np(x: str) -> Tuple[np.ndarray, int]:
    """
    Load audio from path, URL, or base64 string to numpy array.
    
    Args:
        x: Audio source (file path, URL, or base64 string)
        
    Returns:
        Tuple of (audio_array, sample_rate)
    """
    if _is_url(x):
        with urllib.request.urlopen(x) as resp:
            audio_bytes = resp.read()
        with io.BytesIO(audio_bytes) as f:
            audio, sr = sf.read(f, dtype="float32", always_2d=False)
    elif _is_probably_base64(x):
        wav_bytes = _decode_base64_to_wav_bytes(x)
        with io.BytesIO(wav_bytes) as f:
            audio, sr = sf.read(f, dtype="float32", always_2d=False)
    else:
        audio, sr = librosa.load(x, sr=None, mono=True)

    if audio.ndim > 1:
        audio = np.mean(audio, axis=-1)

    return audio.astype(np.float32), int(sr)


def normalize_audio_inputs(audios: Union[AudioLike, List[AudioLike]]) -> List[Tuple[np.ndarray, int]]:
    """
    Normalize audio inputs into a list of (waveform, sr).

    Supported forms:
      - str: wav path / URL / base64 audio string
      - (np.ndarray, sr): waveform + sampling rate
      - list of the above

    Args:
        audios: Audio input(s).

    Returns:
        List[Tuple[np.ndarray, int]]: List of (float32 waveform, original sr).

    Raises:
        ValueError: If a numpy waveform is provided without sr.
    """
    if isinstance(audios, list):
        items = audios
    else:
        items = [audios]

    out: List[Tuple[np.ndarray, int]] = []
    for a in items:
        if isinstance(a, str):
            out.append(load_audio_to_np(a))
        elif isinstance(a, tuple) and len(a) == 2 and isinstance(a[0], np.ndarray):
            out.append((a[0].astype(np.float32), int(a[1])))
        elif isinstance(a, np.ndarray):
            raise ValueError("For numpy waveform input, pass a tuple (audio, sr).")
        else:
            raise TypeError(f"Unsupported audio input type: {type(a)}")
    for i, a in enumerate(out):
        if a[0].ndim > 1:
            a = (np.mean(a[0], axis=-1).astype(np.float32), a[1])
            out[i] = a
    return out


def _ensure_list(x: MaybeList) -> List[Any]:
    """Ensure input is a list."""
    return x if isinstance(x, list) else [x]


def tokenize_text(processor, text: str, device: str = "cpu") -> torch.Tensor:
    """
    Tokenize a single text string.
    
    Args:
        processor: The text processor/tokenizer
        text: Text to tokenize
        device: Device to place tensor on
        
    Returns:
        Tokenized text tensor
    """
    inp = processor(text=text, return_tensors="pt", padding=True)
    input_id = inp["input_ids"].to(device)
    return input_id if input_id.dim() > 1 else input_id.unsqueeze(0)


def tokenize_texts(processor, texts: List[str], device: str = "cpu") -> List[torch.Tensor]:
    """
    Tokenize a list of text strings.
    
    Args:
        processor: The text processor/tokenizer
        texts: List of texts to tokenize
        device: Device to place tensors on
        
    Returns:
        List of tokenized text tensors
    """
    return [tokenize_text(processor, text, device) for text in texts]


def merge_generate_kwargs(
    generate_defaults: Optional[Dict[str, Any]] = None,
    do_sample: Optional[bool] = None,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    temperature: Optional[float] = None,
    repetition_penalty: Optional[float] = None,
    subtalker_dosample: Optional[bool] = None,
    subtalker_top_k: Optional[int] = None,
    subtalker_top_p: Optional[float] = None,
    subtalker_temperature: Optional[float] = None,
    max_new_tokens: Optional[int] = None,
    eos_token_id: Optional[int] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Merge user-provided generation arguments with defaults.

    Rule:
      - If the user explicitly passes a value (not None), use it.
      - Otherwise, use the value from generate_defaults if present.
      - Otherwise, fall back to the hard defaults.

    Args:
        generate_defaults: Default generation config from model
        do_sample, top_k, top_p, temperature, repetition_penalty,
        subtalker_dosample, subtalker_top_k, subtalker_top_p, subtalker_temperature, max_new_tokens:
            Common generation parameters.
        eos_token_id: End of sequence token ID
        **kwargs: Other arguments forwarded to model.generate()

    Returns:
        Dict[str, Any]: Final kwargs to pass into model.generate()
    """
    defaults = generate_defaults or {}

    def pick(name: str, user_val: Any) -> Any:
        if user_val is not None:
            return user_val
        if name in defaults:
            return defaults[name]
        return DEFAULT_QWEN3_CONFIG[name]

    merged = dict(kwargs)
    merged.update(
        do_sample=pick("do_sample", do_sample),
        top_k=pick("top_k", top_k),
        top_p=pick("top_p", top_p),
        temperature=pick("temperature", temperature),
        repetition_penalty=pick("repetition_penalty", repetition_penalty),
        subtalker_dosample=pick("subtalker_dosample", subtalker_dosample),
        subtalker_top_k=pick("subtalker_top_k", subtalker_top_k),
        subtalker_top_p=pick("subtalker_top_p", subtalker_top_p),
        subtalker_temperature=pick("subtalker_temperature", subtalker_temperature),
        max_new_tokens=pick("max_new_tokens", max_new_tokens),
        eos_token_id=eos_token_id,
    )
    return merged


def _supported_languages_set(model) -> Optional[set]:
    """Get supported languages as a set."""
    langs = getattr(model, "get_supported_languages", None)
    if callable(langs):
        v = langs()
        if v is None:
            return None
        return set([str(x).lower() for x in v])
    return None


def _supported_speakers_set(model) -> Optional[set]:
    """Get supported speakers as a set."""
    spks = getattr(model, "get_supported_speakers", None)
    if callable(spks):
        v = spks()
        if v is None:
            return None
        return set([str(x).lower() for x in v])
    return None


def validate_languages(model, languages: List[str]) -> None:
    """
    Validate that requested languages are supported by the model.

    Args:
        model: The TTS model
        languages: Language names for each sample.

    Raises:
        ValueError: If any language is not supported.
    """
    supported = _supported_languages_set(model)
    if supported is None:
        return

    bad = []
    for lang in languages:
        if lang is None:
            bad.append(lang)
            continue
        if str(lang).lower() not in supported:
            bad.append(lang)
    if bad:
        raise ValueError(f"Unsupported languages: {bad}. Supported: {sorted(supported)}")


def validate_speakers(model, speakers: List[Optional[str]]) -> None:
    """
    Validate that requested speakers are supported by the model.

    Args:
        model: The TTS model
        speakers: Speaker names for each sample.

    Raises:
        ValueError: If any speaker is not supported.
    """
    supported = _supported_speakers_set(model)
    if supported is None:
        return

    bad = []
    for spk in speakers:
        if spk is None or spk == "":
            continue
        if str(spk).lower() not in supported:
            bad.append(spk)
    if bad:
        raise ValueError(f"Unsupported speakers: {bad}. Supported: {sorted(supported)}")


def create_voice_clone_prompt(
    model,
    ref_audio: Union[AudioLike, List[AudioLike]],
    ref_text: Optional[Union[str, List[Optional[str]]]] = None,
    x_vector_only_mode: Union[bool, List[bool]] = False,
) -> List[VoiceClonePromptItem]:
    """
    Build voice-clone prompt items from reference audio (and optionally reference text).

    Modes:
      - x_vector_only_mode=True:
          Only speaker embedding is used to clone voice; ref_text/ref_code are ignored.
          This is mutually exclusive with ICL.
      - x_vector_only_mode=False:
          ICL mode is enabled automatically (icl_mode=True). In this case ref_text is required,
          because the model continues/conditions on the reference text + reference speech codes.

    Batch behavior:
      - ref_audio can be a single item or a list.
      - ref_text and x_vector_only_mode can be scalars or lists.
      - If any of them are lists with length > 1, lengths must match.

    Audio input:
      - str: local wav path / URL / base64
      - (np.ndarray, sr): waveform + sampling rate

    Args:
        model: The TTS model
        ref_audio: Reference audio(s) used to extract ref_code and ref_spk_embedding
        ref_text: Reference transcript(s). Required when x_vector_only_mode=False (ICL mode).
        x_vector_only_mode: Whether to use speaker embedding only.

    Returns:
        List[VoiceClonePromptItem]: List of prompt items

    Raises:
        ValueError:
            - If x_vector_only_mode=False but ref_text is missing.
            - If batch lengths mismatch.
    """
    if model.tts_model_type != "base":
        raise ValueError(
            f"model does not support create_voice_clone_prompt (type: {model.tts_model_type})"
        )
    
    ref_audio_list = _ensure_list(ref_audio)
    ref_text_list = _ensure_list(ref_text) if isinstance(ref_text, list) else ([ref_text] * len(ref_audio_list))
    xvec_list = _ensure_list(x_vector_only_mode) if isinstance(x_vector_only_mode, list) else ([x_vector_only_mode] * len(ref_audio_list))

    if len(ref_text_list) != len(ref_audio_list) or len(xvec_list) != len(ref_audio_list):
        raise ValueError("Batch size mismatch.")

    normalized = normalize_audio_inputs(ref_audio_list)

    ref_wavs_for_code: List[np.ndarray] = []
    ref_sr_for_code: List[int] = []
    for wav, sr in normalized:
        ref_wavs_for_code.append(wav)
        ref_sr_for_code.append(sr)

    if len(set(ref_sr_for_code)) == 1:
        enc = model.speech_tokenizer.encode(ref_wavs_for_code, sr=ref_sr_for_code[0])
        ref_codes = enc.audio_codes
    else:
        ref_codes = []
        for wav, sr in normalized:
            ref_codes.append(model.speech_tokenizer.encode(wav, sr=sr).audio_codes[0])

    items: List[VoiceClonePromptItem] = []
    for i, ((wav, sr), code, rtext, xvec_only) in enumerate(zip(normalized, ref_codes, ref_text_list, xvec_list)):
        if not xvec_only and (rtext is None or rtext == ""):
            raise ValueError(f"ref_text is required for ICL mode at index {i}")

        wav_resample = wav
        if sr != model.speaker_encoder_sample_rate:
            wav_resample = librosa.resample(
                y=wav_resample.astype(np.float32), 
                orig_sr=int(sr), 
                target_sr=model.speaker_encoder_sample_rate
            )

        spk_emb = model.extract_speaker_embedding(
            audio=wav_resample,
            sr=model.speaker_encoder_sample_rate
        )

        items.append(
            VoiceClonePromptItem(
                ref_code=None if xvec_only else code,
                ref_spk_embedding=spk_emb,
                x_vector_only_mode=bool(xvec_only),
                icl_mode=bool(not xvec_only),
                ref_text=rtext,
            )
        )
    return items


def get_voice_ref(
    model,
    processor,
    voice_clone_prompt: Optional[List[VoiceClonePromptItem]],
    ref_audio: Optional[Union[AudioLike, List[AudioLike]]],
    ref_text: Optional[Union[str, List[Optional[str]]]],
) -> Tuple[List[VoiceClonePromptItem], Optional[List[Optional[torch.Tensor]]]]:
    """
    Get voice reference data for generation.
    
    This is a convenience function that handles both direct voice_clone_prompt
    inputs and creating prompts from reference audio.
    
    Args:
        model: The TTS model
        processor: The text processor
        voice_clone_prompt: Pre-created voice clone prompt (List[VoiceClonePromptItem])
        ref_audio: Reference audio path/URL/base64 (optional)
        ref_text: Reference text transcript (optional)
        
    Returns:
        Tuple of:
            - voice_clone_prompt_items: List[VoiceClonePromptItem] - The voice clone prompt items
            - ref_ids: Optional[List[Optional[torch.Tensor]]] - Tokenized reference text IDs (None if no ref_text)
    """
    x_vector_only_mode = not bool(ref_text)
    if voice_clone_prompt is None:
        if ref_audio is None:
            raise ValueError("Either `voice_clone_prompt` or `ref_audio` must be provided.")
        prompt_items = create_voice_clone_prompt(
            model=model,
            ref_audio=ref_audio, 
            ref_text=ref_text, 
            x_vector_only_mode=x_vector_only_mode
        )
        ref_texts_for_ids = [it.ref_text for it in prompt_items]
    else:
        prompt_items = voice_clone_prompt
        ref_texts_for_ids = [it.ref_text for it in prompt_items]

    ref_ids = None
    if ref_texts_for_ids is not None:
        device = getattr(model, "gpu_device", "cpu")
        ref_ids = []
        for rt in ref_texts_for_ids:
            if rt is None or rt == "":
                ref_ids.append(None)
            else:
                ref_ids.append(tokenize_text(processor, processor.build_ref_text(rt), device))
                
    return prompt_items, ref_ids


def tokenizer_decode(
    model,
    talker_codes_list: List[torch.Tensor],
    voice_clone_prompt: Optional[List[VoiceClonePromptItem]] = None
) -> Tuple[List[np.ndarray], int]:
    """
    Decode talker codes to audio waveforms.
    
    Args:
        model: The TTS model with speech_tokenizer
        talker_codes_list: List of talker code tensors
        voice_clone_prompt: Optional list of VoiceClonePromptItem with ref_code per sample
        
    Returns:
        Tuple of (list of audio arrays, sample_rate)
    """
    # DEBUG: Check code values
    for i, codes in enumerate(talker_codes_list):
        print(f"DEBUG: talker_codes_list[{i}] shape={codes.shape}, min={codes.min().item()}, max={codes.max().item()}, dtype={codes.dtype}")
    
    codes_for_decode = []
    for i, codes in enumerate(talker_codes_list):
        ref_code = voice_clone_prompt[i].ref_code if voice_clone_prompt is not None else None
        if ref_code is not None:
            combined = torch.cat([ref_code.to(codes.device), codes], dim=0)
            print(f"DEBUG: codes_for_decode[{i}] combined shape={combined.shape}, min={combined.min().item()}, max={combined.max().item()}")
            codes_for_decode.append(combined)
        else:
            print(f"DEBUG: codes_for_decode[{i}] shape={codes.shape}, min={codes.min().item()}, max={codes.max().item()}")
            codes_for_decode.append(codes)

    wavs_all, fs = model.speech_tokenizer.decode([{"audio_codes": c} for c in codes_for_decode])

    wavs_out: List[np.ndarray] = []
    for i, wav in enumerate(wavs_all):
        ref_code = voice_clone_prompt[i].ref_code if voice_clone_prompt is not None else None
        if ref_code is not None:
            ref_len = int(ref_code.shape[0])
            total_len = int(codes_for_decode[i].shape[0])
            cut = int(ref_len / max(total_len, 1) * wav.shape[0])
            wavs_out.append(wav[cut:])
        else:
            wavs_out.append(wav)

    return wavs_out, fs
