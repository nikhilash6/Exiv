import os

import torch

from transformers import AutoTokenizer

from .core.talker_base import Qwen3TTSBase
from .core.config import Qwen3TTSConfig
from .core.text_processor import Qwen3TTSTextProcessor
from ...audio_encoders.qwen3_tts_tokenizer import Qwen3TTSTokenizer
from ....model_utils.autoregressive_model_mixin import ARModelArchConfig
from ....components.enum import Model
from ....config import LOADING_MODE
from ....model_utils.helper_methods import get_state_dict, move_module
from ....utils.file import ensure_model_availability, read_safetensors_header
from ....utils.file_path import FilePaths
from ....utils.logging import app_logger


def _detect_qwen3_tts_variant(model_path: str):
    """
    Inspect the checkpoint to determine model variant.
    Returns: (model_type_enum, tts_model_type, tts_model_size, speaker_enc_dim)
    """
    ext = os.path.splitext(model_path)[1].lower()
    if ext in (".safetensors", ".sft"):
        header = read_safetensors_header(model_path)
        shapes = {k: v["shape"] for k, v in header.items()}
    else:
        # fallback: load state dict for .ckpt/.pt/.pth
        sd = get_state_dict(model_path, model_type="checkpoint")
        shapes = {k: list(v.shape) for k, v in sd.items()}
        del sd

    talker_hidden_size = None
    for key in shapes:
        if key.endswith("talker.model.norm.weight"):
            talker_hidden_size = shapes[key][0]
            break

    has_speaker_encoder = any("speaker_encoder." in k for k in shapes)
    speaker_enc_dim = None
    if has_speaker_encoder:
        for key in shapes:
            if "speaker_encoder.fc.weight" in key:
                speaker_enc_dim = shapes[key][0]
                break
        if speaker_enc_dim is None:
            for key in shapes:
                if "speaker_encoder." in key and key.endswith(".weight"):
                    speaker_enc_dim = shapes[key][0]
                    break

    if talker_hidden_size == 1024:
        tts_model_size = "0b6"
        if has_speaker_encoder:
            tts_model_type = "base"
            model_type_enum = Model.QWEN3_TTS_BASE.value
        else:
            tts_model_type = "custom_voice"
            model_type_enum = Model.QWEN3_TTS_CUSTOM_VOICE.value
    elif talker_hidden_size == 2048:
        tts_model_size = "1b7"
        if has_speaker_encoder:
            tts_model_type = "base"
            model_type_enum = Model.QWEN3_TTS_BASE.value
        else:
            # 1.7B CustomVoice and VoiceDesign are architecturally identical in weights
            # using filename as a narrow fallback for this specific case
            path_lower = str(model_path).lower()
            if "voice_design" in path_lower:
                tts_model_type = "voice_design"
                model_type_enum = Model.QWEN3_TTS_VOICE_DESIGN.value
            elif "custom_voice" in path_lower:
                tts_model_type = "custom_voice"
                model_type_enum = Model.QWEN3_TTS_CUSTOM_VOICE.value
            else:
                # default to voice_design for backward compatibility
                tts_model_type = "voice_design"
                model_type_enum = Model.QWEN3_TTS_VOICE_DESIGN.value
    else:
        app_logger.warning(
            f"Unrecognized talker hidden size {talker_hidden_size}, falling back to filename heuristics."
        )
        path_lower = str(model_path).lower()
        if "custom_voice" in path_lower:
            tts_model_type = "custom_voice"
            model_type_enum = Model.QWEN3_TTS_CUSTOM_VOICE.value
            tts_model_size = "unknown"
        elif "base" in path_lower:
            tts_model_type = "base"
            model_type_enum = Model.QWEN3_TTS_BASE.value
            tts_model_size = "unknown"
        elif "voice_design" in path_lower:
            tts_model_type = "voice_design"
            model_type_enum = Model.QWEN3_TTS_VOICE_DESIGN.value
            tts_model_size = "unknown"
        else:
            tts_model_type = "voice_design"
            model_type_enum = Model.QWEN3_TTS_VOICE_DESIGN.value
            tts_model_size = "unknown"

    return model_type_enum, tts_model_type, tts_model_size, speaker_enc_dim


def get_qwen3_tts_instance(
    model_path,
    download_url=None,
    force_load_mode=LOADING_MODE.NORMAL_LOAD.value,
    force_dtype=None,
):
    app_logger.info("Resolving model availability...")
    path_data = FilePaths.get_path(model_path, file_type="checkpoint")
    model_path = path_data.path
    download_url = download_url if download_url else path_data.url
    model_path = ensure_model_availability(model_path, download_url)
    
    cls = Qwen3TTSBase
    dict_dtype = torch.float16
    config = Qwen3TTSConfig()
    
    model_type_enum, tts_model_type, tts_model_size, speaker_enc_dim = _detect_qwen3_tts_variant(model_path)
    config.model_type = model_type_enum
    config.tts_model_type = tts_model_type
    config.tts_model_size = tts_model_size
    
    if tts_model_type == "base" and speaker_enc_dim is not None:
        config.speaker_encoder_config.enc_dim = speaker_enc_dim
    
    model_arch_config = ARModelArchConfig(model_type=model_type_enum)
    app_logger.info(f"Detected Qwen3-TTS variant: size={tts_model_size}, type={tts_model_type}")
    
    dtype = force_dtype or dict_dtype
    app_logger.info(f"Initializing {cls.__name__}...")
    qwen3_tts = cls(config=config, force_load_mode=force_load_mode, dtype=dtype)
    qwen3_tts.model_arch_config = model_arch_config
    qwen3_tts.load_model(model_path=model_path, download_url=download_url, model_type="checkpoint")
    
    # tokenizers are hardcoded during training and switching them doesn't make much sense
    try:
        text_tokenizer_backend = AutoTokenizer.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign")
        text_tokenizer = Qwen3TTSTextProcessor(tokenizer=text_tokenizer_backend)
    except Exception as e:
        app_logger.warning(f"Could not load text tokenizer: {e}")
        text_tokenizer = None
        
    try:
        st_path_data = FilePaths.get_path("qwen3_tts_tokenizer_12hz.safetensors", file_type="audio_encoder")
        st_model_path = ensure_model_availability(
            st_path_data.path, 
            st_path_data.url,
            force_download=False
        )
        
        speech_tokenizer = Qwen3TTSTokenizer(model_path="Qwen/Qwen3-TTS-Tokenizer-12Hz")
        speech_tokenizer.model.load_model(
            model_path=st_model_path,
            download_url=st_path_data.url,
            dtype=force_dtype if force_dtype is not None else torch.float16,
            model_type="checkpoint"
        )
        qwen3_tts.speech_tokenizer = speech_tokenizer
        app_logger.info("Speech tokenizer loaded successfully")
    except Exception as e:
        app_logger.warning(f"Could not load Speech Tokenizer: {e}")
        import traceback
        app_logger.debug(traceback.format_exc())
        speech_tokenizer = None

    return qwen3_tts, text_tokenizer, speech_tokenizer
