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
from ....utils.file import ensure_model_availability
from ....utils.file_path import FilePaths
from ....utils.logging import app_logger

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
    
    # TODO: add proper config loading from checkpoint
    cls = Qwen3TTSBase
    dict_dtype = torch.float16
    config = Qwen3TTSConfig()
    
    # TODO: just a temp check adding during dev, will remove it later
    model_path_lower = str(model_path).lower()
    if "custom_voice" in model_path_lower:
        config.model_type = Model.QWEN3_TTS_CUSTOM_VOICE.value
        config.tts_model_type = "custom_voice"
        model_arch_config = ARModelArchConfig(model_type=Model.QWEN3_TTS_CUSTOM_VOICE.value)
    elif "base" in model_path_lower:
        config.model_type = Model.QWEN3_TTS_BASE.value
        config.tts_model_type = "base"
        # NOTE: base model uses larger speaker encoder (enc_dim=2048)
        config.speaker_encoder_config.enc_dim = 2048
        model_arch_config = ARModelArchConfig(model_type=Model.QWEN3_TTS_BASE.value)
    elif "voice_design" in model_path_lower:
        config.model_type = Model.QWEN3_TTS_VOICE_DESIGN.value
        config.tts_model_type = "voice_design"
        model_arch_config = ARModelArchConfig(model_type=Model.QWEN3_TTS_VOICE_DESIGN.value)
    else:
        # Default to voice_design for backward compatibility
        config.model_type = Model.QWEN3_TTS_VOICE_DESIGN.value
        config.tts_model_type = "voice_design"
        model_arch_config = ARModelArchConfig(model_type=Model.QWEN3_TTS_VOICE_DESIGN.value)
    
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
