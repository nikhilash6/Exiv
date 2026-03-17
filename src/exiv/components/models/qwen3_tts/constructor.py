import torch

from transformers import AutoTokenizer

from .core.talker_base import Qwen3TTSForConditionalGeneration
from .core.config import Qwen3TTSConfig
from .core.text_prorcessor import Qwen3TTSTextProcessor
from ...audio_encoders.qwen3_tts_tokenizer import Qwen3TTSTokenizer
from ....model_utils.autoregressive_model_mixin import ARModelArchConfig
from ....components.enum import Model
from ....config import LOADING_MODE
from ....model_utils.helper_methods import get_state_dict
from ....utils.file import ensure_model_availability
from ....utils.logging import app_logger

def get_qwen3_tts_instance(
    model_path,
    download_url=None,
    force_load_mode=LOADING_MODE.NORMAL_LOAD.value,
    force_dtype=None,
):
    app_logger.info("Resolving model availability...")
    model_path = ensure_model_availability(model_path, download_url)
    
    # TODO: update this with proper logic, right now hardcoding to voice_design
    # TODO: add shape detection for various layers
    cls = Qwen3TTSForConditionalGeneration
    dict_dtype = torch.float16
    config = Qwen3TTSConfig()
    config.model_type = Model.QWEN3_TTS_VOICE_DESIGN.value
    model_arch_config = ARModelArchConfig(model_type=Model.QWEN3_TTS_VOICE_DESIGN.value)
    
    dtype = force_dtype or dict_dtype
    app_logger.info(f"Initializing {cls.__name__}...")
    qwen3_tts = cls(**config, force_load_mode=force_load_mode, dtype=dtype)
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
        speech_tokenizer = Qwen3TTSTokenizer.from_pretrained("Qwen/Qwen3-TTS-Tokenizer-12Hz", dtype=force_dtype)
        qwen3_tts.speech_tokenizer = speech_tokenizer
    except Exception as e:
        app_logger.warning(f"Could not load Speech Tokenizer: {e}")
        speech_tokenizer = None

    return qwen3_tts, text_tokenizer, speech_tokenizer
