import torch
from torch import Tensor

from typing import Callable, Dict, List, Optional

from exiv.components.vae.base import VAEBase, get_vae
from exiv.utils.common import fix_frame_count
from exiv.utils.file import MediaProcessor
from exiv.utils.tensor import common_upscale

from .text_vision_encoder.common import TextEncoderOutput, VisionEncoderOutput
from .text_vision_encoder.encoder_base import VisionEncoder
from .text_vision_encoder.text_encoder import TextPipeline, create_text_pipeline
from .text_vision_encoder.vision_encoder import create_vision_encoder
from ..utils.device import MemoryManager
from ..model_utils.common_classes import AuxCondType, BatchedConditioning, Conditioning, ConditioningType, Latent, ModelWrapper

_PREPROCESSORS = {}     # global registry
def register_preprocessor(model_type):
    def decorator(func):
        if isinstance(model_type, list):
            for m in model_type: _PREPROCESSORS[m] = func
        else:
            _PREPROCESSORS[model_type] = func
        return func
    return decorator

def preprocess_conds(
    model_wrapper: ModelWrapper, 
    cond_list: List[Conditioning],
    **kwargs
):
    model_type = model_wrapper.model.model_type
    
    if kwargs.pop("cfg", 7) == 1:
        cond_list = list(filter(lambda c: c.group_name != "negative", cond_list))   # TODO: hardcoded string, should be fixed
    
    if model_type not in _PREPROCESSORS:
        raise NotImplementedError(
            f"No preprocessor found for {model_type}. "
            f"Did you forget to import the model's processor module?"
        )
    
    return _PREPROCESSORS[model_type](model_wrapper, cond_list, **kwargs)

def get_image_tensor(img: Tensor | str, height: int, width: int):
    if isinstance(img, Tensor): return img
    img = MediaProcessor.load_image_list(img)[0]
    img = common_upscale(img.unsqueeze(0), height, width)[0]
    return img

def get_text_embeddings(
    input_data: List[TextEncoderOutput] | List[str] | str,      # if encoder output is not provided, it is calculated based on the str
    te_model_filename = None,                                   # can be overriden to a custom model
    te_model_type = None
) -> List[TextEncoderOutput]:
    if not isinstance(input_data, list): input_data = [input_data] 
    if all(isinstance(i_d, TextEncoderOutput) for i_d in input_data): return input_data     # return as-is

    # load the model and generate the embedding
    te_pipeline: TextPipeline = create_text_pipeline(te_model_filename, te_model_type, dtype=torch.float16)
    te_pipeline.load_model()
    res = []
    for txt in input_data:
        embed: TextEncoderOutput = te_pipeline.encode(txt)
        res.append(embed)
    
    del te_pipeline
    MemoryManager.clear_memory()
    return res

def get_vision_embeddings(
    input_data: List[VisionEncoderOutput] | List[Tensor] | Tensor,
    ve_model_filename = None,
    ve_model_type = None,
) -> List[VisionEncoderOutput]:
    if not isinstance(input_data, list): input_data = [input_data] 
    if all(isinstance(i_d, VisionEncoderOutput) for i_d in input_data): return input_data     # return as-is
    
    ve_encoder: VisionEncoder = create_vision_encoder(
        filename=ve_model_filename, 
        model_type=ve_model_type, 
        dtype=torch.float16
    )
    ve_encoder.load_model()
    res = []
    for img in input_data:
        clip_embed: VisionEncoderOutput = ve_encoder.encode_image(img)
        res.append(clip_embed)
    del ve_encoder
    return res