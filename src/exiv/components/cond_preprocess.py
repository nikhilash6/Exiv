import torch
from torch import Tensor

from typing import List

from ..components.text_vision_encoder.common import TextEncoderOutput, VisionEncoderOutput
from ..components.text_vision_encoder.encoder_base import VisionEncoder
from ..components.text_vision_encoder.text_encoder import TextPipeline, create_text_pipeline
from ..components.text_vision_encoder.vision_encoder import create_vision_encoder
from ..utils.device import MemoryManager
from ..model_utils.common_classes import BatchedConditioning, Conditioning, ConditioningType

_PREPROCESSORS = {}     # global registry
def register_preprocessor(model_type):
    def decorator(func):
        if isinstance(model_type, list):
            for m in model_type: _PREPROCESSORS[m] = func
        else:
            _PREPROCESSORS[model_type] = func
        return func
    return decorator

def preprocess_conds(wrapper, positive, negative, resource_config=None, **kwargs):
    model_type = wrapper.model.model_type
    
    if model_type not in _PREPROCESSORS:
        raise NotImplementedError(
            f"No preprocessor found for {model_type}. "
            f"Did you forget to import the model's processor module?"
        )
    
    return _PREPROCESSORS[model_type](wrapper, positive, negative, resource_config, **kwargs)

def create_base_batch(pos_data, neg_data):
    p = Conditioning(pos_data.last_hidden_state, ConditioningType.EMBEDDING, pos_data.extra.copy())
    n = Conditioning(neg_data.last_hidden_state, ConditioningType.EMBEDDING, neg_data.extra.copy())
    return BatchedConditioning({"positive": [p], "negative": [n]}, ["positive", "negative"]), p, n

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