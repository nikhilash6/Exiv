import torch
from torch import Tensor

from typing import Callable, Dict, List

from ..components.text_vision_encoder.common import TextEncoderOutput, VisionEncoderOutput
from ..components.text_vision_encoder.encoder_base import VisionEncoder
from ..components.text_vision_encoder.text_encoder import TextPipeline, create_text_pipeline
from ..components.text_vision_encoder.vision_encoder import create_vision_encoder
from ..utils.device import MemoryManager
from ..model_utils.common_classes import BatchedConditioning, Conditioning, ConditioningType, Latent, ModelWrapper

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

def preprocess_wan_conditionals(
        model_wrapper: ModelWrapper,
        conditions: Dict,
        input_img: Tensor,
        clip_embed: VisionEncoderOutput,
        height: int, width: int, frame_count: int,
        progress_callback: Callable = None
) -> tuple[BatchedConditioning, Latent]:
    
    progress_callback(0.1, "Loading Images")
    height, width, output_frame_count = 512, 512, 81
    output_frame_count = fix_frame_count(output_frame_count)
    input_img = MediaProcessor.load_image_list("./tests/test_utils/assets/media/boy_anime.jpg")[0]
    input_img = common_upscale(input_img.unsqueeze(0), height, width)[0]
    
    progress_callback(0.2, "Encoding prompts")
    # generate text embeddings
    te_embeds: List[TextEncoderOutput] = get_text_embeddings([positive_prompt, negative_prompt], te_model_type=TextEncoderType.UMT5_XXL.value)
    pos_embed, neg_embed = te_embeds[0], te_embeds[1]
    
    progress_callback(0.3, "Generating CLIP embeddings")
    # generate img embeddings
    clip_embed: VisionEncoderOutput = get_vision_embeddings(input_img, ve_model_type=VisionEncoderType.CLIP_H.value)[0]
    
    
    pos_cond = Conditioning(
        data=pos_embed.last_hidden_state,
        type=ConditioningType.EMBEDDING,
        extra=pos_embed.extra.copy()
    )
    
    neg_cond = Conditioning(
        data=neg_embed.last_hidden_state,
        type=ConditioningType.EMBEDDING,
        extra=neg_embed.extra.copy()
    )
    
    # creating aux visual embedding
    if clip_embed is not None:
        aux_clip = AuxConditioning(
            type=AuxCondType.VISUAL_EMBEDDING,
            data=clip_embed.intermediate_hidden_states,
        )

        pos_cond.aux = [aux_clip]
        neg_cond.aux = [aux_clip]

    wan_vae = get_vae(
        vae_type=VAEType.WAN21.value,
        vae_dtype=vae_dtype,
        use_tiling=use_vae_tiling
    )
    
    latent_format = model_wrapper.model.model_arch_config.latent_format
    blank_latent = Latent()
    blank_latent.encode_keyframe_condition(
        height, 
        width, 
        frame_count, 
        latent_format, 
        wan_vae,
    )
    
    if input_img is not None:
        data = model_wrapper.model.model_arch_config.get_ref_latent(
            start_image=input_img,
            vae=wan_vae,
            length=frame_count,
            width=width,
            height=height,
        )
        ref_latent = AuxConditioning(
            type=AuxCondType.REF_LATENT,
            data=data,
        )
        pos_cond.aux.append(ref_latent)
        neg_cond.aux.append(ref_latent)

    batched_cond = BatchedConditioning(
        groups={
            "positive": [pos_cond],
            "negative": [neg_cond],
        },
        execution_order=["positive", "negative"]
    )
    
    return batched_cond, blank_latent


def create_base_batch(pos_data, neg_data):
    p = Conditioning(data=pos_data.last_hidden_state, type=ConditioningType.EMBEDDING, extra=pos_data.extra.copy())
    n = Conditioning(data=neg_data.last_hidden_state, type=ConditioningType.EMBEDDING, extra=neg_data.extra.copy())
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