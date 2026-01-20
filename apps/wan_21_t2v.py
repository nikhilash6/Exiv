import torch
from torch import Tensor
import torch.nn.functional as F

from typing import List

from exiv.components.enum import KSamplerType, SchedulerType, TextEncoderType, VAEType, VisionEncoderType
from exiv.components.cond_preprocess import get_text_embeddings, get_vision_embeddings
from exiv.components.latent_format import LatentFormat
from exiv.components.models.wan.constructor import get_wan_instance
from exiv.components.models.wan.main import Wan21ModelArchConfig
from exiv.components.samplers.model_sampling import KSampler
from exiv.components.text_vision_encoder.common import TextEncoderOutput, VisionEncoderOutput
from exiv.components.text_vision_encoder.vision_encoder import create_vision_encoder
from exiv.components.vae.base import get_vae
from exiv.components.vae.wan_vae import Wan21VAE
from exiv.model_patching.cache_hook import enable_step_caching
from exiv.model_patching.sliding_context_hook import BlendType, SlidingContextConfig, enable_sliding_context
from exiv.model_utils.common_classes import AuxCondType, AuxConditioning, BatchedConditioning, Conditioning, ConditioningType, Latent
from exiv.model_utils.common_classes import ModelWrapper
from exiv.server.app_core import App, AppOutputType, Input, Output
from exiv.utils.common import fix_frame_count
from exiv.utils.device import MemoryManager
from exiv.utils.file import MediaProcessor
from exiv.utils.file_path import FilePathData, FilePaths
from exiv.utils.tensor import common_upscale
from exiv.utils.logging import app_logger

use_vae_tiling = False
vae_dtype = torch.float16 # torch.bfloat16

def preprocess_wan_conditionals(
        model_wrapper: ModelWrapper,
        pos_embed: TextEncoderOutput, 
        neg_embed: TextEncoderOutput, 
        input_img: Tensor,
        clip_embed: VisionEncoderOutput,
        height: int, width: int, frame_count: int,
    ) -> tuple[BatchedConditioning, Latent]:
    
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

def main(**params):
    
    report_progress = params.get("report_progress")
    def progress_callback(progress_fraction, stage): 
        app_logger.debug(f"Percent: {progress_fraction}  -- Stage: {stage}")
        report_progress(progress_fraction, {"stage": stage, "status": "Processing"}) 
    
    # outside inputs
    positive_prompt = params.get("positive")
    negative_prompt = params.get("negative")
    seed = params.get("seed")
    steps = params.get("steps")
    cfg = params.get("cfg")
    sampler_name = params.get("sampler_name")
    scheduler_name = params.get("scheduler_name")
    
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
    
    # create a model wrapper
    # cur_model = "wan21_480p_i2v_fp16_14B.safetensors"
    cur_model = "wan21_1_3B.safetensors"
    model_path_data: FilePathData = FilePaths.get_path(filename=cur_model, file_type="checkpoint")
    wan_dit_model = get_wan_instance(model_path_data.path, model_path_data.url, force_dtype=torch.float16)
    enable_step_caching(wan_dit_model)
    # config = SlidingContextConfig(ctx_len=20, ctx_overlap=5, blend_type=BlendType.PYRAMIND.value)
    # enable_sliding_context(wan_dit_model, config=config)
    model_wrapper = ModelWrapper(model=wan_dit_model)
    
    # preprocess conditionals
    batched_cond, blank_latent = preprocess_wan_conditionals(
                                    model_wrapper,
                                    pos_embed, 
                                    neg_embed, 
                                    input_img,
                                    clip_embed,
                                    height, 
                                    width, 
                                    output_frame_count,
                                )
    
    MemoryManager.clear_memory()

    progress_callback(0.35, "Sampling loop")
    # the main sampling loop
    main_sampler = KSampler(
        wrapped_model=model_wrapper,
        seed=seed,
        steps=steps,
        cfg=cfg,
        sampler_name=sampler_name,
        scheduler_name=scheduler_name,
        batched_conditioning=batched_cond,
        latent_image=blank_latent
    )
    
    out = main_sampler.run_sampling(callback=lambda i, s: progress_callback(0.35 + round(i * 0.6, 2), s))
    wan_dit_model.to("cpu")
    del wan_dit_model, model_wrapper
    MemoryManager.clear_memory()
    
    progress_callback(0.95, "Decoding output latents")
    out = out.to(dtype=vae_dtype)
    wan_vae = get_vae(
        vae_type=VAEType.WAN21.value,
        vae_dtype=vae_dtype,
        use_tiling=use_vae_tiling
    )
    out = wan_vae.decode(out, (height, width, output_frame_count))
    output_paths = MediaProcessor.save_latents_to_media(out)
    
    return {"1": output_paths[0]}


app = App(
    name="Text to Video",
    inputs={
        'positive': Input(
            label="Positive Prompt",
            type="str",
            default="Cinematic anime style, medium close-up of a teenage boy with messy dark hair. 0-2s: The boy is looking down with a somber expression, his eyes shadowed. 2-4s: He slowly lifts his head to look directly into the camera, his expression shifting to one of sudden realization and determination, eyes widening with a subtle catchlight. Background is a soft-focus urban rooftop at sunset. Cel-shaded, vibrant colors, fluid character animation, high-quality rendering.",
            resizable=True,
        ),
        'negative': Input(
            label="Negative Prompt",
            type="str",
            default="bad image, blurry, low quality",
            resizable=True,
        ),
        'seed': Input(
            label="Seed",
            type="number",
            # default=-1,
            default=256347,
        ),
        'steps': Input(
            label="Steps",
            type="number",
            default=30,
            increment_controls=True,
            increment_step=2,
        ),
        'cfg': Input(
            label="CFG",
            type="number",
            default=6,
            increment_controls=True,
            increment_step=0.2,
        ),
        'sampler_name': Input(
            label="Sampler Name",
            type="select",
            options=KSamplerType.value_list(),
            default=KSamplerType.EULER.value,
        ),
        'scheduler_name': Input(
            label="Scheduler Name",
            type="select",
            options=SchedulerType.value_list(),
            default=SchedulerType.SIMPLE.value,
        )
    },
    outputs=[Output(id=1, type=AppOutputType.VIDEO.value)],
    handler=main
)

if __name__ == "__main__":
    app.run_standalone()