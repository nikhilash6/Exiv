import torch
from torch import Tensor

from exiv.components.enum import KSamplerType, SchedulerType, VAEType
from exiv.components.models.wan.constructor import get_wan_instance
from exiv.components.samplers.model_sampling import KSampler
from exiv.components.text_vision_encoder.common import TextEncoderOutput, VisionEncoderOutput
from exiv.components.text_vision_encoder.te_t5 import UMT5XXL
from exiv.components.text_vision_encoder.text_encoder import WanEncoder
from exiv.components.text_vision_encoder.vision_encoder import create_vision_encoder
from exiv.components.vae.base import get_vae
from exiv.components.vae.wan_vae import Wan21VAE
from exiv.model_patching.cache_hook import enable_step_caching
from exiv.model_utils.common_classes import AuxCondType, AuxConditioning, BatchedConditioning, Conditioning, ConditioningType, Latent
from exiv.model_utils.common_classes import ModelWrapper
from exiv.model_utils.helper_methods import move_model
from exiv.server.app_core import App, AppOutputType, Input, Output
from exiv.utils.device import OFFLOAD_DEVICE, VRAM_DEVICE, MemoryManager
from exiv.utils.file import MediaProcessor, ensure_model_availability
from exiv.utils.file_path import FilePathData, FilePaths
from exiv.utils.tensor import common_upscale
from exiv.utils.logging import app_logger

use_vae_tiling = False
vae_dtype = torch.bfloat16

def preprocess_wan_conditionals(
        pos_embed: TextEncoderOutput, 
        neg_embed: TextEncoderOutput, 
        clip_embed: VisionEncoderOutput, 
        input_img: Latent, 
        height: int, width: int, frame_count: int,
        batch_size: int
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
            data=clip_embed,
        )

        pos_cond.aux = [aux_clip]
        neg_cond.aux = [aux_clip]

    # creating concat conditioning
    if input_img is not None:
        wan_vae = get_vae(
            vae_type=VAEType.WAN21.value,
            vae_dtype=vae_dtype,
            use_tiling=use_vae_tiling
        )
        
        concat_cond = input_img.prepare_concat_latent(
            height, width, frame_count, wan_vae
        )
        
        pos_cond.concat = concat_cond
        neg_cond.concat = concat_cond


    batched_cond = BatchedConditioning(
        groups={
            "positive": [pos_cond],
            "negative": [neg_cond],
        },
        execution_order=["positive", "negative"]
    )
    
    # NOTE: batch_size repeat is removed from here, cross check
    return batched_cond, None

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
    input_img = MediaProcessor.load_image_list("./tests/test_utils/assets/media/boy_anime.jpg")[0]
    height, width, output_frame_count = 512, 512, 81
    
    # resizing img
    input_img = common_upscale(input_img.unsqueeze(0), height, width)   # (B, C, H, W)
    
    progress_callback(0.2, "Encoding prompts")
    # generate text embeddings
    cur_model = "umt5_xxl_fp16.safetensors"
    model_path_data: FilePathData = FilePaths.get_path(filename=cur_model, file_type="text_encoder")
    t5_xxl = UMT5XXL(model_path=model_path_data.path, dtype=torch.float16)
    wan_encoder = WanEncoder(t5_xxl=t5_xxl)
    wan_encoder.load_model(t5_xxl_download_url=model_path_data.url)
    pos_embed_dict = wan_encoder.encode(positive_prompt)
    neg_embed_dict = wan_encoder.encode(negative_prompt)
    del t5_xxl
    del wan_encoder
    
    progress_callback(0.3, "Generating CLIP embeddings")
    # generate img embeddings
    cur_model = "CLIP-ViT-H-fp16.safetensors"
    model_path_data: FilePathData = FilePaths.get_path(filename=cur_model, file_type="vision_encoder")
    clip_model = create_vision_encoder(model_path=model_path_data.path, download_url=model_path_data.url, dtype=torch.float16)
    clip_model.load_model()
    clip_embed_dict = clip_model.encode_image(input_img)
    del clip_model
    
    # preprocess conditionals
    batched_cond, blank_latent = preprocess_wan_conditionals(
                                            pos_embed_dict, 
                                            neg_embed_dict, 
                                            clip_embed_dict,
                                            input_img, 
                                            height, 
                                            width, 
                                            output_frame_count, 
                                            1
                                        )
    
    MemoryManager.clear_memory()
    
    # create a model wrapper
    cur_model = "wan21_1_3B.safetensors"
    model_path_data: FilePathData = FilePaths.get_path(filename=cur_model, file_type="checkpoint")
    wan_dit_model = get_wan_instance(model_path_data.path, model_path_data.url, force_dtype=torch.float16)
    enable_step_caching(wan_dit_model)
    model_wrapper = ModelWrapper(model=wan_dit_model)

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
            default="a dog running in the park",
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