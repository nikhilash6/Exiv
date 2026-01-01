import os

from exiv.components.text_vision_encoder.common import TextEncoderOutput, VisionEncoderOutput
from exiv.components.vae.base import get_vae
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import torch
from torch import Tensor

from exiv.components.enum import KSamplerType, SchedulerType, VAEType
from exiv.components.models.wan.constructor import get_wan_instance
from exiv.components.models.wan.main import Wan22ModelArchConfig
from exiv.components.samplers.model_sampling import KSampler
from exiv.components.text_vision_encoder.te_t5 import UMT5XXL
from exiv.components.text_vision_encoder.text_encoder import WanEncoder
from exiv.components.text_vision_encoder.vision_encoder import create_vision_encoder
from exiv.components.vae.wan_vae import Wan21VAE
from exiv.components.vae.wan_vae22 import Wan22VAE
from exiv.model_patching.cache_hook import enable_step_caching
from exiv.model_utils.common_classes import Conditioning, BatchedConditioning, ConditioningType, Latent
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
        input_img: Tensor, 
        height, width, frame_count, batch_size
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
    
    batched_cond = BatchedConditioning(
        groups={
            "positive": [pos_cond],
            "negative": [neg_cond],
        },
        execution_order=["positive", "negative"]
    )
    
    # encode the entire sequence
    wan_vae = get_vae(
        vae_type=VAEType.WAN22.value,
        vae_dtype=vae_dtype,
        use_tiling=use_vae_tiling
    )
    
    # empty tensor
    blank_latent = Latent()
    compression_factor = wan_vae.spatial_compression_ratio
    latent = torch.zeros([1, 48, ((frame_count - 1) // 4) + 1, height // compression_factor, width // compression_factor], device=VRAM_DEVICE)     # B, C, T, H, W

    # NOTE: this is a divergence from the original repo to aim for 100% perfect first frame
    # insteads of passing the image as a hint, we pass it directly attached to the latent with mask 
    # that tells not to touch the attached image part
    if input_img is None:
        blank_latent.samples = latent
        return batched_cond, blank_latent

    b = latent.shape[0]
    h = latent.shape[-2]
    w = latent.shape[-1]
    t = ((frame_count - 1) // 4) + 1
    mask = torch.ones([b, 1, t, h, w], device=VRAM_DEVICE)
    
    if input_img is not None:
        # B, C, H, W -> B, C, 1, H, W
        input_img = input_img.unsqueeze(2)
        input_img = input_img.to(vae_dtype)
        input_img = input_img.to(VRAM_DEVICE)
        latent_temp = wan_vae.encode(input_img)                 # requires  (B, C, T, H, W)
        latent_temp = latent_temp.to(vae_dtype)
        latent[:, :, :latent_temp.shape[2]] = latent_temp
        mask[:, :, :latent_temp.shape[2]] *= 0.0                # setting mask to zero for the first concatenated latent

    latent_format = Wan22ModelArchConfig().latent_format
    latent = latent_format.process_out(latent) * mask + latent * (1.0 - mask)
    blank_latent.samples = latent.repeat((batch_size, ) + (1,) * (latent.ndim - 1))
    blank_latent.samples = blank_latent.samples.to(vae_dtype)                           # TODO: auto convert these (should be auto converting ?)
    blank_latent.noise_mask = mask.repeat((batch_size, ) + (1,) * (mask.ndim - 1))
    blank_latent.noise_mask = blank_latent.noise_mask.to(vae_dtype)
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
    input_img = MediaProcessor.load_image_list("./tests/test_utils/assets/media/dog_realistic.jpg")[0]
    height, width, output_frame_count = 480, 832, 81
    
    # resizing img
    input_img = common_upscale(input_img.unsqueeze(0), width, height)   # (B, C, H, W)
    
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
    MemoryManager.clear_memory()
    
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
    cur_model = "wan22_5B_ti2v_fp16"
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
    
    # from torch_tracer import TorchTracer
    
    # with TorchTracer("./exiv_2.pkl"):
    out = main_sampler.run_sampling(callback=lambda i, s: progress_callback(0.35 + round(i * 0.6, 2), s))
    wan_dit_model.to("cpu")
    del wan_dit_model, model_wrapper
    MemoryManager.clear_memory()
    
    progress_callback(0.95, "Decoding output latents")
    wan_vae = get_vae(
        vae_type=VAEType.WAN22.value,
        vae_dtype=vae_dtype,
        use_tiling=use_vae_tiling
    )
    out = out.to(vae_dtype)
    out = wan_vae.decode(out, (height, width, output_frame_count))
    
    metadata = {
        "seed": seed,
        "positive_prompt": positive_prompt,
        "negative_prompt": negative_prompt,
        "resolution": f"{width}x{height}",
        "frame_count": output_frame_count,
        "steps": steps,
        "cfg": cfg,
        "sampler_name": sampler_name,
        "scheduler_name": scheduler_name
    }
    output_paths = MediaProcessor.save_latents_to_media(out, metadata=metadata)
    
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
            default=20,
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