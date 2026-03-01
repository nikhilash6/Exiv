import torch
from torch import Tensor
import torch.nn.functional as F

import json
from typing import Dict, List

from exiv.components.enum import KSamplerType, SchedulerType
from exiv.components.cond_registry import preprocess_conds
from exiv.components.models.wan.constructor import get_wan_instance
from exiv.components.samplers.model_sampling import KSampler
from exiv.components.vae.base import get_vae
from exiv.model_utils.common_classes import AuxConditioning, AuxCondType, BatchedConditioning, Conditioning, ExtraCond, Latent
from exiv.model_utils.common_classes import ModelWrapper
from exiv.server.app_core import App, AppOutputType, Input, Output
from exiv.utils.common import null_func
from exiv.utils.device import MemoryManager
from exiv.utils.file import MediaProcessor
from exiv.utils.file_path import FilePathData, FilePaths
from exiv.utils.logging import app_logger

use_vae_tiling = False
vae_dtype = torch.float16

def main(**params):
    context = params.get("context")
    if context:
        context.start_anchor("Setup", steps=1)

    def progress_callback(progress_fraction, stage): 
        app_logger.debug(f"Percent: {progress_fraction}  -- Stage: {stage}")
        if context:
            context.progress(progress_fraction, "Processing", stage=stage) 
    
    initial_image = params.get("initial_image")
    final_image = params.get("final_image")
    positive = params.get("positive", "a nice transition")
    negative = params.get("negative", "bad quality")
    seed = params.get("seed", -1)
    steps = params.get("steps", 20)
    cfg = params.get("cfg", 6.0)
    sampler_name = params.get("sampler_name", KSamplerType.EULER.value)
    scheduler_name = params.get("scheduler_name", SchedulerType.SIMPLE.value)
    height = params.get("height", 720)
    width = params.get("width", 720)
    frame_count = params.get("frame_count", 81)
    
    if not initial_image or not final_image:
        raise ValueError("Both initial and final images must be provided.")

    if context: context.start_anchor("Preprocessing", steps=6)
    
    cur_model = "wan21_vace_1_3B_fp16.safetensors"
    model_path_data = FilePaths.get_path(filename=cur_model, file_type="checkpoint")
    wan_dit_model = get_wan_instance(model_path_data.path, model_path_data.url, force_dtype=torch.float16)
    model_wrapper = ModelWrapper(model=wan_dit_model)
    
    # build conditionals
    pos_cond = Conditioning(group_name="positive", input_metadata=positive)
    pos_cond.aux.append(AuxConditioning(
        type=AuxCondType.KEYFRAMES,
        input_metadata={
            0: initial_image,
            -1: final_image
        }
    ))
    neg_cond = Conditioning(group_name="negative", input_metadata=negative)
    cond_list = [pos_cond, neg_cond]
    
    batched_cond: BatchedConditioning = preprocess_conds(
        model_wrapper=model_wrapper,
        cond_list=cond_list,
        height=height, 
        width=width, 
        frame_count=frame_count,
        cfg=cfg,
        progress_callback=lambda percent, tag: context.progress(percent, tag) if context else null_func
    )
    
    wan_vae = get_vae(
        vae_type=model_wrapper.model.model_arch_config.default_vae_type,
        vae_dtype=vae_dtype,
        use_tiling=use_vae_tiling
    )
    extra_frames = batched_cond.conds[0].extra.get(ExtraCond.EXTRA_LATENT_FRAMES, 0)
    latent_format = model_wrapper.model.model_arch_config.latent_format
    latent = Latent()
    latent.encode_keyframe_condition( 
        width, 
        height,
        frame_count + (extra_frames * wan_vae.temporal_compression_ratio), 
        latent_format, 
        wan_vae,
    )
    t_c = wan_vae.temporal_compression_ratio
    del wan_vae
    MemoryManager.clear_memory()

    if context: context.start_anchor("Sampling", steps=12)
    main_sampler = KSampler(
        wrapped_model=model_wrapper,
        seed=seed,
        total_steps=steps,
        cfg=cfg,
        sampler_name=sampler_name,
        scheduler_name=scheduler_name,
        batched_conditioning=batched_cond,
        latent_image=latent
    )
    out = main_sampler.run_sampling(callback=lambda i, s: progress_callback(i, s))
    if extra_frames > 0: out = out[:, :, extra_frames*t_c:]
    wan_dit_model.to("cpu")
    wan_type = model_wrapper.model.model_arch_config.default_vae_type
    del wan_dit_model, model_wrapper
    MemoryManager.clear_memory()
    
    if context: context.start_anchor("Decoding", steps=1)
    out = out.to(dtype=vae_dtype)
    wan_vae = get_vae(
        vae_type=wan_type,
        vae_dtype=vae_dtype,
        use_tiling=use_vae_tiling
    )
    out = wan_vae.decode(out, (width, height, frame_count))
    output_paths = MediaProcessor.save_latents_to_media(out, start_frame=extra_frames)
    
    return {"1": output_paths[0]}

app = App(
    name="Simple Interpolation",
    inputs={
        'initial_image': Input(label="Initial Frame", type="str", default=""),
        'final_image': Input(label="Final Frame", type="str", default=""),
        'positive': Input(label="Positive Prompt", type="str", default="smooth transition"),
        'negative': Input(label="Negative Prompt", type="str", default="bad quality, static, deformed"),
        'seed': Input(label="Seed", type="number", default=-1),
        'steps': Input(label="Steps", type="number", default=20),
        'cfg': Input(label="CFG", type="number", default=6.0),
        'height': Input(label="Height", type="number", default=720),
        'width': Input(label="Width", type="number", default=720),
        'frame_count': Input(label="Frame Count", type="number", default=81),
    },
    outputs=[Output(id=1, type=AppOutputType.VIDEO.value)],
    handler=main
)

if __name__ == "__main__":
    app.run_standalone()