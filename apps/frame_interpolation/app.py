import torch
from torch import Tensor
import torch.nn.functional as F

import json
from typing import Dict, List

from exiv.utils.logging import app_logger
from exiv.components import KSamplerType, SchedulerType, KSampler
from exiv.components.cond_registry import preprocess_conds
from exiv.components.models.wan.constructor import get_wan_instance
from exiv.components.vae.base import get_vae
from exiv.model_utils.common_classes import AuxConditioning, AuxCondType, BatchedConditioning, Conditioning, ExtraCond, Latent
from exiv.model_utils.common_classes import ModelWrapper
from exiv.server.app_core import App, AppOutputType, Input, Output
from exiv.utils.inputs import ModelInput
from exiv.utils.common import null_func
from exiv.utils.device import MemoryManager
from exiv.utils.file import MediaProcessor
from exiv.utils.file_path import FilePathData, FilePaths

use_vae_tiling = False
vae_dtype = torch.float16

def main(**params):
    context = params.get("context")
    keyframes = params.get("keyframes", [])
    
    if context:
        context.start_anchor("Setup", steps=1)

    def progress_callback(progress_fraction, stage): 
        app_logger.debug(f"Percent: {progress_fraction}  -- Stage: {stage}")
        if context:
            context.progress(progress_fraction, "Processing", stage=stage) 
    prompts = params.get("prompts", [])
    durations = params.get("durations", []) # durations in seconds
    negative = params.get("negative", "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走")
    seed = params.get("seed", -1)
    steps = params.get("steps", 20)
    cfg = params.get("cfg", 6.0)
    sampler_name = params.get("sampler_name", KSamplerType.EULER.value)
    scheduler_name = params.get("scheduler_name", SchedulerType.SIMPLE.value)
    height = params.get("height", 512)
    width = params.get("width", 512)
    
    # model overrides
    wan_model_name = params.get("wan_model_name", "wan21_vace_14B_fp16.safetensors")
    t5_model_name = params.get("t5_model_name")
    vae_model_name = params.get("vae_model_name")
    
    if len(keyframes) < 2:
        raise ValueError("At least two keyframes must be provided.")
    if len(prompts) < len(keyframes) - 1:
        # Pad prompts if necessary
        prompts = prompts + ["smooth transition"] * (len(keyframes) - 1 - len(prompts))
    if len(durations) < len(keyframes) - 1:
        # Pad durations if necessary
        durations = durations + [5] * (len(keyframes) - 1 - len(durations))

    if context: context.start_anchor("Loading Model", steps=2)
    
    model_path_data = FilePaths.get_path(filename=wan_model_name, file_type="checkpoint")
    wan_dit_model = get_wan_instance(model_path_data.path, model_path_data.url, force_dtype=torch.float16)
    model_wrapper = ModelWrapper(model=wan_dit_model)
    
    total_segments = len(keyframes) - 1
    all_segment_frames = []

    
    latent_format = model_wrapper.model.model_arch_config.latent_format

    if context: context.start_anchor("Processing Segments", steps=16)

    for i in range(total_segments):
        start_img = keyframes[i]
        end_img = keyframes[i+1]
        prompt = prompts[i]
        duration = durations[i]
        segment_frame_count = int(duration * 16) + 1 # 16fps, e.g. 5s -> 81 frames
        
        def chunk_progress_callback(p, s, stage_offset=0.0, stage_weight=1.0):
            local_p = (stage_offset + p * stage_weight) / total_segments
            global_p = (i / total_segments) + local_p
            progress_callback(global_p, s)
        
        # build conditionals
        pos_cond = Conditioning(group_name="positive", input_metadata=prompt)
        pos_cond.aux.append(AuxConditioning(
            type=AuxCondType.KEYFRAMES,
            input_metadata={
                0: start_img,
                -1: end_img
            }
        ))
        neg_cond = Conditioning(group_name="negative", input_metadata=negative)
        neg_cond.aux.append(AuxConditioning(
            type=AuxCondType.KEYFRAMES,
            input_metadata={
                0: start_img,
                -1: end_img
            }
        ))
        cond_list = [pos_cond, neg_cond]
        
        batched_cond: BatchedConditioning = preprocess_conds(
            model_wrapper=model_wrapper,
            cond_list=cond_list,
            height=height,
            width=width,
            frame_count=segment_frame_count,
            cfg=cfg,
            progress_callback=lambda p, s: chunk_progress_callback(p, s, stage_offset=0.0, stage_weight=0.2),
            t5_model_name=t5_model_name,
            vae_model_name=vae_model_name
        )
        wan_vae = get_vae(
            vae_type=model_wrapper.model.model_arch_config.default_vae_type,
            vae_dtype=vae_dtype,
            use_tiling=use_vae_tiling,
            override_filename=vae_model_name
        )
        
        latent = Latent()
        latent.encode_keyframe_condition( 
            width, 
            height,
            segment_frame_count, 
            latent_format, 
            wan_vae,
        )

        main_sampler = KSampler(
            wrapped_model=model_wrapper,
            seed=seed if i == 0 else (seed + i if seed != -1 else -1),
            total_steps=steps,
            cfg=cfg,
            sampler_name=sampler_name,
            scheduler_name=scheduler_name,
            batched_conditioning=batched_cond,
            latent_image=latent
        )
        out = main_sampler.run_sampling(callback=lambda p, s: chunk_progress_callback(p, s, stage_offset=0.2, stage_weight=0.8))
        out = out.to(dtype=vae_dtype)
        decoded = wan_vae.decode(out, (width, height, segment_frame_count))
        
        # decoded shape is likely [B, C, T, H, W]
        if i < total_segments - 1:
            # Overwrite last frame: take all but last frame of this segment
            all_segment_frames.append(decoded[:, :, :-1, :, :].detach().cpu())
        else:
            all_segment_frames.append(decoded.detach().cpu())
        
        del main_sampler, out, batched_cond, pos_cond, neg_cond, cond_list, latent, decoded, wan_vae
        MemoryManager.clear_memory()

    if context: context.start_anchor("Combining", steps=1)
    final_video_tensor = torch.cat(all_segment_frames, dim=2)
    
    output_paths = MediaProcessor.save_outputs(final_video_tensor)
    
    wan_dit_model.to("cpu")
    del wan_dit_model, model_wrapper
    MemoryManager.clear_memory()
    
    return {"1": output_paths[0]}

app = App(
    name="Frame Interpolation",
    inputs={
        'keyframes': Input(label="Keyframes", type="list", default=[]),
        'prompts': Input(label="Prompts", type="list", default=[]),
        'durations': Input(label="Durations", type="list", default=[]),
        'seed': Input(label="Seed", type="number", default=-1),
        'steps': Input(label="Steps", type="number", default=20),
        'cfg': Input(label="CFG", type="number", default=6.0),
        'height': Input(label="Height", type="number", default=512),
        'width': Input(label="Width", type="number", default=512),
        'wan_model_name': ModelInput(label="Wan Model", categories=["checkpoint"], default="wan21_vace_14B_fp16.safetensors"),
        't5_model_name': ModelInput(label="T5 Text Encoder", categories=["text_encoder"], default="umt5_xxl_fp16.safetensors"),
        'vae_model_name': ModelInput(label="VAE Model", categories=["vae"], default="wan_2_1_vae.safetensors"),
    },
    outputs=[Output(id=1, type=AppOutputType.VIDEO.value)],
    handler=main
)

if __name__ == "__main__":
    app.run_standalone()