import torch
from torch import Tensor
import torch.nn.functional as F

from exiv.components.enum import KSamplerType, SchedulerType, VAEType
from exiv.components.cond_registry import preprocess_conds
from exiv.components.models.wan.constructor import get_wan_instance
from exiv.components.samplers.model_sampling import KSampler
from exiv.components.vae.base import get_vae
from exiv.model_patching.cache_hook import enable_step_caching
from exiv.model_utils.common_classes import AuxConditioning, AuxCondType, Conditioning, BatchedConditioning, ConditioningType, Latent, ModelWrapper
from exiv.server.app_core import App, AppOutputType, Input, Output
from exiv.utils.device import MemoryManager
from exiv.utils.file import MediaProcessor
from exiv.utils.file_path import FilePaths
from exiv.utils.logging import app_logger
from exiv.utils.common import fix_frame_count

VAE_DTYPE = torch.float16
USE_VAE_TILING = True

class WanAnimateMode:
    ANIMATION = "animation"
    REPLACEMENT = "replacement"

def main(**params):
    context = params.get("context")
    if context:
        context.start_anchor("Setup", steps=1)

    def progress_callback(progress_fraction, stage): 
        app_logger.debug(f"Percent: {progress_fraction}  -- Stage: {stage}")
        if context:
            context.progress(progress_fraction, "Processing", stage=stage) 
    
    pos_prompt = params.get("positive", "a man dancing in a studio, high quality")
    neg_prompt = params.get("negative", "bad quality, blurry, distorted, disfigured")
    seed = params.get("seed", 42)
    steps = params.get("steps", 20)
    cfg = params.get("cfg", 6.0)
    sampler_name = params.get("sampler_name", KSamplerType.EULER.value)
    scheduler_name = params.get("scheduler_name", SchedulerType.SIMPLE.value)
    
    ref_img_path = params.get("reference_image", "ref_image.png")
    pose_video_path = params.get("pose_video", "pose.mp4")
    face_video_path = params.get("face_video", "face.mp4")
    mode = params.get("mode", WanAnimateMode.ANIMATION)
    
    bg_video_path = params.get("background_video")
    mask_video_path = params.get("mask_video")
    
    if mode == WanAnimateMode.REPLACEMENT and (not bg_video_path or not mask_video_path):
        raise ValueError("Replacement mode requires Background Video and Mask Video.")
    
    height, width, frame_count = 640, 640, 81
    frame_count = fix_frame_count(frame_count, 4)
    
    if context: context.start_anchor("Preprocessing", steps=6)
    
    # model_name = "wan22_animate_14b_bf16"
    model_name = "wan22_animate_14b_fp8_e4m3_scaled"
    model_path_data = FilePaths.get_path(filename=model_name, file_type="checkpoint")
    wan_dit_model = get_wan_instance(model_path_data.path, model_path_data.url, force_dtype=torch.float16)
    model_wrapper = ModelWrapper(model=wan_dit_model)
    # enable_step_caching(wan_dit_model)
    
    def create_cond(group_name, prompt):
        cond = Conditioning(
            group_name=group_name,
            input_metadata=prompt,
            extra={
                "animate_mode": mode,
                "background_video_path": bg_video_path,
                "character_mask_path": mask_video_path
            }
        )
        cond.aux = [
            AuxConditioning(type=AuxCondType.REF_LATENT, input_metadata=ref_img_path),
            AuxConditioning(type=AuxCondType.VISUAL_EMBEDDING, input_metadata=ref_img_path),
            AuxConditioning(type=AuxCondType.POSE_LATENTS, input_metadata=pose_video_path),
            AuxConditioning(type=AuxCondType.FACE_PIXEL_VALUES, input_metadata=face_video_path)
        ]
        return cond

    cond_list = [create_cond("positive", pos_prompt), create_cond("negative", neg_prompt)]
    batched_cond: BatchedConditioning = preprocess_conds(
        model_wrapper=model_wrapper,
        cond_list=cond_list,
        height=height, 
        width=width, 
        frame_count=frame_count,
        cfg=cfg,
        progress_callback=lambda p, s: progress_callback(0.1 + p * 0.5, s)
    )
    
    wan_vae = get_vae(VAEType.WAN21.value, VAE_DTYPE, USE_VAE_TILING)
    latent_format = model_wrapper.model.model_arch_config.latent_format
    latent = Latent() 
    latent.encode_keyframe_condition( 
        width, 
        height,
        frame_count, 
        latent_format, 
        wan_vae,
    )
    
    if context: context.start_anchor("Sampling", steps=12)
    sampler = KSampler(
        wrapped_model=model_wrapper,
        seed=seed,
        total_steps=steps,
        cfg=cfg,
        sampler_name=sampler_name,
        scheduler_name=scheduler_name,
        batched_conditioning=batched_cond,
        latent_image=latent
    )
    
    out = sampler.run_sampling(callback=lambda i, s: progress_callback(i, s))
    
    wan_dit_model.to("cpu")
    del wan_dit_model, model_wrapper
    MemoryManager.clear_memory()
    
    if context: context.start_anchor("Decoding", steps=1)
    out = out.to(dtype=VAE_DTYPE)
    vae = get_vae(VAEType.WAN21.value, VAE_DTYPE, USE_VAE_TILING)
    out = vae.decode(out, (width, height, frame_count))
    # hardcoding rn will change later
    if out.shape[2] > 8: out = out[:, :, 8:]
    
    metadata = {
        "positive": pos_prompt, "seed": seed, "mode": mode,
        "model": "Wan2.2 Animate"
    }
    out_paths = MediaProcessor.save_latents_to_media(out, metadata=metadata)
    return {"1": out_paths[0]}

app = App(
    name="Wan Animate",
    inputs={
        'positive': Input(label="Positive Prompt", type="str", default="a girl talking", resizable=True),
        'negative': Input(label="Negative Prompt", type="str", default="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走", resizable=True),
        'seed': Input(label="Seed", type="number", default=-1),
        'steps': Input(label="Steps", type="number", default=20),
        'cfg': Input(label="CFG", type="number", default=6.0, increment_step=0.1),
        'sampler_name': Input(label="Sampler", type="select", options=KSamplerType.value_list(), default=KSamplerType.EULER.value),
        'scheduler_name': Input(label="Scheduler", type="select", options=SchedulerType.value_list(), default=SchedulerType.SIMPLE.value),
        
        # Media
        'reference_image': Input(label="Reference Image", type="str", default="ref_image.png"),
        'pose_video': Input(label="Pose Video", type="str", default="pose.mp4"),
        'face_video': Input(label="Face Video", type="str", default="face.mp4"),
        
        # Mode
        'mode': Input(label="Mode", type="select", options=[WanAnimateMode.ANIMATION, WanAnimateMode.REPLACEMENT], default=WanAnimateMode.REPLACEMENT),
        'background_video': Input(label="Background Video (Replacement)", type="str", default="background.mp4"),
        'mask_video': Input(label="Mask Video (Replacement)", type="str", default="character.mp4"),
    },
    outputs=[Output(id=1, type=AppOutputType.VIDEO.value)],
    handler=main
)

if __name__ == "__main__":
    app.run_standalone()