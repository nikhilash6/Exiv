import torch
import os
from exiv.components.enum import KSamplerType, SchedulerType
from exiv.components.cond_registry import preprocess_conds
from exiv.components.models.wan.constructor import get_wan_instance
from exiv.components.samplers.model_sampling import KSampler
from exiv.components.vae.base import get_vae
from exiv.model_utils.common_classes import Conditioning, ModelWrapper, Latent
from exiv.server.app_core import App, AppOutputType, Input, Output
from exiv.utils.device import MemoryManager
from exiv.utils.file import MediaProcessor
from exiv.utils.file_path import FilePaths
from exiv.utils.logging import app_logger

def main(**params):
    # Extract parameters
    prompt = params.get("prompt")
    negative_prompt = params.get("negative_prompt", "bad quality, blurry, low res")
    seed = params.get("seed", -1)
    steps = params.get("steps", 20)
    cfg = params.get("cfg", 6.0)
    width = params.get("width", 832)
    height = params.get("height", 480)
    frame_count = params.get("frame_count", 81)
    
    app_logger.info(f"Starting T2V with prompt: {prompt}")

    # 1. Load Model (Wan2.1 1.3B)
    # Using the vace version as it's the 1.3B model available
    model_name = "wan21_vace_1_3B_fp16.safetensors"
    model_path_data = FilePaths.get_path(filename=model_name, file_type="checkpoint")
    
    app_logger.info(f"Loading model: {model_name}")
    wan_dit_model = get_wan_instance(model_path_data.path, model_path_data.url, force_dtype=torch.float16)
    model_wrapper = ModelWrapper(model=wan_dit_model)

    # 2. Preprocess Conditionals
    app_logger.info("Preprocessing conditionals...")
    cond_list = [
        Conditioning(group_name="positive", input_metadata=prompt),
        Conditioning(group_name="negative", input_metadata=negative_prompt)
    ]
    
    batched_cond = preprocess_conds(
        model_wrapper=model_wrapper,
        cond_list=cond_list,
        height=height,
        width=width,
        frame_count=frame_count,
        cfg=cfg
    )

    # 3. Setup Latent
    app_logger.info("Setting up latent...")
    vae_type = model_wrapper.model.model_arch_config.default_vae_type
    wan_vae = get_vae(vae_type=vae_type, vae_dtype=torch.float16)
    
    latent = Latent()
    latent.encode_keyframe_condition(
        width, 
        height, 
        frame_count, 
        model_wrapper.model.model_arch_config.latent_format, 
        wan_vae
    )
    
    MemoryManager.clear_memory()

    # 4. Sampling
    app_logger.info(f"Starting sampling for {steps} steps...")
    sampler = KSampler(
        wrapped_model=model_wrapper,
        seed=seed,
        total_steps=steps,
        cfg=cfg,
        sampler_name=KSamplerType.EULER.value,
        scheduler_name=SchedulerType.SIMPLE.value,
        batched_conditioning=batched_cond,
        latent_image=latent
    )
    
    # Simple callback for progress
    def progress_callback(i, s):
        print(f"Sampling step {i}/{steps} - {s}")

    out = sampler.run_sampling(callback=progress_callback)
    
    # Move model to CPU to free up VRAM for decoding
    wan_dit_model.to("cpu")
    MemoryManager.clear_memory()
    
    # 5. Decode
    app_logger.info("Decoding latents to video...")
    out = out.to(dtype=torch.float16)
    video_out = wan_vae.decode(out, (width, height, frame_count))
    
    # 6. Save
    output_paths = MediaProcessor.save_latents_to_media(video_out)
    app_logger.info(f"Video saved to: {output_paths[0]}")
    
    return {"1": output_paths[0]}

app = App(
    name="Wan T2V",
    inputs={
        'prompt': Input(
            label="Prompt", 
            type="text", 
            default="A stylish woman walking down a Tokyo street with neon lights, cinematic lighting, 4k"
        ),
        'negative_prompt': Input(
            label="Negative Prompt", 
            type="text", 
            default="bad quality, blurry, low res, static, flickering"
        ),
        'seed': Input(label="Seed", type="number", default=-1),
        'steps': Input(label="Steps", type="number", default=20),
        'cfg': Input(label="CFG", type="number", default=6.0),
        'width': Input(label="Width", type="number", default=832),
        'height': Input(label="Height", type="number", default=480),
        'frame_count': Input(label="Frame Count", type="number", default=81),
    },
    outputs=[Output(id=1, type=AppOutputType.VIDEO.value)],
    handler=main
)

if __name__ == "__main__":
    app.run_standalone()
