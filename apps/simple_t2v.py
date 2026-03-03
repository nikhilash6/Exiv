import torch
import os
from exiv.utils.logging import app_logger
from exiv.components import KSamplerType, SchedulerType, KSampler
from exiv.components.cond_registry import preprocess_conds
from exiv.components.models.wan.constructor import get_wan_instance
from exiv.components.samplers.utils import normalize_seed
from exiv.components.vae.base import get_vae
from exiv.model_utils.common_classes import Conditioning, ModelWrapper, Latent
from exiv.server.app_core import App, AppOutputType, Input, Output
from exiv.utils.device import MemoryManager
from exiv.utils.file import MediaProcessor
from exiv.utils.file_path import FilePaths

def main(**params):
    # Extract parameters
    prompt = params.get("prompt")
    width = params.get("width", 512)
    height = params.get("height", 512)
    
    # Hardcoded/Default values
    negative_prompt = "(worst quality, low quality, normal quality, lowres, low resolution, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, jpeg artifacts,\
     signature, watermark, username, blurry, artist name, deformed, disfigured, poorly drawn face, mutation, mutated, extra limbs, extra legs, extra arms, fused fingers, too many fingers,\
     long neck, cross-eyed, mutated hands, polar lowres, bad body, bad proportions, gross proportions, malformed limbs, missing arms, missing legs, extra foot, out of frame, body out of\
     frame, canvas boundary, grainy, tiling, poorly drawn hands, poorly drawn feet, out of focus, duplicate, morbidity, mutilation, trite, logo, watermark, banner)"
    seed = normalize_seed(-1)
    steps = 20
    cfg = 6.0
    frame_count = 81
    
    app_logger.info(f"Starting T2V with prompt: {prompt}")

    # 1. Load Model (Wan2.1 1.3B)
    model_name = "wan21_1_3B.safetensors"
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
    context = params.get("context")
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
    def progress_callback(progress_fraction, stage):
        if context:
            context.progress(progress_fraction, "Processing", stage=stage)
        else:
            print(f"Sampling progress {progress_fraction*100:.1f}% - {stage}")

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
    name="Simple Text to Video",
    description="Lightweight T2V, mostly for quickstart demo",
    inputs={
        'prompt': Input(
            label="Prompt", 
            type="text", 
            default="A dog running in the park, cinematic lighting, 4k"
        ),
        'width': Input(label="Width", type="number", default=512),
        'height': Input(label="Height", type="number", default=512),
    },
    outputs=[Output(id=1, type=AppOutputType.VIDEO.value)],
    handler=main
)

if __name__ == "__main__":
    app.run_standalone()
