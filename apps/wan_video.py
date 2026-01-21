import json
import torch
from torch import Tensor
import torch.nn.functional as F

from typing import List

from exiv.components.enum import KSamplerType, SchedulerType, TextEncoderType, VAEType, VisionEncoderType
from exiv.components.cond_preprocess import get_text_embeddings, get_vision_embeddings, preprocess_conds
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
from exiv.server.app_core import App, AppOutputType, Input, Output, TaskContext
from exiv.utils.common import fix_frame_count
from exiv.utils.device import MemoryManager
from exiv.utils.file import MediaProcessor
from exiv.utils.file_path import FilePathData, FilePaths
from exiv.utils.tensor import common_upscale
from exiv.utils.logging import app_logger
from apps.utils.defaults import get_default_cond

use_vae_tiling = False
vae_dtype = torch.float16 # torch.bfloat16

def main(**params):
    
    context = params.get("context")
    if context:
        context.start_anchor("Setup", steps=1) # 5%

    def progress_callback(progress_fraction, stage): 
        app_logger.debug(f"Percent: {progress_fraction}  -- Stage: {stage}")
        if context:
            context.progress(progress_fraction, "Processing", stage=stage) 
    
    # main settingss
    conditions = params.get("conditions")
    cond_list: List[Conditioning] = []
    for c in conditions:
        if c_obj:=Conditioning.from_json(c) is not None:
            cond_list.append(c_obj)
        else:
            app_logger.warning("Malformed cond dict, aborting process")
            
    seed = params.get("seed")
    steps = params.get("steps")
    cfg = params.get("cfg")
    sampler_name = params.get("sampler_name")
    scheduler_name = params.get("scheduler_name")
    height = params.get("height")
    width = params.get("width")
    frame_count = params.get("frame_count")

    if context: context.start_anchor("Preprocessing", steps=6) # 30%
    
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
    batched_cond, blank_latent = preprocess_conds(
                                    model_wrapper,
                                    cond_list,
                                    height, 
                                    width, 
                                    frame_count,
                                )
    
    MemoryManager.clear_memory()

    if context: context.start_anchor("Sampling", steps=12) # 60%
    
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
    
    # callback now returns local progress (0 - 1.0)
    out = main_sampler.run_sampling(callback=lambda i, s: progress_callback(i, s))
    wan_dit_model.to("cpu")
    del wan_dit_model, model_wrapper
    MemoryManager.clear_memory()
    
    if context: context.start_anchor("Decoding", steps=1) # 5%
    out = out.to(dtype=vae_dtype)
    wan_vae = get_vae(
        vae_type=VAEType.WAN21.value,
        vae_dtype=vae_dtype,
        use_tiling=use_vae_tiling
    )
    out = wan_vae.decode(out, (height, width, frame_count))
    output_paths = MediaProcessor.save_latents_to_media(out)
    
    return {"1": output_paths[0]}

DEFAULT_CONDS = get_default_cond()
app = App(
    name="Text to Video",
    inputs={
        'conditions': Input(label="Conditions (JSON)", type="json", default=DEFAULT_CONDS,),
        'seed': Input(label="Seed", type="number", default=256347,),
        'steps': Input(label="Steps", type="number", default=30, increment_controls=True, increment_step=2,),
        'cfg': Input(label="CFG", type="number", default=6, increment_controls=True, increment_step=0.2,),
        'sampler_name': Input(label="Sampler Name", type="select", options=KSamplerType.value_list(), \
            default=KSamplerType.EULER.value,),
        'scheduler_name': Input(label="Scheduler Name", type="select", options=SchedulerType.value_list(), \
            default=SchedulerType.SIMPLE.value,),
        'height': Input(label="Height", type="number", default=512),
        'width': Input(label="Width", type="number", default=512),
        'frame_count': Input(label="Frame Count", type="number", default=81),
    },
    outputs=[Output(id=1, type=AppOutputType.VIDEO.value)],
    handler=main
)

if __name__ == "__main__":
    app.run_standalone()