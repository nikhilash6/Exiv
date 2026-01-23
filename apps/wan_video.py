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
from exiv.model_patching.common import apply_hook_json
from exiv.model_utils.common_classes import Conditioning, Latent
from exiv.model_utils.common_classes import ModelWrapper
from exiv.server.app_core import App, AppOutputType, Input, Output
from exiv.utils.common import null_func
from exiv.utils.device import MemoryManager
from exiv.utils.file import MediaProcessor
from exiv.utils.file_path import FilePathData, FilePaths
from exiv.utils.logging import app_logger
from utils.defaults import get_dummy_cond, get_dummy_hook, get_dummy_latent

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
        if (c_obj:=Conditioning.from_json(c)) is not None:
            cond_list.append(c_obj)
        else:
            raise RuntimeError("Malformed cond json, aborting process")
    hooks = params.get("hooks")
    latent_json = params.get("latent")
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
    # cur_model = "wan21_1_3B.safetensors"
    cur_model = "wan22_5B_ti2v_fp16"
    model_path_data: FilePathData = FilePaths.get_path(filename=cur_model, file_type="checkpoint")
    wan_dit_model = get_wan_instance(model_path_data.path, model_path_data.url, force_dtype=torch.float16)
    apply_hook_json(wan_dit_model, hooks)
    model_wrapper = ModelWrapper(model=wan_dit_model)
    
    # input latent
    wan_vae = get_vae(
        vae_type=model_wrapper.model.model_arch_config.default_vae_type,
        vae_dtype=vae_dtype,
        use_tiling=use_vae_tiling
    )
    latent_format = model_wrapper.model.model_arch_config.latent_format
    latent: Latent = Latent.from_json(latent_json=latent_json)
    latent.encode_keyframe_condition( 
        width, 
        height,
        frame_count, 
        latent_format, 
        wan_vae,
    )
    del wan_vae
    
    # preprocess conditionals
    batched_cond = preprocess_conds(
                    model_wrapper=model_wrapper,
                    cond_list=cond_list,
                    height=height, 
                    width=width, 
                    frame_count=frame_count,
                    progress_callback=lambda percent, tag: context.progress(percent, tag) if context else null_func
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
        latent_image=latent
    )
    out = main_sampler.run_sampling(callback=lambda i, s: progress_callback(i, s))
    
    wan_dit_model.to("cpu")
    wan_type = model_wrapper.model.model_arch_config.default_vae_type
    del wan_dit_model, model_wrapper
    MemoryManager.clear_memory()
    
    if context: context.start_anchor("Decoding", steps=1) # 5%
    out = out.to(dtype=vae_dtype)
    wan_vae = get_vae(
        vae_type=wan_type,
        vae_dtype=vae_dtype,
        use_tiling=use_vae_tiling
    )
    out = wan_vae.decode(out, (width, height, frame_count))
    output_paths = MediaProcessor.save_latents_to_media(out)
    
    return {"1": output_paths[0]}

DEFAULT_CONDS = get_dummy_cond(positive="a dog running the park")
DEFAULT_HOOKS = get_dummy_hook(enable_step_caching=True)
DEFAULT_LATENT = get_dummy_latent(img_path_list=["./tests/test_utils/assets/media/dog_realistic.jpg"])
# DEFAULT_LATENT = get_dummy_latent()
app = App(
    name="Text to Video",
    inputs={
        'conditions': Input(label="Conditions (JSON)", type="json", default=DEFAULT_CONDS,),
        'hooks': Input(label="Hooks (JSON)", type="json", default=DEFAULT_HOOKS),
        'latent': Input(label="Latent", type="json", default=DEFAULT_LATENT),
        'seed': Input(label="Seed", type="number", default=256347,),
        'steps': Input(label="Steps", type="number", default=30, increment_controls=True, increment_step=2,),
        'cfg': Input(label="CFG", type="number", default=6, increment_controls=True, increment_step=0.2,),
        'sampler_name': Input(label="Sampler Name", type="select", options=KSamplerType.value_list(), \
            default=KSamplerType.EULER.value,),
        'scheduler_name': Input(label="Scheduler Name", type="select", options=SchedulerType.value_list(), \
            default=SchedulerType.SIMPLE.value,),
        'height': Input(label="Height", type="number", default=480),
        'width': Input(label="Width", type="number", default=832),
        # 'height': Input(label="Height", type="number", default=512),
        # 'width': Input(label="Width", type="number", default=512),
        'frame_count': Input(label="Frame Count", type="number", default=81),
    },
    outputs=[Output(id=1, type=AppOutputType.VIDEO.value)],
    handler=main
)

if __name__ == "__main__":
    app.run_standalone()