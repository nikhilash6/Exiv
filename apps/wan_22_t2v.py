import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import torch

from exiv.components.enum import KSamplerType, SchedulerType
from exiv.components.models.wan.constructor import get_wan_instance
from exiv.components.models.wan.main import Wan22ModelArchConfig
from exiv.components.samplers.model_sampling import KSampler
from exiv.components.text_vision_encoder.te_t5 import UMT5XXL
from exiv.components.text_vision_encoder.text_encoder import WanEncoder
from exiv.components.text_vision_encoder.vision_encoder import create_vision_encoder
from exiv.components.vae.wan_vae import Wan21VAE
from exiv.components.vae.wan_vae22 import Wan22VAE
from exiv.model_patching.cache_hook import enable_step_caching
from exiv.model_utils.common_classes import Latent
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

# TODO: move to the tensors file
def conditioning_set_values(conditioning_list, new_values_dict = {}):
    is_list = lambda x : isinstance(x, list)
    is_tuple = lambda x : isinstance(x, tuple)
    is_not_list_of_list = is_list(conditioning_list) and not \
        (is_list(conditioning_list[0]) or is_tuple(conditioning_list[0]))
    if is_not_list_of_list or is_tuple(conditioning_list):
        # general format is to have list of lists, to accomodate for different
        # types of prompts for the single generation (like regional prompting)
        conditioning_list = [conditioning_list]     # list of [tensor, options_dict] pairs
        
    updated_conditioning_list = []
    for conditioning_item in conditioning_list:
        updated_item = [conditioning_item[0], conditioning_item[1].copy()]
        options_dict_to_update = updated_item[1]

        for key, value_to_add in new_values_dict.items():
            value_to_set = value_to_add
            options_dict_to_update[key] = value_to_set

        updated_conditioning_list.append(updated_item)

    return updated_conditioning_list

def preprocess_wan_conditionals(pos_embed_dict, neg_embed_dict, clip_embed_dict, input_img, height, width, frame_count, batch_size) -> tuple[list, list, Latent]:
    # converting pos and neg embed dict in the appropriate format (list of lists)
    pos_embed = [[pos_embed_dict.pop("output"), pos_embed_dict]]
    neg_embed = [[neg_embed_dict.pop("output"), neg_embed_dict]]
    
    # encode the entire sequence
    cur_model = "wan_2_2_vae.safetensors"
    model_path_data: FilePathData = FilePaths.get_path(filename=cur_model, file_type="vae")
    model_path = ensure_model_availability(model_path=model_path_data.path, download_url=model_path_data.url)
    wan_vae = Wan22VAE(dtype=vae_dtype, use_tiling=use_vae_tiling)
    wan_vae.load_model(model_path=model_path)
    move_model(wan_vae, VRAM_DEVICE)
    
    # empty tensor
    blank_latent = Latent()
    compression_factor = wan_vae.spatial_compression_ratio
    latent = torch.zeros([1, 48, ((frame_count - 1) // 4) + 1, height // compression_factor, width // compression_factor], device=VRAM_DEVICE)     # B, C, T, H, W

    # NOTE: this is a divergence from the original repo to aim for 100% perfect first frame
    # insteads of passing the image as a hint, we pass it directly attached to the latent with mask 
    # that tells not to touch the attached image part
    if input_img is None:
        blank_latent.samples = latent
        return pos_embed, neg_embed, blank_latent

    mask = torch.ones([latent.shape[0], 1, ((frame_count - 1) // 4) + 1, latent.shape[-2], latent.shape[-1]], device=VRAM_DEVICE)
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
    return pos_embed, neg_embed, blank_latent

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
    model_path_data: FilePathData = FilePaths.get_path(filename=cur_model, file_type="clip")
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
    model_path_data: FilePathData = FilePaths.get_path(filename=cur_model, file_type="clip_vision")
    clip_model = create_vision_encoder(model_path=model_path_data.path, download_url=model_path_data.url, dtype=torch.float16)
    clip_model.load_model()
    clip_embed_dict = clip_model.encode_image(input_img)
    del clip_model
    MemoryManager.clear_memory()
    
    # preprocess conditionals
    pos_embed, neg_embed, blank_latent = preprocess_wan_conditionals(
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
        positive=pos_embed,
        negative=neg_embed,
        latent_image=blank_latent
    )
    
    # from torch_tracer import TorchTracer
    
    # with TorchTracer("./exiv_2.pkl"):
    out = main_sampler.run_sampling(callback=lambda i, s: progress_callback(0.35 + round(i * 0.6, 2), s))
    wan_dit_model.to("cpu")
    del wan_dit_model, model_wrapper
    MemoryManager.clear_memory()
    
    progress_callback(0.95, "Decoding output latents")
    cur_model = "wan_2_2_vae.safetensors"
    model_path_data: FilePathData = FilePaths.get_path(filename=cur_model, file_type="vae")
    out = out.to(vae_dtype)
    wan_vae = Wan22VAE(dtype=vae_dtype, use_tiling=use_vae_tiling)
    wan_vae.load_model(model_path=model_path_data.path)
    move_model(wan_vae, VRAM_DEVICE)
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
            default="a dog running in the park then rolling over",
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