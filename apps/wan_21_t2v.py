import torch

from exiv.components.enum import KSamplerType, ModelType, SchedulerType
from exiv.components.models.wan.constructor import get_wan_21_instance
from exiv.components.models.wan.main import Wan21Model, WanModelArchConfig
from exiv.components.samplers.cfg_methods import default_cfg
from exiv.components.samplers.model_sampling import KSampler
from exiv.components.samplers.sampler_types import get_model_sampling
from exiv.components.text_vision_encoder.te_t5 import T5XXL, UMT5XXL
from exiv.components.text_vision_encoder.text_encoder import WanEncoder
from exiv.components.text_vision_encoder.vision_encoder import create_vision_encoder
from exiv.components.vae.wan_vae import Wan21VAE
from exiv.config import LOADING_MODE
from exiv.model_utils.common_classes import Latent
from exiv.model_utils.common_classes import ModelWrapper
from exiv.model_utils.helper_methods import move_model
from exiv.server.app_core import App, AppOutputType, Input, Output
from exiv.utils.device import OFFLOAD_DEVICE, VRAM_DEVICE, MemoryManager, ProcDevice
from exiv.utils.file import MediaProcessor, ensure_model_availability
from exiv.utils.tensor import common_upscale
from exiv.utils.logging import app_logger

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
    
    # empty tensor
    blank_latent = Latent()
    blank_latent.samples = torch.zeros([batch_size, 16, ((frame_count - 1) // 4) + 1, height // 8, width // 8], device=OFFLOAD_DEVICE)
    if input_img is not None:
        # empty image latent (T, H, W, C)
        image = torch.ones((frame_count, height, width, input_img.shape[1]), device=input_img.device, dtype=input_img.dtype) * 0.5
        image[0] = input_img[0].permute(1, 2, 0)     # first conditionals are replaced by the input_img

        # (T, H, W, C) -> (B, C, T, H, W)
        image = image.permute(3, 0, 1, 2).unsqueeze(0)
        
        # encode the entire sequence
        download_url = "https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P-Diffusers/resolve/main/vae/diffusion_pytorch_model.safetensors?download=true"
        model_path = "./tests/test_utils/assets/models/wan_2_1_vae.safetensors"
        model_path = ensure_model_availability(model_path=model_path, download_url=download_url)
        
        image = image.to(VRAM_DEVICE)
        wan_vae = Wan21VAE()
        wan_vae.load_model(model_path=model_path)
        move_model(wan_vae, VRAM_DEVICE)
        concat_latent_image = wan_vae.encode(image)
        del wan_vae, image

        mask = torch.ones((1, 1, blank_latent.samples.shape[2], concat_latent_image.shape[-2], concat_latent_image.shape[-1]), device=input_img.device, dtype=input_img.dtype)
        mask[:, :, :((input_img.shape[0] - 1) // 4) + 1] = 0.0      # setting the mask to 0 for the conditioning image

        # following comfy's flow of just adding the conditionals to the dict
        pos_embed = conditioning_set_values(pos_embed, {"concat_latent_image": concat_latent_image, "concat_mask": mask})
        neg_embed = conditioning_set_values(neg_embed, {"concat_latent_image": concat_latent_image, "concat_mask": mask})

    if clip_embed_dict:
        pos_embed = conditioning_set_values(pos_embed, {"clip_vision_output": clip_embed_dict})
        neg_embed = conditioning_set_values(neg_embed, {"clip_vision_output": clip_embed_dict})

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
    input_img = MediaProcessor.load_image_list("./tests/test_utils/assets/media/test.jpg")[0]
    height, width, output_frame_count = 512, 512, 81
    
    # resizing img
    input_img = common_upscale(input_img.unsqueeze(0), height, width)   # (B, C, H, W)
    
    progress_callback(0.2, "Encoding prompts")
    # generate text embeddings
    model_path = "./tests/test_utils/assets/models/umt5_xxl_fp16.safetensors"
    download_url = "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp16.safetensors?download=true"
    t5_xxl = UMT5XXL(model_path=model_path, dtype=torch.float16)
    wan_encoder = WanEncoder(t5_xxl=t5_xxl)
    wan_encoder.load_model(t5_xxl_download_url=download_url)
    pos_embed_dict = wan_encoder.encode(positive_prompt)
    neg_embed_dict = wan_encoder.encode(negative_prompt)
    del t5_xxl
    del wan_encoder
    
    progress_callback(0.3, "Generating CLIP embeddings")
    # generate img embeddings
    clip_vision_model_path = "./tests/test_utils/assets/models/CLIP-ViT-H-fp16.safetensors"
    download_url = "https://huggingface.co/Kijai/CLIPVisionModelWithProjection_fp16/resolve/main/CLIP-ViT-H-fp16.safetensors?download=true"
    clip_model = create_vision_encoder(model_path=clip_vision_model_path, download_url=download_url, dtype=torch.float16)
    clip_model.load_model()
    clip_embed_dict = clip_model.encode_image(input_img)
    del clip_model
    
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
    model_path = "./tests/test_utils/assets/models/wan21_1_3B.safetensors"
    # model_path = "./tests/test_utils/assets/models/wan21_14B.safetensors"
    download_url = "https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B/resolve/main/diffusion_pytorch_model.safetensors?download=true"
    # download_url = "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/diffusion_models/wan2.1_t2v_14B_fp16.safetensors?download=true"
    wan_dit_model = get_wan_21_instance(model_path, download_url, force_dtype=torch.float16)
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
    
    out = main_sampler.run_sampling(callback=lambda i, s: progress_callback(0.35 + round(i * 0.6, 2), s))
    wan_dit_model.to("cpu")
    del wan_dit_model, model_wrapper
    MemoryManager.clear_memory()
    
    progress_callback(0.95, "Decoding output latents")
    model_path = "./tests/test_utils/assets/models/wan_2_1_vae.safetensors"
    wan_vae = Wan21VAE()
    wan_vae.load_model(model_path=model_path)
    move_model(wan_vae, VRAM_DEVICE)
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
            default=-1,
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