import torch
import torchvision

from exiv.components.enum import KSamplerType, ModelType, SchedulerType
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
from exiv.utils.device import OFFLOAD_DEVICE, VRAM_DEVICE, MemoryManager, ProcDevice
from exiv.utils.file import ImageProcessor, ensure_model_available
from exiv.utils.tensor import common_upscale

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
        model_path = ensure_model_available(model_path=model_path, download_url=download_url)
        
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

def main():
    positive_prompt = "a dog running in the park"
    negative_prompt = "blurry, bad quality"
    input_img = ImageProcessor.load_image_list("./tests/test_utils/assets/media/test.jpg")[0]
    height, width, output_frame_count = 512, 512, 81
    
    # resizing img
    input_img = common_upscale(input_img.unsqueeze(0), height, width)   # (B, C, H, W)
    
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
    
    print("here")
    MemoryManager.clear_memory()
    
    # create a model wrapper
    model_path = "./tests/test_utils/assets/models/wan21_14B.safetensors"
    # download_url = "https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B/resolve/main/diffusion_pytorch_model.safetensors?download=true"
    download_url = "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/diffusion_models/wan2.1_t2v_14B_fp16.safetensors?download=true"
    wan_dit_model = Wan21Model(force_load_mode=LOADING_MODE.LOW_VRAM.value, dtype=torch.float16)
    # wan_dit_model = Wan21Model(dtype=torch.float16)
    wan_dit_model.load_model(model_path=model_path, download_url=download_url)
    model_sampling = get_model_sampling(ModelType.EDM)
    model_wrapper = ModelWrapper(
        model=wan_dit_model,
        model_sampling=model_sampling,
        cfg_func=default_cfg
    )

    print("before sampling")
    # the main sampling loop
    main_sampler = KSampler(
        wrapped_model=model_wrapper,
        seed=123,
        steps=10,
        cfg=7.0,
        sampler_name=KSamplerType.EULER.value,
        scheduler_name=SchedulerType.SIMPLE.value,
        positive=pos_embed,
        negative=neg_embed,
        latent_image=blank_latent
    )
    
    out = main_sampler.run_sampling()
    wan_dit_model.to("cpu")
    del wan_dit_model, model_wrapper
    MemoryManager.clear_memory()
    
    print("final decode")
    model_path = "./tests/test_utils/assets/models/wan_2_1_vae.safetensors"
    wan_vae = Wan21VAE()
    wan_vae.load_model(model_path=model_path)
    move_model(wan_vae, VRAM_DEVICE)
    out = wan_vae.decode(out, (height, width, output_frame_count))
    save_video(out)
    
def save_video(out):
    video_tensor = out.sample if hasattr(out, "sample") else out

    # rescale from [-1, 1] to [0, 255] and cast to uint8
    video_tensor = ((video_tensor.clamp(-1.0, 1.0) + 1.0) * 127.5).to(torch.uint8)

    # current shape: (Batch, Channels, Time, Height, Width) -> e.g., (1, 3, 121, 512, 768)
    for i, video in enumerate(video_tensor):
        # (C, T, H, W) -> (T, H, W, C), for torchvision
        video_formatted = video.permute(1, 2, 3, 0).cpu()
        
        save_path = f"output_video_{i}.mp4"
        torchvision.io.write_video(
            save_path,
            video_formatted,
            fps=24,
            options={"crf": "5"}  # 'Constant Rate Factor' for quality (lower is better)
        )
        print(f"Saved {save_path}") 
    
main()