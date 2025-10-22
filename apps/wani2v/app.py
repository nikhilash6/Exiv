import torch

from exiv.components.enum import KSamplerType, ModelType, SchedulerType
from exiv.components.models.wan.main import WanModel, WanModelArchConfig
from exiv.components.samplers.model_sampling import KSampler
from exiv.components.samplers.sampler_types import get_model_sampling
from exiv.components.text_vision_encoder.te_t5 import T5XXL
from exiv.components.text_vision_encoder.text_encoder import WanEncoder
from exiv.components.text_vision_encoder.vision_encoder import create_vision_encoder
from exiv.components.vae.wan_vae import WanVAE
from exiv.model_utils.latent import Latent
from exiv.model_utils.model_wrapper import ModelWrapper
from exiv.utils.device import OFFLOAD_DEVICE, ProcDevice
from exiv.utils.file import ImageProcessor
from exiv.utils.tensor import common_upscale


def conditioning_set_values(embed, dict):
    pass

def preprocess_wan_conditionals(pos_embed, neg_embed, clip_embed, wan_vae, input_img, height, width, frame_count, batch_size):
    # empty tensor
    blank_latent = torch.zeros([batch_size, 16, ((frame_count - 1) // 4) + 1, height // 8, width // 8], device=OFFLOAD_DEVICE)
    if input_img:
        # empty image latent
        image = torch.ones((frame_count, height, width, input_img.shape[-1]), device=input_img.device, dtype=input_img.dtype) * 0.5
        image[:input_img.shape[0]] = input_img      # first conditionals are replaced by the input_img

        # encode the entire sequence
        concat_latent_image = wan_vae.encode(image[:, :, :, :3])

        mask = torch.ones((1, 1, blank_latent.shape[2], concat_latent_image.shape[-2], concat_latent_image.shape[-1]), device=input_img.device, dtype=input_img.dtype)
        mask[:, :, :((input_img.shape[0] - 1) // 4) + 1] = 0.0      # setting the mask to 0 for the conditioning image

        # TODO: right now just going with comfy's flow, will change this later
        # injecting mask and the latent image into the embeds
        pos_embed = conditioning_set_values(pos_embed, {"concat_latent_image": concat_latent_image, "concat_mask": mask})
        neg_embed = conditioning_set_values(neg_embed, {"concat_latent_image": concat_latent_image, "concat_mask": mask})

    if clip_embed:
        pos_embed = conditioning_set_values(pos_embed, {"clip_vision_output": clip_embed})
        neg_embed = conditioning_set_values(neg_embed, {"clip_vision_output": clip_embed})

    return pos_embed, neg_embed, Latent(samples=blank_latent)

def main():
    positive_prompt = "a dog running in the park"
    negative_prompt = "blurry, bad quality"
    input_img = ImageProcessor.load_image_list("./assets/boy.jpg")[0]
    height, width, output_frame_count = 512, 512, 81
    
    # resizing img
    input_img = common_upscale([input_img], height, width)
    
    # generate text embeddings
    t5_xxl = T5XXL(model_path="")
    wan_encoder = WanEncoder(t5_xxl=t5_xxl)
    pos_embed = wan_encoder.encode(positive_prompt)
    neg_embed = wan_encoder.encode(negative_prompt)
    
    # generate img embeddings
    clip_vision_model_path = "temp"
    clip_model = create_vision_encoder(clip_vision_model_path)
    clip_embed = clip_model.encode_image(input_img)
    
    # encoded image latent
    wan_vae = WanVAE()
    
    # preprocess conditionals
    pos_embed, neg_embed, blank_latent = preprocess_wan_conditionals(
                                            pos_embed, 
                                            neg_embed, 
                                            clip_embed, 
                                            wan_vae, 
                                            input_img, 
                                            height, 
                                            width, 
                                            output_frame_count, 
                                            1
                                        )
    
    # create a model wrapper
    wan_dit_model = WanModel()
    model_sampling = get_model_sampling(ModelType.EDM)
    model_wrapper = ModelWrapper(
        model=wan_dit_model,
        model_sampling=model_sampling,
        model_arch_config=WanModelArchConfig()
    )

    # the main sampling loop
    main_sampler = KSampler(
        wrapped_model=model_wrapper,
        seed=123,
        steps=50,
        cfg=7.0,
        sampler_name=KSamplerType.EULER.value,
        scheduler_name=SchedulerType.SIMPLE.value,
        positive=pos_embed,
        negative=neg_embed,
        latent_image=blank_latent
    )
    
    out = main_sampler.run_sampling()
    out = wan_vae.decode(out)