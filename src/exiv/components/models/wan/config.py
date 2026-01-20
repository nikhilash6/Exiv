import torch

from ...latent_format import Wan21VAELatentFormat, Wan22VAELatentFormat
from ....model_utils.model_mixin import ModelArchConfig
from ....utils.device import VRAM_DEVICE
from ....utils.tensor import common_upscale
from ...enum import Model, VAEType


is_text_model = lambda model_type: model_type in [Model.WAN21_1_3B_T2V.value, Model.WAN22_5B_T2V.value]
is_img_model = lambda model_type: not is_text_model(model_type)

class Wan21ModelArchConfig(ModelArchConfig):
    def __init__(self, model_type):
        self.model_type = model_type
        self.latent_format = Wan21VAELatentFormat()
        
        # default models
        self.default_vae_type = VAEType.WAN21.value
        self.default_text_encoder = "umt5_xxl_fp16.safetensors"
        self.default_vision_encoder = "CLIP-ViT-H-fp16.safetensors"
        
    def get_ref_latent(self, start_image, vae, length, width, height):
        if is_text_model(self.model_type):
            return None
        start_image = common_upscale(start_image, width, height, "bilinear", "center")[0]
        video = torch.ones((1, 3, length, height, width), device=start_image.device, dtype=start_image.dtype) * 0.5
        video[:, :, 0, :, :] = start_image
        video = video.to(dtype=vae.dtype)
        concat_latent_image = vae.encode(video)
        concat_latent_image = self.latent_format.process_in(concat_latent_image)
        mask = torch.zeros(
            (
                1, 
                4, 
                ((length - 1) // vae.temporal_compression_ratio) + 1, 
                concat_latent_image.shape[-2], 
                concat_latent_image.shape[-1]
            ), 
            device=start_image.device,
            dtype=start_image.dtype
        )
        mask[:, :, :((start_image.shape[0] - 1) // vae.temporal_compression_ratio) + 1] = 1.0
        
        mask = mask.to(VRAM_DEVICE)
        concat_latent_image = concat_latent_image.to(VRAM_DEVICE)
        conditioning = torch.cat([mask, concat_latent_image], dim=1)
        return conditioning

class Wan22ModelArchConfig(Wan21ModelArchConfig):
    def __init__(self, model_type=Model.WAN22_5B_T2V.value):
        self.model_type = model_type
        self.latent_format = Wan22VAELatentFormat()
        
        # default models
        self.default_vae_type = VAEType.WAN22.value