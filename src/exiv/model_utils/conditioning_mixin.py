import torch
from torch import Tensor

from typing import List, Any

from ..utils.logging import app_logger
from ..utils.tensor import common_upscale, repeat_to_batch_size
from .common_classes import Conditioning, ConditioningType, ModelForwardInput

class ConditioningMixin:
    """
    Handles all conditioning preparation, filtering, and formatting.
    """
    
    def get_concat_components(
        self, 
        cond: Conditioning, 
        noise: Tensor, 
        extra_channels: int = 0, 
        mask_channels: int = 4
    ):
        """
        Prepares the concatenation conditioning (Mask + Reference Image) for I2V/Inpainting.
        
        Inserts 'concat_mask' in a specific position ('concat_mask_index') in the 
        'concat_latent_image' tensor channels. Insert position defaults to 0.
        """
        assert noise is not None, "noise tensor can't be None during cond preprocessing"
        
        # extra channels supported by the model
        if extra_channels == 0:
            return None, None, None

        device = noise.device
        if cond.concat is None:
            image, mask, mask_index = None, None, 0
        else:
            image = cond.concat.data
            mask = cond.concat.mask
            mask_index = cond.concat.mask_index
        
        if image is None:
            # CASE 1: No reference image was provided 
            # (but we have to fill the extra channels)
            shape_image = list(noise.shape)
            shape_image[1] = extra_channels
            image = torch.zeros(shape_image, dtype=noise.dtype, layout=noise.layout, device=noise.device)
        else:
            # CASE 2: A reference image exists.
            latent_dim = self.latent_format.latent_channels
            
            # upscale/resize + process to match vae's distribution
            image = common_upscale(image.to(device), noise.shape[-1], noise.shape[-2], "bilinear", "center")
            for i in range(0, image.shape[1], latent_dim):
                image[:, i: i + latent_dim] = self.process_latent_in(image[:, i: i + latent_dim])
            
            image = repeat_to_batch_size(image, noise.shape[0])

        # 4 channels are reserved for the mask, if image has too many then we crop it
        if extra_channels != image.shape[1] + mask_channels:
            # only truncate if we are strictly in an I2V scenario or overflowing
            if getattr(self, "image_to_video", False) or image.shape[1] > (extra_channels - mask_channels):
                 image = image[:, :(extra_channels - mask_channels)]

        # --- Mask Processing ---
        if mask is None:
            # defaulting to all zeros ("keep the original")
            mask = torch.zeros_like(noise)[:, :mask_channels]
        else:
            if mask.shape[1] != mask_channels:
                mask = torch.mean(mask, dim=1, keepdim=True)
            
            mask = 1.0 - mask
            mask = common_upscale(mask.to(device), noise.shape[-1], noise.shape[-2], "bilinear", "center")
            if noise.ndim == 5 and mask.shape[2] < noise.shape[2]:
                pad_len = noise.shape[2] - mask.shape[2]
                # F.pad format: (left, right, top, bottom, front, back)
                mask = torch.nn.functional.pad(mask, (0, 0, 0, 0, 0, pad_len), mode='constant', value=0)
            
            # expand to mask_channels channels if needed
            if mask.shape[1] == 1:
                mask = mask.repeat(1, mask_channels, *([1] * (mask.ndim - 2)))
            
            mask = repeat_to_batch_size(mask, noise.shape[0])

        return image, mask, mask_index
    
    def prepare_concat_latent(self, cond, noise):
        # NOTE: needs to be overidden inside the model
        app_logger.warning("concat latent not supported by the model, skipping")
        return None
    
    def prepare_model_input(self, cond: Conditioning, **ctx) -> ModelForwardInput:
        output = ModelForwardInput()
        noise = ctx.get("noise")

        # channel concat latents
        concat_tensor = self.prepare_concat_latent(cond, noise)
        if concat_tensor is not None:
            output.concat_map = concat_tensor

        # text embeds
        if cond.type == ConditioningType.EMBEDDING:
             output.cross_attn = cond.data
        # ipa or vision embeds
        elif cond.type == ConditioningType.VISION:
             output.visual_embedding = cond.data

        # auxiliary signals
        if cond.aux:
            if cond.aux.time_hint is not None:
                t = self.process_latent_in(cond.aux.time_hint)
                output.time_hint = t

            if cond.aux.reference_latents is not None:
                refs = cond.aux.reference_latents
                processed_ref = self.process_latent_in(refs[-1])[:, :, 0]   # taking only the first frame (b,c,t,h,w)
                output.reference_latent = processed_ref

        return output
    
    def filter_conditionings(self, conditionings: List[Conditioning] | None) -> List[Conditioning]:
        if conditionings is None: return None
        supported_types = getattr(self, "supported_conditioning", [])
        
        filtered_conds = []
        for cond in conditionings:
            is_supported_type = cond.type in supported_types
            has_valid_aux = cond.aux is not None and (cond.aux.time_hint is not None or cond.aux.reference_latents is not None)
            has_valid_concat = cond.concat is not None

            if is_supported_type or has_valid_aux or has_valid_concat:
                filtered_conds.append(cond)
            else:
                app_logger.warning(f"{cond.type} is not supported for the current generation! skipping it.")
        
        return filtered_conds

    @staticmethod
    def prepare_mask(mask: Tensor, target_shape: tuple, device: Any) -> Tensor:
        """
        Ensures the mask is of the proper dimensions.
        - Matches the number of dimensions of the target shape.
        - Interpolates the spatial dimensions (last two).
        - Adjusts the batch and channel dimensions to match the target shape.
        """
        mask = mask.to(device)
        
        # Ensure mask has the same number of dimensions as shape
        while mask.ndim < len(target_shape):
            mask = mask.unsqueeze(0)
        
        # Spatial interpolation on the last two dimensions
        if mask.shape[-2:] != target_shape[-2:]:
            mask = common_upscale(mask, target_shape[-1], target_shape[-2], upscale_method="bilinear", crop="none")

        # Adjust channel dimension (shape[1])
        if mask.shape[1] != target_shape[1]:
            if mask.shape[1] == 1:
                # Expand works for both 4D [B, C, H, W] and 5D [B, C, T, H, W]
                mask = mask.expand(mask.shape[0], target_shape[1], *mask.shape[2:])
            else:
                mask = repeat_to_batch_size(mask, target_shape[1], dim=1)

        # Adjust batch dimension (shape[0])
        mask = repeat_to_batch_size(mask, target_shape[0])
        
        # Adjust temporal dimension if 5D
        if mask.ndim == 5 and target_shape[2] != mask.shape[2]:
            mask = repeat_to_batch_size(mask, target_shape[2], dim=2)

        return mask
