import torch
from torch import Tensor

from typing import List, Any

from ..utils.logging import app_logger
from ..utils.tensor import common_upscale, repeat_to_batch_size
from .common_classes import AuxCondType, Conditioning, ConditioningType, ModelForwardInput

class ConditioningMixin:
    """
    Handles all conditioning preparation, filtering, and formatting.
    """

    def prepare_model_input(self, cond: Conditioning, **ctx) -> ModelForwardInput:
        output = ModelForwardInput()
        noise = ctx.get("noise")

        # text embeds
        if cond.type == ConditioningType.EMBEDDING:
             output.cross_attn = cond.data

        # auxiliary signals
        if cond.aux and len(cond.aux):
            for c_aux in cond.aux:
                if c_aux.type == AuxCondType.TIME_HINT and c_aux.data is not None:
                    t = self.process_latent_in(c_aux.data)
                    output.time_hint = t

                elif c_aux.type == AuxCondType.REF_LATENT and (refs:=c_aux.data) is not None:
                    output.reference_latent = refs
                    
                # ipa / custom vision embeds
                elif c_aux.type == AuxCondType.VISUAL_EMBEDDING and (vis_embed:=c_aux.data) is not None:
                    output.visual_embedding = vis_embed

        return output
    
    def filter_conditionings(self, conditionings: List[Conditioning] | None) -> List[Conditioning]:
        # TODO: NOT in use rn, will integrate as more models are added
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
            mask = common_upscale(mask, target_shape[-1], target_shape[-2], upscale_method="bilinear", crop="none")[0]

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
