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
                    
                # vace ctx
                elif c_aux.type == AuxCondType.VACE_CTX and (vace_ctx:=c_aux.data) is not None:
                    output.vace_context = vace_ctx[0]
                    output.vace_strength = vace_ctx[1]
                
                # wan animate
                elif c_aux.type == AuxCondType.POSE_LATENTS and (pose_latents:=c_aux.data) is not None:
                    output.pose_latents = pose_latents
                
                elif c_aux.type == AuxCondType.FACE_PIXEL_VALUES and (face_pixel_values:=c_aux.data) is not None:
                    output.face_pixel_values = face_pixel_values

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