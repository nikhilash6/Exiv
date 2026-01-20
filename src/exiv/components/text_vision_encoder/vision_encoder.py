from typing import Optional

import torch

from exiv.utils.file_path import FilePathData, FilePaths
from .utils import convert_sd_to_hf_format
from .encoder_base import VisionEncoder
from .ve_clip import CLIPViTH, CLIPViTL, CLIPVitLlava
from ..enum import VisionEncoderType
from ...model_utils.model_mixin import ModelMixin, get_state_dict
from ...utils.device import OFFLOAD_DEVICE, VRAM_DEVICE
from ...utils.file import ensure_model_availability


VE_TYPE_CLS_MAP = {
    VisionEncoderType.CLIP_L.value: CLIPViTL,
    VisionEncoderType.CLIP_H.value: CLIPViTH,
    VisionEncoderType.CLIP_L_LLAVA.value: CLIPVitLlava,
}

def get_ve_type(sd):
    if "vision_model.encoder.layers.47.layer_norm1.weight" in sd:
        return VisionEncoderType.CLIP_G.value
    elif "vision_model.encoder.layers.30.layer_norm1.weight" in sd:
        return VisionEncoderType.CLIP_H.value
    elif "vision_model.encoder.layers.22.layer_norm1.weight" in sd:
        embed_shape = sd["vision_model.embeddings.position_embedding.weight"].shape[0]
        if sd["vision_model.encoder.layers.0.layer_norm1.weight"].shape[0] == 1152:
            if embed_shape == 729:
                return VisionEncoderType.SIGLIP_384.value
            elif embed_shape == 1024:
                return VisionEncoderType.SIGLIP_512.value
        elif embed_shape == 577:
            if "multi_modal_projector.linear_1.bias" in sd:
                return VisionEncoderType.CLIP_L_LLAVA.value
            else:
                return VisionEncoderType.CLIP_L_336.value
        else:
            return VisionEncoderType.CLIP_L.value
    elif 'encoder.layer.39.layer_scale2.lambda1' in sd:
        return VisionEncoderType.DINO2_G.value
    elif 'encoder.layer.23.layer_scale2.lambda1' in sd:
        return VisionEncoderType.DINO2_L.value
    else:
        return None


def create_vision_encoder(
    filename: Optional[str] = None, 
    model_type: Optional[str] = None, 
    dtype=torch.float16
) -> VisionEncoder:
    assert filename is not None or model_type is not None, "atleast one of filename or model_type is required to create the vision encoder"
    
    model_path = None   # NOTE: VE models use the default model from their config
    if filename:
        model_path_data: FilePathData = FilePaths.get_path(filename=filename, file_type="vision_encoder")
        model_path: str = ensure_model_availability(model_path_data.path, model_path_data.url)
        state_dict = get_state_dict(model_path, device=OFFLOAD_DEVICE)
        state_dict = convert_sd_to_hf_format(state_dict)
        ve_type: str = get_ve_type(state_dict)
        del state_dict
    else:
        ve_type: str = model_type
        
    if ve_type is None or ve_type not in VE_TYPE_CLS_MAP:
        raise Exception("Vision encoder not supported")
    
    ve_cls = VE_TYPE_CLS_MAP[ve_type]
    return ve_cls(model_path=model_path, dtype=dtype)