from .utils import convert_sd_to_hf_format
from .encoder_base import VisionEncoder
from .ve_clip import CLIPViTH, CLIPViTL, CLIPVitLlava
from ..enum import VisionEncoderType
from ...model_utils.model_mixin import ModelMixin
from ...utils.device import VRAM_DEVICE


VE_TYPE_CLS_MAP = {
    VisionEncoderType.CLIP_L: CLIPViTL,
    VisionEncoderType.CLIP_H: CLIPViTH,
    VisionEncoderType.CLIP_L_LLAVA: CLIPVitLlava,
}

def ve_type(sd):
    if "vision_model.encoder.layers.47.layer_norm1.weight" in sd:
        return VisionEncoderType.CLIP_G
    elif "vision_model.encoder.layers.30.layer_norm1.weight" in sd:
        return VisionEncoderType.CLIP_H
    elif "vision_model.encoder.layers.22.layer_norm1.weight" in sd:
        embed_shape = sd["vision_model.embeddings.position_embedding.weight"].shape[0]
        if sd["vision_model.encoder.layers.0.layer_norm1.weight"].shape[0] == 1152:
            if embed_shape == 729:
                return VisionEncoderType.SIGLIP_384
            elif embed_shape == 1024:
                return VisionEncoderType.SIGLIP_512
        elif embed_shape == 577:
            if "multi_modal_projector.linear_1.bias" in sd:
                return VisionEncoderType.CLIP_L_LLAVA
            else:
                return VisionEncoderType.CLIP_L_336
        else:
            return VisionEncoderType.CLIP_L
    elif 'encoder.layer.39.layer_scale2.lambda1' in sd:
        return VisionEncoderType.DINO2_G
    elif 'encoder.layer.23.layer_scale2.lambda1' in sd:
        return VisionEncoderType.DINO2_L
    else:
        return None


def create_vision_encoder(model_path) -> VisionEncoder:
    state_dict = ModelMixin.get_state_dict(model_path, VRAM_DEVICE)
    state_dict = convert_sd_to_hf_format(state_dict)
    
    ve_type: VisionEncoderType = ve_type(state_dict)
    if ve_type is None or ve_type not in VE_TYPE_CLS_MAP:
        raise Exception("Vision encoder not supported")
    
    ve_cls = VE_TYPE_CLS_MAP[ve_type]
    del state_dict
    return ve_cls(model_path)