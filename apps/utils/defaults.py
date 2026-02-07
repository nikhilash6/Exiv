
import json
from typing import List, Dict, Optional, Union

from exiv.model_patching.cache_hook import CacheType
from exiv.model_patching.hook_registry import FeatureType, HookType
from exiv.model_patching.sliding_context_hook import BlendType
from exiv.model_utils.common_classes import AuxCondType

# temp_img = "./tests/test_utils/assets/media/boy_anime.jpg"
temp_img = "./input_img.png"
anime_boy_prompt = "Cinematic anime style, medium close-up of a teenage boy with messy dark hair. 0-2s: The boy is looking down with a somber expression, his eyes shadowed. 2-4s: He slowly lifts his head to look directly into the camera, his expression shifting to one of sudden realization and determination, eyes widening with a subtle catchlight. Background is a soft-focus urban rooftop at sunset. Cel-shaded, vibrant colors, fluid character animation, high-quality rendering."
realistic_girl_flowers = "The girl is dancing in a sea of flowers, slowly moving her hands. There is a close - up shot of her upper body. The character is surrounded by other transparent glass flowers in the style of Nicoletta Ceccoli, creating a beautiful, surreal, and emotionally expressive movie scene with a white, transparent feel and a dreamy atmosphere."

def get_dummy_cond(
    positive: str = realistic_girl_flowers,
    negative: str = "过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走,",
    enable_ref_latent=False,
    enable_visual_embed=False,
    enable_vace_ctx=False
) -> str:
    defaults = [
        {
            "group_name": "positive",
            "input_metadata": positive,
            "timestep_range": [0.0, -1],
            "frame_range": [0.0, -1],
            "aux": []
        },
        {
            "group_name": "negative",
            "timestep_range": [0.0, -1],
            "frame_range": [0.0, -1],
            "input_metadata": negative,
            "aux": []
        }
    ]
    
    for i, d in enumerate(defaults):
        if enable_ref_latent: d["aux"].append({ "type": AuxCondType.REF_LATENT, "input_metadata": temp_img})
        if enable_visual_embed: d["aux"].append({ "type": AuxCondType.VISUAL_EMBEDDING, "input_metadata": temp_img})
        if enable_vace_ctx: 
            d["aux"].append({"type": AuxCondType.VACE_CTX, "input_metadata": {"reference_image_path": temp_img, "strength": 1.0}})
    
    return defaults

def get_dummy_hook(
    enable_sliding_context=False,
    enable_step_caching=False,
    enable_causvid_lora=False
) -> str:
    defaults = []
    
    if enable_sliding_context:
        hook = {
            "type": FeatureType.SLIDING_CONTEXT.value,
            "kwarg_data": {
                "config": {
                    "ctx_len": 20,
                    "ctx_overlap": 5,
                    "blend_type": BlendType.PYRAMIND.value,
                },
            }
        }
        defaults.append(hook)
        
    if enable_step_caching:
        hook = {
            "type": FeatureType.STEP_CACHING.value,
            "kwarg_data": {
                "cache_type": CacheType.TAYLOR_SEER_LITE.value
            }
        }   # TODO: add proper 'type' this is not 'step caching', 'taylor lite' is a specific case of step caching
        defaults.append(hook)
        
    if enable_causvid_lora:
        # "wan21_causvid_bidirect2_T2V_1_3B_lora_rank32.safetensors"
        # wan21_causvid_14B_T2V_lora_rank32.safetensors"
        hook = {
            "type": FeatureType.LORA.value,
            "kwarg_data": {
                "filename": "wan21_causvid_14B_T2V_lora_rank32.safetensors",
                "base_strength": 1.0
            }
        }
        defaults.append(hook)
        
    return defaults

def get_dummy_latent(
    img_path_list: List[str] = [],
    noise_mask: Optional[list] = None 
):
    return {
        "image_path_list": img_path_list,
        "noise_mask": noise_mask
    }