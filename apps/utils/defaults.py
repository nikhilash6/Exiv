
import json
from typing import List, Dict, Union

from exiv.model_patching.cache_hook import CacheType
from exiv.model_patching.hook_registry import FeatureType, HookType
from exiv.model_patching.sliding_context_hook import BlendType
from exiv.model_utils.common_classes import AuxCondType

temp_img = "./tests/test_utils/assets/media/boy_anime.jpg"

def get_dummy_cond(
    positive: str = "Cinematic anime style, medium close-up of a teenage boy with messy dark hair. 0-2s: The boy is looking down with a somber expression, his eyes shadowed. 2-4s: He slowly lifts his head to look directly into the camera, his expression shifting to one of sudden realization and determination, eyes widening with a subtle catchlight. Background is a soft-focus urban rooftop at sunset. Cel-shaded, vibrant colors, fluid character animation, high-quality rendering.",
    negative: str = "bad image, blurry, low quality"
) -> str:
    defaults = [
        {
            "group": "positive",
            "input_metadata": positive,
            "timestep_range": [0.0, -1],
            "frame_range": [0.0, -1],
            "aux": [
                { "type": AuxCondType.REF_LATENT, "input_metadata": temp_img},
                { "type": AuxCondType.VISUAL_EMBEDDING, "input_metadata": temp_img},
            ]
        },
        {
            "group": "negative",
            "timestep_range": [0.0, -1],
            "frame_range": [0.0, -1],
            "input_metadata": negative,
            "aux": [
                { "type": AuxCondType.REF_LATENT, "input_metadata": temp_img},
                { "type": AuxCondType.VISUAL_EMBEDDING, "input_metadata": temp_img},
            ]
        }
    ]
    return json.dumps(defaults, indent=2)

def get_dummy_hook(
    enable_sliding_context=False,
    enable_inpainting=False,
    enable_step_caching=False,
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
            "type": FeatureType.STEP_CACHING.vale,
            "kwarg_data": {
                "cache_type": CacheType.TAYLOR_SEER_LITE.value
            }
        }   # TODO: add proper 'type' this is not 'step caching', 'taylor lite' is a specific case of step caching
        defaults.append(hook)
        
    return json.dumps(defaults)