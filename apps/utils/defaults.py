
import json
from typing import List, Dict, Union

def get_default_cond(
    positive: str = "Cinematic anime style, medium close-up of a teenage boy.",
    negative: str = "bad quality, blurry, low res"
) -> str:
    defaults = [
        {
            "group": "positive",
            "input_metadata": positive,
            "timestep_range": [0.0, -1],
            "frame_range": [0.0, -1],
            # "aux": [{ "type": "clip_image", "input_metadata": "assets/ref.jpg" }]
        },
        {
            "group": "negative",
            "timestep_range": [0.0, -1],
            "frame_range": [0.0, -1],
            "input_metadata": negative,
        }
    ]
    return json.dumps(defaults, indent=2)
