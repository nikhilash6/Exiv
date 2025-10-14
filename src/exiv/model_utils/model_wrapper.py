import torch
from torch import Tensor

from typing import List

from ..utils.enum import ExtendedEnum


class ModelOption(ExtendedEnum):
    POST_CFG_FUNC = "sampler_post_cfg_function"

class ModelWrapper:
    def __init__(self, *args, **kwargs):
        pass
    
    # TODO: probably needs to be broken into smaller funcs
    def update_options(self, key, value):
        assert key in ModelOption.value_list(), "invalid model option"
        
        self.model_options[key] = self.model_options.get(key, []) + [value]
        return self.model_options
    
    def process_latent_in(self, latent_image: Tensor) -> Tensor:
        pass
    
    def process_latent_out(self, samples: List[Tensor]) -> List[Tensor]:
        pass