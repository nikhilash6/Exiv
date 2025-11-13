import torch
from torch import Tensor

from typing import Any, List
from dataclasses import dataclass, field

from .model_mixin import ModelMixin
from ..utils.enum import ExtendedEnum

@dataclass
class Latent:
    samples: Tensor | None = None
    batch_index: List[int] | None = None
    noise_mask: Tensor | None = None
    

class ModelOption(ExtendedEnum):
    POST_CFG_FUNC = "sampler_post_cfg_function"

# basic config defining model arch (awkwardly placed rn, will move later)
class ModelArchConfig:
    latent_channels = 16

class ModelWrapper:
    '''
    contains
    - the main model
    - methods of sampling
    - model arch details like num of channels
    '''
    
    # TODO: create protocol class from model sampling, rn using Any
    def __init__(
        self, 
        model: ModelMixin, 
        model_sampling: Any, 
        model_arch_config: ModelArchConfig
    ):
        self.model = model
        self.model_sampling = model_sampling
        self.model_arch_config = model_arch_config
    
    # TODO: probably needs to be broken into smaller funcs
    def update_options(self, key, value):
        assert key in ModelOption.value_list(), "invalid model option"
        
        self.model_options[key] = self.model_options.get(key, []) + [value]
        return self.model_options
    
    def process_latent_in(self, latent_image: Tensor) -> Tensor:
        pass
    
    def process_latent_out(self, samples: List[Tensor]) -> List[Tensor]:
        pass