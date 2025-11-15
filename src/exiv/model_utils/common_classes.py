import torch
from torch import Tensor

from typing import Any, List
from dataclasses import dataclass, field

from exiv.components.latent_format import Wan21VAELatentFormat

from .model_mixin import ModelMixin
from ..utils.enum import ExtendedEnum

@dataclass
class Latent:
    samples: Tensor | None = None
    batch_index: List[int] | None = None
    noise_mask: Tensor | None = None

class ModelOption(ExtendedEnum):
    POST_CFG_FUNC = "sampler_post_cfg_function"

class ModelArchConfig:
    # will extend this as more models are added
    latent_format = None
    
class Wan21ModelArchConfig(ModelArchConfig):
    latent_format = Wan21VAELatentFormat()

class ModelWrapper:
    '''
    contains
    - the main model
    - methods of sampling
    - model arch details
    '''
    def __init__(
        self, 
        model: ModelMixin, 
        model_sampling: Any, 
        model_arch_config: ModelArchConfig
    ):
        self.model = model
        self.model_sampling = model_sampling
        self.model_arch_config = model_arch_config
    
    def update_options(self, key, value):
        assert key in ModelOption.value_list(), "invalid model option"
        
        self.model_options[key] = value
        return self.model_options
    
    def process_latent_in(self, latent_in: Tensor) -> Tensor:
        return self.model_arch_config.latent_format.process_in(latent_in)
    
    def process_latent_out(self, latent_out: Tensor) -> Tensor:
        return self.model_arch_config.latent_format.process_out(latent_out)