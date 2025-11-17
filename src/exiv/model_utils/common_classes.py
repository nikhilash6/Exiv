import torch
from torch import Tensor

from typing import Any, Callable, List
from dataclasses import dataclass, field

from .model_mixin import ModelMixin
from ..utils.enum import ExtendedEnum
from ..components.latent_format import Wan21VAELatentFormat
from ..components.samplers.cfg_methods import default_cfg

@dataclass
class Latent:
    samples: Tensor | None = None
    batch_index: List[int] | None = None
    noise_mask: Tensor | None = None

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
    - specific options  (will group these if the list becomes big)
        - enable special kind of CFG
    '''
    def __init__(
        self, 
        model: ModelMixin, 
        model_sampling: Any, 
        model_arch_config: ModelArchConfig,
        disable_cfg: bool = False,
        cfg_func: Callable = default_cfg
    ):
        self.model = model
        self.model_sampling = model_sampling
        self.model_arch_config = model_arch_config
        
        self.disable_cfg = disable_cfg
        self.cfg_func = cfg_func
    
    def process_latent_in(self, latent_in: Tensor) -> Tensor:
        return self.model_arch_config.latent_format.process_in(latent_in)
    
    def process_latent_out(self, latent_out: Tensor) -> Tensor:
        return self.model_arch_config.latent_format.process_out(latent_out)