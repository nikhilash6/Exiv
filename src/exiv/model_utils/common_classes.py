import torch
from torch import Tensor

from typing import Any, Callable, List
from dataclasses import dataclass, field

from .model_mixin import ModelMixin
from ..utils.enum import ExtendedEnum
from ..components.latent_format import LatentFormat, Wan21VAELatentFormat
from ..components.samplers.cfg_methods import default_cfg

@dataclass
class Latent:
    samples: Tensor | None = None
    batch_index: List[int] | None = None
    noise_mask: Tensor | None = None

class ModelArchConfig:
    # will extend this as more models are added
    latent_format: LatentFormat = None

class ModelWrapper:
    '''
    contains
    - the main model
    - methods of sampling
    - specific options  (will group these if the list becomes big)
        - enable special kind of CFG
    '''
    def __init__(
        self, 
        model: ModelMixin, 
        model_sampling: Any = None,
        disable_cfg: bool = False,
        cfg_func: Callable = default_cfg
    ):
        self.model: ModelMixin = model
        self.model_sampling = model_sampling or model.get_model_sampling_obj()
        
        self.disable_cfg: bool = disable_cfg
        self.cfg_func: Callable = cfg_func
    
    # TODO: very bad practice, this should be in the modelmixin
    def scale_latent_inpaint(self, sigma, noise, latent_image, **kwargs):
        # return self.model_sampling.noise_scaling(sigma.reshape([sigma.shape[0]] + [1] * (len(noise.shape) - 1)), noise, latent_image)
        return latent_image