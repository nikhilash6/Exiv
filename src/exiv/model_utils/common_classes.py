import torch
from torch import Tensor

from typing import Any, Callable, List, Tuple, Optional, Union, Dict
from dataclasses import dataclass, field

from .model_mixin import ModelMixin
from ..utils.enum import ExtendedEnum
from ..components.latent_format import LatentFormat
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
    

    def scale_latent_inpaint(self, sigma, noise, latent_image, **kwargs):
        # NOTE: we generally reverse the scaling this will go through with 'calculate_input'
        # but there can be model specific overrides
        if hasattr(self.model, 'scale_latent_inpaint'):
            return self.model.scale_latent_inpaint(sigma, noise, latent_image, **kwargs)

        return self.model_sampling.noise_scaling(sigma.reshape([sigma.shape[0]] + [1] * (len(noise.shape) - 1)), noise, latent_image)

class ConditioningType(ExtendedEnum):
    EMBEDDING = "embedding"
    CONTROLNET = "controlnet"
    LORA = "lora"
    IPA = "ipa"

@dataclass
class ModelForwardInput:
    """
    Formatted conds that go in the forward pass (see format_conds)
    """
    cross_attn: Optional[Any] = None            # main prompt
    concat_map: Optional[Any] = None            # image + mask
    visual_embedding: Optional[Any] = None      # 'ref_latent' or 'clip_fea'
    controlnet: Optional[Any] = None            # controlnet signal
    time_hint: Optional[Any] = None             # time_dim_concat

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if v is not None}

@dataclass
class ConcatConditioning:
    """
    Right now structured container for Image-to-Video / Inpainting specific inputs,
    will update later as new inputs are added
    """
    data: Tensor                   # VAE-encoded reference image (for now)
    mask: Tensor                   # inpainting mask
    mask_index: int = 0            # channel index to insert the mask (Model specific)

@dataclass
class AuxiliaryConditioning:
    """
    Extra / Support signals for the generation process
    """
    time_hint: Optional[Tensor] = None                  # time (duration, camera pose, etc.)
    reference_latents: Optional[List[Tensor]] = None    # style transfer, audio timbre etc..

@dataclass
class Conditioning:
    """
    Common conditioning type that supports ALL conditioning inputs
    during the model inference
    """
    data: Tensor
    type: ConditioningType = ConditioningType.EMBEDDING

    # --- Timings & Ranges ---
    timestep_range: Tuple[float, float] = (0.0, 1.0)    # (0.0=start, 1.0=end)
    frame_range: Optional[Tuple[int, int]] = None       # (start_idx, end_idx)

    # --- Spatial & Temporal Masks ---
    # Supports:
    # - (1, H, W): Spatial mask applied to all frames
    # - (T, H, W): Unique mask per frame
    # - (B, C, T, H, W): Full explicit mask
    mask: Tensor | None = None

    # --- Strength / Intensity ---
    # Supports:
    # - float: Constant strength (e.g. 1.0)
    # - Tensor: Per-frame/pixel strength (e.g. shape [T] or [B, H, W])
    # - List[float]: Per-step strength schedule (e.g. [0.0, 0.5, 1.0...])
    strength: Union[float, Tensor, List[float]] = 1.0

    # structural / geometry
    concat: Optional[ConcatConditioning] = None
    # auxiliary / modifiers
    aux: Optional[AuxiliaryConditioning] = None
    
    # model-specific extra params (not in use rn)
    extra: dict = field(default_factory=dict)
    
    # final processed inputs, ready for the inference step
    model_input: Optional[ModelForwardInput] = None

    def set_extra(self, **kwargs):
        self.extra.update(kwargs)
        return self


@dataclass
class BatchedConditioning:
    """
    Holds all conditioning groups and defines how they map to model batches,
    explictly defining the inference batches
    """
    # store lists of Conditioning objects by name
    # e.g. {"positive": [...], "negative": [...], "neutral": [...]}
    groups: Dict[str, List['Conditioning']] = field(default_factory=dict)

    # explicitly define the execution order and the batch size
    # e.g. ["positive", "negative"] implies batch_size=2
    execution_order: List[str] = field(default_factory=lambda: ["positive", "negative"])

    def add_cond_to_group(self, group_name: str, cond: 'Conditioning'):
        """
        Adds a single Conditioning object to a specific group.
        Creates the group if it doesn't exist.
        """
        if group_name not in self.groups:
            self.groups[group_name] = []
            # auto-add to execution order if it's a new group? 
            # usually safer to let user define order explicitly, but flexible logic:
            if group_name not in self.execution_order:
                self.execution_order.append(group_name)
        
        self.groups[group_name].append(cond)
    
    def set_execution_order(self, order: List[str]):
        """
        Manually define the batch structure.
        e.g. set_execution_order(["positive", "negative", "neutral"])
        """
        requested = set(order)
        available = set(self.groups.keys())

        missing_in_data = requested - available
        missing_in_order = available - requested
        
        if missing_in_data or missing_in_order:
            error_msg = "Execution order mismatch!\n"
            if missing_in_data:
                error_msg += f"  - Requested groups not found in data: {missing_in_data}\n"
            if missing_in_order:
                error_msg += f"  - Data groups missing from execution order: {missing_in_order}\n"
            
            raise ValueError(error_msg)
        
        self.execution_order = order
        
    def get_groups_in_order(self):
        return [(name, self.groups[name]) for name in self.execution_order]