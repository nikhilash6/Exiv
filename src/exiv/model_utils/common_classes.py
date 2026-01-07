import torch
from torch import Tensor

from typing import Any, Callable, List, Tuple, Optional, Union, Dict
from dataclasses import dataclass, field

from exiv.utils.device import VRAM_DEVICE

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
        model: "ModelMixin", 
        model_sampling: Any = None,
        disable_cfg: bool = False,
        cfg_func: Callable = default_cfg
    ):
        self.model: "ModelMixin" = model
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
    EMBEDDING = "embedding"     # ref latent and aux signals go here
    LORA = "lora"

@dataclass
class ModelForwardInput:
    """
    Formatted conds that go in the forward pass (see prepare_model_input).
    Keeping a separate copy so as to not change the original conds.
    """
    cross_attn: Optional[Any] = None            # main prompt
    concat_map: Optional[Any] = None            # image + mask
    visual_embedding: Optional[Any] = None      # ipa / vision embeds
    controlnet: Optional[Any] = None            # controlnet signal
    time_hint: Optional[Any] = None             # time_dim_concat
    reference_latent: Optional[Any] = None      # vae encoded latent as a hint

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if v is not None}

# TODO: will be generalized more as more models are added
@dataclass
class ConcatConditioning:
    """
    Stuff directly concated to the input channels (e.g. Wan refs)
    """
    data: Tensor                   # VAE-encoded reference image (for now)
    mask: Tensor                   # inpainting mask
    mask_index: int = 0            # channel index to insert the mask (Model specific)

class AuxCondType:
    TIME_HINT = "time_hint"                 # time (duration, camera pose, etc.)
    REF_LATENT = "ref_latent"               # style transfer, audio timbre etc..
    VISUAL_EMBEDDING = "visual_embedding"   # ipa / vision embeds
    CONTROLNET = "controlnet"               # controlnet signal

@dataclass
class AuxConditioning:
    """
    Extra / Support signals for the generation process
    """
    type: str | None = None
    data: Optional[Tensor] = None
    timestep_range: Tuple[float, float] = (0.0, 1.0)    # (0.0=start, 1.0=end)
    frame_range: Optional[Tuple[int, int]] = None       # (start_idx, end_idx)

@dataclass
class Conditioning:
    """
    Common conditioning type that supports ALL conditioning inputs
    during the model inference
    """
    data: Tensor
    type: ConditioningType = ConditioningType.EMBEDDING

    # --- Timings & Ranges ---
    timestep_range: Tuple[float, float] = (-1, -1)          # (start, end), -1 means it spans the complete range
    frame_range: Optional[Tuple[int, int]] = (-1, -1)       # (start_idx, end_idx)

    # NOTE: this will be merged with the frame_range mask 
    # during output masking in the sampler in "combined_mask" property
    # (H, W): Spatial mask applied to all frames
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
    aux: Optional[List[AuxConditioning]] = None
    
    # model-specific extra params (not in use rn)
    extra: dict = field(default_factory=dict)
    
    # final processed inputs, ready for the inference step
    model_input: Optional[ModelForwardInput] = None

    def set_extra(self, **kwargs):
        self.extra.update(kwargs)
        return self

    def check_integrity(self):
        # per step strength (if provided) should match the timestep length
        pass
    
    def get_combined_mask(self, num_frames: int) -> torch.Tensor:
        """
        Combines spatial mask (H, W) with temporal frame_range to produce (T, H, W).
        """
        # TODO / NOTE: this method will break if the input latent is of a differnt shape than the mask
        # we are assuming that the user provides the correctly shaped mask as an input
        
        h, w = self.mask.shape
        device = VRAM_DEVICE
        
        if self.mask is not None:
            # default to full white if no mask provided
            mask = torch.ones((h, w), device=device, dtype=torch.float32)

        # (H, W) -> (T, H, W)
        mask = mask.unsqueeze(0).repeat(num_frames, 1, 1)

        if self.frame_range is not None:
            start, end = self.frame_range
            start = max(0, start)
            end = min(num_frames, end)

            # zero out frames outside the range
            if start > 0:
                mask[:start] = 0.0
            if end < num_frames:
                mask[end:] = 0.0
        
        return mask     # (T, H, W)
        

"""
NOTE: there are a couple of dataclasses and all of them are being used at different points in the code
will consolidate them later. here is a short summary of what is being used where -

Conditioning        -   user defined conditioning unit. passed as a group / batch
BatchedConditioning -   provided by the user. the Conditioning objects inside this don't have ModelForwardInput yet
ModelForwardInput   -   depending on what model format of conds model require this is created and attached to the 
                        respective Conditioining objects
ExecutionBatch      -   This is the final batch of conds, created after filtering and batching compatible conds, that
                        ultimtely gets passed to the model
"""
@dataclass
class ExecutionBatch:
    feed_x: Tensor
    feed_t: Tensor
    feed_input: Dict[str, Tensor] 
    
    group_names: List[str] 
    conds: List[Conditioning]

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

    def set_group_cond(self, group_name: str, conds: Union['Conditioning', List['Conditioning']], replace: bool = False):
        """
        Updates a conditioning group.
        - conds: Single object or list.
        - replace: If True, overwrites the group. If False, appends/extends.
        """
        if group_name not in self.groups:
            self.groups[group_name] = []
            if group_name not in self.execution_order:
                self.execution_order.append(group_name)
        
        if not isinstance(conds, list):
            conds = [conds]
            
        if replace:
            self.groups[group_name] = conds
        else:
            self.groups[group_name].extend(conds)
    
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