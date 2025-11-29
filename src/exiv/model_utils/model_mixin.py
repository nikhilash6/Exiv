import torch
import torch.nn as nn
from torch import Tensor

from typing import List
import uuid

from .lora_mixin import LoraMixin
from .helper_methods import get_state_dict, set_module_tensor_to_device
from ..utils.dtype import cast_to
from ..utils.device import VRAM_DEVICE,ProcDevice
from ..utils.file import ensure_model_available
from ..utils.logging import app_logger
from ..config import BYTES_IN_MB
from ..quantizers.base import QuantType, Quantizer, get_quantizer
from ..model_patching.efficient_loading_hook import enable_efficient_loading


# bypassing weight creation at model init
class ModuleMeta(type(nn.Module)):
    def __call__(cls, *args, **kwargs):
        model_dtype = kwargs.pop("dtype", torch.float32)
        quant_type = kwargs.get("quant_type", None)
        force_load_mode = kwargs.get("force_load_mode", None)
        original_dtype = torch.get_default_dtype()
        
        try:
            torch.set_default_dtype(model_dtype)
            
            # zero init weight load
            with torch.device("meta"):
                instance = super().__call__(*args, **kwargs)
                quantizer: Quantizer = get_quantizer(quant_type=quant_type)
                instance.quantizer = quantizer
                if quantizer is not None:
                    quantizer.validate_environment()
                    quantizer.process_model_before_weight_loading(model=instance)
                
                if isinstance(instance, ModelMixin):    # mainly for safety
                    instance.force_load_mode = getattr(instance, "force_load_mode", None) or force_load_mode
                    enable_efficient_loading(instance)  # kinda default hook
                    if not getattr(instance, 'dtype', None):
                        instance.dtype = model_dtype
                        
        finally:
            torch.set_default_dtype(original_dtype)
        
        return instance


class ModelMixin(nn.Module, LoraMixin, metaclass=ModuleMeta):
    '''
    Adds additional feature to the base model
    
    - (TODO) telemetry / stats
    - lora patching
    - zero init loading
    - (TODO) multi gpu sharding
    - (TODO) implement low cpu mem usage feature
    - auto block swapping during low memory
    - (TODO) priority swapping
    - (TODO) cuda streams for offloading
    - (TODO) support GGUF loading
    - quantization support
    - safetensor support
    - (TODO) improve safetensor loading
    - URL download support
    '''
    def __init__(self, device: str = None, quant_type: QuantType = None, model_path: str = None, dtype = torch.float32, **kwargs):     # quant_type, force_load_mode, dtype is used by the meta class
        super().__init__()
        LoraMixin.__init__(self)
        
        self.gpu_device = device or VRAM_DEVICE
        self.model_path = model_path
        self.model_arch_config = None
    
    def clear_cache(self):
        # TODO: legacy code, will remove after a final check
        pass
    
    @staticmethod
    def is_leaf_module(module: nn.Module) -> bool:
        return len(list(module.children())) == 0
    
    @staticmethod
    def has_orphan_params(module: nn.Module) -> bool:
        return len(list(module.parameters(recurse=False))) > 0 or \
                      len(list(module.buffers(recurse=False))) > 0

    @staticmethod
    def _module_size(module: nn.Module):
        ms = 0
        for param in module.parameters(recurse=False):
            ms += param.nelement() * param.element_size()
        return round(ms / BYTES_IN_MB, 2)
    
    def __call__(self, *args, **kwargs):
        def original_call(*args, **kwargs):
            with torch.inference_mode():
                # moving the inputs to GPU
                app_logger.debug(f"moving the inputs to {self.gpu_device}")
                new_args = tuple(cast_to(a, device=self.gpu_device, dtype=self.dtype) if torch.is_tensor(a) else a for a in args)
                new_kwargs = {k: (cast_to(v, device=self.gpu_device, dtype=self.dtype) if torch.is_tensor(v) else v) for k, v in kwargs.items()}

                return super(ModelMixin, self).__call__(*new_args, **new_kwargs)
            
        registry = getattr(self, "hook_registry", None)
        if registry and registry.head.next_hook != registry.tail:
            wrapped_call = registry.get_modified_call(original_call)
            return wrapped_call(*args, **kwargs)
        else:
            return original_call(*args, **kwargs)

    # code adapted from Huggingface Diffusers
    def load_model(
        self,
        model_path = None,              # model file path (override for flexibility)
        force_download=False,           # re_download models
        download_url=None,              # file url (optional)
        dtype=None                      # TODO: hardware specific dtype
    ):
        model_path = model_path or self.model_path
        assert model_path is not None, "model_path is required"
        # loading everything on the CPU, then modularly offloading to the GPU
        device = ProcDevice.CPU.value
        self.dtype = dtype or self.dtype
        
        model_path = ensure_model_available(model_path, download_url, force_download)
        
        state_dict = get_state_dict(model_path)
        model_state_dict = self.state_dict()
        
        for param_name, param in state_dict.items():
            if param_name not in model_state_dict: 
                app_logger.warning(f"skipping the param {param_name} as it's not present in the model definition")
                continue
            
            if self.dtype is not None:
                if self.quantizer is not None:
                    pass    # not overiding dtype of quantized models
                else:
                    param = param.to(self.dtype)
            
            # For compatibility with PyTorch load_state_dict which converts state dict dtype to existing dtype in model, and which
            # uses `param.copy_(input_param)` that preserves the contiguity of the parameter in the model
            # Reference: https://github.com/pytorch/pytorch/blob/db79ceb110f6646523019a59bbd7b838f43d4a86/torch/nn/modules/module.py#L2040C29-L2040C29
            old_param = self
            splits = param_name.split(".")
            for split in splits:
                # recursively drill down: model.down_blocks[0].attentions[0].proj_in.weight
                old_param = getattr(old_param, split)
            
            # param_name might be for a buffer or something not loadable, skip it
            if not isinstance(old_param, (torch.nn.Parameter, torch.Tensor)):
                old_param = None
                
            if old_param is not None:
                if self.dtype is None:
                    param = param.to(old_param.dtype)
                    
                if old_param.is_contiguous():
                    param = param.contiguous()
            
            # bnb params are flattened.
            # gguf quants have a different shape based on the type of quantization applied
            if model_state_dict[param_name].shape != param.shape:
                if self.quantizer is not None:
                    self.quantizer.check_quantized_param_shape(param_name, model_state_dict[param_name], param)
                else:
                    raise ValueError(
                        f"Cannot load {model_path} because {param_name} expected shape {model_state_dict[param_name].shape}, but got {param.shape}."
                    )
            
            # final assignment
            if self.quantizer is not None and self.quantizer.check_if_quantized_param(
                self, param, param_name, state_dict, dtype=self.dtype
            ):
                self.quantizer.create_quantized_param(
                    self,
                    param,
                    param_name,
                    device,
                    state_dict,
                    dtype=self.dtype
                )
            else:
                set_module_tensor_to_device(self, param_name, device, value=param, dtype=self.dtype)
        
        if hasattr(self, "create_model_lora_key_map"):
            self.create_model_lora_key_map(state_dict)
        else:
            app_logger.warning("Unable to create key map for model and lora keys. Lora loading won't be supported")
    
    def process_latent_in(self, latent_in: Tensor) -> Tensor:
        assert self.model_arch_config is not None, "model_arch_config not set"
        return self.model_arch_config.latent_format.process_in(latent_in)
    
    def process_latent_out(self, latent_out: Tensor) -> Tensor:
        assert self.model_arch_config is not None, "model_arch_config not set"
        return self.model_arch_config.latent_format.process_out(latent_out)
    
    def prepare_conds_for_model(self, cond_group_name: str, cond_list: List[List], noise, **kwargs):
        out = []
        for cond in cond_list:
            # --- Step 1: Standardization ---
            # cond : [tensor, dict]
            temp = cond[1].copy()
            model_conds = {}
            
            # move the text embedding to a standard key for the model to find later
            if cond[0] is not None:
                temp["cross_attn"] = cond[0]
                
            temp["model_conds"] = model_conds
            temp["uuid"] = uuid.uuid4()         # differentiating conds

            # --- Step 2: Formatting  ---
            params = temp.copy()
            params["device"] = self.gpu_device
            params["noise"] = noise
            
            spatial_compression_factor = kwargs.get("spatial_compression_factor", 8)
            if len(noise.shape) >= 4:
                params["width"] = params.get("width", noise.shape[3] * spatial_compression_factor)
                params["height"] = params.get("height", noise.shape[2] * spatial_compression_factor)
                
            params["prompt_type"] = params.get("prompt_type", cond_group_name)
            
            # extra arguments (like latent_image, denoise_mask)
            for k in kwargs:
                if k not in params:
                    params[k] = kwargs[k]

            # **** model specific formatting ****
            formatted_results = self.format_conds(**params)

            # --- Step 3: Update the 'model_conds' dictionary ---
            current_model_conds = temp['model_conds'].copy()
            for k in formatted_results:
                current_model_conds[k] = formatted_results[k]
                
            temp['model_conds'] = current_model_conds
            out.append(temp)
            
        return out
    
    def format_conds(self):
        # this formats the conds to a format as required by the underlying model
        raise NotImplementedError("Child instance has not overriden this empty impl.")
