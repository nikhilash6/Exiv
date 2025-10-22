import os
import functools
import torch
import torch.nn as nn
import safetensors

from ..utils.device import VRAM_DEVICE
from ..utils.file import ensure_model_available
from ..utils.logging import app_logger
from ..config import global_config, BYTES_IN_MB
from ..quantizers.base import Quantizer
from .model_loader import EfficientModelLoader


# bypassing weight creation at model init
class ModuleMeta(type(nn.Module)):
    def __call__(cls, *args, **kwargs):
        model_dtype = kwargs.pop("dtype", torch.float32)
        original_dtype = torch.get_default_dtype()
        
        try:
            torch.set_default_dtype(model_dtype)
            
            # zero init weight load
            with torch.device("meta"):
                instance = super().__call__(*args, **kwargs)
                if isinstance(instance, ModelMixin):    # mainly for safety
                    EfficientModelLoader.patch_forward_pass(instance)   # optimizations for dynamic model loading
                    if not hasattr(instance, 'dtype'):
                        instance.dtype = model_dtype
                        
        finally:
            torch.set_default_dtype(original_dtype)
        
        return instance


class ModelMixin(nn.Module, metaclass=ModuleMeta):
    '''
    Adds additional feature to the base model
    
    - (TODO) telemetry / stats
    - (TODO) better / modular patch system
    - zero init loading
    - (TODO) multi gpu sharding
    - auto block swapping during low memory
    - (TODO) priority swapping
    - (TODO) cuda streams for offloading
    - quantization support
    - safetensor support
    - URL download support
    '''
    def __init__(self, device: str = None, quantizer: Quantizer = None, model_path: str = None, dtype = torch.float32):     # dtype is used by the meta class
        super().__init__()
        self.gpu_device = device or VRAM_DEVICE
        self.model_path = model_path
        self._patched = False
        self._fully_loaded = False
        
        self.quantizer = quantizer
    
    def clear_cache(self):
        # add other cleanup in future
        self.__class__.clear_cls_cache()

    @classmethod
    def clear_cls_cache(cls):
        cls._module_size.cache_clear()
        cls.is_leaf_module.cache_clear()
    
    @staticmethod
    @functools.lru_cache(maxsize=None)
    def is_leaf_module(module: nn.Module) -> bool:
        return len(list(module.children())) == 0

    @staticmethod
    @functools.lru_cache(maxsize=None)
    def _module_size(module: nn.Module):
        ms = 0
        for param in module.parameters(recurse=False):
            ms += param.nelement() * param.element_size()
        return round(ms / BYTES_IN_MB, 2)
    
    def __call__(self, *args, **kwargs):
        with torch.inference_mode():
            # moving the inputs to GPU
            new_args = tuple(a.to(self.gpu_device, non_blocking=True) if torch.is_tensor(a) else a for a in args)
            new_kwargs = {k: (v.to(self.gpu_device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in kwargs.items()}
            return super().__call__(*new_args, **new_kwargs)
    
    # TODO: support GGUF loading
    def load_model(
        self,
        model_path = None,              # path or url   (passing this in the init as well for flexibility)
        device = None,                  # device to load this on
        force_download=False,           # re_download models
        download_path=None,             # defaults to folder util
    ):
        model_path = model_path or self.model_path
        assert model_path is not None, "model_path is required"
        
        self.gpu_device = device or self.gpu_device
        model_path = ensure_model_available(model_path, download_path, force_download)
        self.load_state_dict(ModelMixin.get_state_dict(model_path), assign=True)
    
    # code adapted from ComfyUI
    @staticmethod
    def get_state_dict(model_path, device=torch.device("cpu")):
        file_extension = os.path.basename(model_path).split(".")[-1]
        if file_extension in ["safetensors", "sft"]:
            try:
                # safetensor's zero copy loading (pt - pytorch)
                kwargs = {"framework": "pt"}
                # safetensors only support cpu and cuda, and doesn't take cpu as param
                if device.type == "cuda": kwargs["device"] = device.type
                with safetensors.safe_open(model_path, **kwargs) as f:
                    sd = {}
                    for k in f.keys():
                        tensor = f.get_tensor(k)    # loading one key at a time; low mem pressure
                        if global_config.disable_mmap:
                            # moving to device (no zero copying)
                            tensor = tensor.to(device=device, copy=True)
                        sd[k] = tensor
            except Exception as e:
                app_logger.error(str(e))
                raise e
        else:   # ckpt, pth, pt
            torch_args = {}
            # using simple flags rn, will fix later
            if not global_config.disable_mmap: torch_args["mmap"] = True
            if global_config.always_safe_load: torch_args["weights_only"] = True
            
            sd = torch.load(model_path, map_location=device, **torch_args)
            if "state_dict" in sd:  
                sd = sd["state_dict"]   # loading state_dict if available
            elif len(sd) == 1:          # loading the first key (if it's a dict)
                val = next(iter(sd.values()))
                sd = val if isinstance(val, dict) else sd
                
        return sd