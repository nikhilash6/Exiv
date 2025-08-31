import weakref
import functools
import torch
import torch.nn as nn
import psutil

from .k_math import clamp
from .file import ensure_model_available
from .logging import app_logger
from ..constants import LOW_VRAM_MODE

LOADED_MODELS = []

class MemoryManager:
    available_memory = 0
    total_memory = 0
    
    @classmethod
    def update_mem_amount(cls, delta):
        cls.available_memory = clamp(cls.available_memory + delta, 0)
        return cls.available_memory, cls.total_memory
    
    @classmethod
    def get_mem_amount(cls):
        return cls.available_memory, cls.total_memory
    
    @classmethod
    def reset_mem_amount(cls, device="cpu"):
        device = torch.device(device)
        
        free_mem, total_mem = 4 * 1024**2, 4 * 1024**2
        if device.type == "cuda" or device.type == "hip":
            free_mem, total_mem = torch.cuda.mem_get_info(device)

        elif device.type == "mps":
            total_mem = psutil.virtual_memory().total
            driver_alloc = torch.mps.current_allocated_memory()
            # "free" = system RAM available - current allocated
            free_mem = max(total_mem - driver_alloc, 0)

        else:   # device.type == "cpu" 
            total_mem = psutil.virtual_memory().total
            free_mem = psutil.virtual_memory().available
            
        cls.available_memory, cls.total_memory = free_mem, total_mem


# bypassing weight creation at model init
class ModuleMeta(type):
    def __call__(cls, *args, **kwargs):
        with torch.device("meta"):
            instance = super().__call__(*args, **kwargs)
        if isinstance(instance, ModelMixin):    # mainly for safety
            instance.patch_forward_pass()
        return instance

class ModelMixin(nn.Module, metaclass=ModuleMeta):
    '''
    Adds additional feature to the base model
    
    - telemetry / stats
    - zero init loading
    - (TODO) multi gpu sharding
    - auto block swaping during low memory
    - (TODO) priority swapping
    - quantization support
    - safetensor support
    - URL download support
    '''
    def __init__(self, device, dtype):
        self.device = device
        self.dtype = dtype
        self.model_path = None
        
    def _patch_modules_forward(self):
        current = 0
        self.full_load = []

        app_logger.debug("******** patching modules")
        def _modified_forward(obj, *args, **kwargs):
            og_forward, module = kwargs["og_forward"], kwargs["module"]
            del kwargs["og_forward"]
            del kwargs["module"]
            
            try:
                app_logger.debug("Moving ", module.__class__.__name__, f" to {self.device}")
                module.to(self.device)
                out = og_forward(obj, *args, **kwargs)
            finally:
                app_logger.debug("Moving back ", module.__class__.__name__, " to cpu")
                module.to("cpu")
            return out

        for m in self.modules():
            if m is self or not self.is_leaf_module(m):
                continue

            current += self._module_size(m)
            if current < MemoryManager.available_memory - 50_000_000:  # 50 MB buffer
                self.full_load.append(weakref.ref(m))
            else:
                og_forward = m.forward
                m.forward = functools.partial(_modified_forward, og_forward=og_forward, module=m)
                
        # load initial full_load modules
        for m_ref in self.full_load:
            m = m_ref()
            if m is not None:
                m.to(self.device)
        
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
        return ms
    
    # overridden
    def __call__(self, *args, **kwargs):
        pass
    
    def fast_load(
        self,
        model_path,                     # path or url
        device: torch.device,           # device to load this on
        torch_dtype,                    # 'auto' or 'torch.dtype'
        force_download=False,           # re_download models
        download_path=None,             # defaults to folder util
    ):
        assert model_path is not None, "model_path is required"
        model_path = ensure_model_available(model_path, download_path, force_download)
                    
        if LOW_VRAM_MODE: device = "cpu"
        # zero init load state dict
        self.load_state_dict(torch.load(model_path, map_location=device), assign=True)
