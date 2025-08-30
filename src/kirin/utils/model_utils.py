import torch
import torch.nn as nn
import psutil

from .k_math import clamp

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
            driver_alloc = torch.mps.driver_allocated_memory()
            # "free" = system RAM available - driver allocated
            free_mem = max(total_mem - driver_alloc, 0)

        else:   # device.type == "cpu" 
            total_mem = psutil.virtual_memory().total
            free_mem = psutil.virtual_memory().available
            
        cls.available_memory, cls.total_memory = free_mem, total_mem


# bypassing weight creation at model init
class InitMeta(type):
    def __call__(cls, *args, **kwargs):
        with torch.device("meta"):
            instance = super().__call__(*args, **kwargs)
        return instance

class ModelMixin(nn.Module, metaclass=InitMeta):
    '''
    Manages available memory (auto load / offload), telemetry and 
    efficient model loading
    '''
    def __init__(self, device, dtype):
        self.device = device
        self.dtype = dtype
    
    # overridden
    def __call__(self, *args, **kwargs):
        pass
    
    def fast_load(
        self,
        model_path,                     # path or url
        device,                         # device to load this on
        torch_dtype,                    # 'auto' or 'torch.dtype'
        force_download=False,           # re_download models
        download_path=None,             # defaults to folder util
    ):
        '''
        - zero init loading
        - model sharding
        - device mapping / auto load - offload (using memory manager)
        - quantization support
        - safetensor support
        - URL download support
        '''
        pass
    
    
    