import torch
import torch.nn as nn
from torch.nn import init
from contextlib import contextmanager

from .k_math import clamp

class MemoryManager:
    available_memory = 0
    
    @classmethod
    def update_mem_amount(cls, delta):
        cls.available_memory = clamp(cls.available_memory + delta, 0)
    
    @classmethod
    def get_mem_amount(cls):
        return cls.available_memory
    
    @classmethod
    def reset_mem_amount(cls):
        # TODO: add code to fetch fresh memory through hardware APIs
        cls.available_memory = 1000

# bypassing weight creation at model init
@contextmanager
def no_init_weights():
    with torch.device("meta"):
        yield

class ModelMixin(nn.Module):
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
    
    def fast_load(self):
        '''
        - zero init loading
        - model sharding
        - device mapping / auto load - offload (using memory manager)
        - quantization
        - safetensor loading
        '''
        pass
    
    
    