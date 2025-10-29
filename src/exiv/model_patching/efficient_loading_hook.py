import torch

import weakref
from typing import List, Any, Tuple

from .hook_registry import HookRegistry, HookType, ModelHook
from ..utils.logging import app_logger
from ..utils.device import OFFLOAD_DEVICE, RESERVED_MEM, MemoryManager
from ..config import LOADING_MODE, global_config

# move module to the gpu_device
def load_module(model, module, module_name, quant_enabled=False):
    if module is None: return   # m_ref can turn out to be None
    
    app_logger.debug(f"Moving {module.__class__.__name__} to {model.gpu_device}")
    
    module_class_name = module.__class__.__name__
    is_bnb_module = module_class_name in ["Linear8bitLt", "Linear4bit"]

    if any(p.device.type == "meta" for p in module.parameters(recurse=False)):
        module.to_empty(device=model.gpu_device)
    
    elif is_bnb_module:
        device_index = torch.device(model.gpu_device).index
        if device_index is None:
             device_index = torch.cuda.current_device() # Get default index if "cuda"
        
        # .cuda(device_index) is overridden by bnb
        module.cuda(device_index)
        if hasattr(module.weight, "CB"):
            module.weight.CB = module.weight.CB.cuda(device_index)
            
        if hasattr(module.weight, "SCB"):
            module.weight.SCB = module.weight.SCB.cuda(device_index)
    
    else:
        # standard .to() for all other regular modules
        module.to(device=model.gpu_device)

    app_logger.debug(f"modules rn: {[m.__class__.__name__ for mn, m in model.named_modules() if m != model]}")
    
    MemoryManager.clear_memory()


class EfficientModelLoaderHook(ModelHook):
    """
    this hook loads the initial set of full_load modules before the main model's forward call
    """
    
    def __init__(self, full_load: List[Tuple[weakref.ref, str]]):
        self.hook_type = HookType.EFFICIENT_MODEL_LOADER.value
        self.full_load = full_load
    
    # module here is the main model
    def pre_forward(self, module, *args, **kwargs):
        app_logger.debug(f"full load modules: {[m_name for m_name in self.full_load]}")
        self._full_load(module)
        return args, kwargs
    
    def _full_load(self, model):
        if getattr(model, "_fully_loaded", False): return
        # load initial full_load modules
        for m_ref, m_name in self.full_load:
            m = m_ref()
            app_logger.debug(f"Loading via full load: {m_name}")
            load_module(model=model, module=m, module_name=m_name, quant_enabled=True)

        model._fully_loaded = True
    

class EfficientModuleLoaderHook(ModelHook):
    """
    this hook is responsible for loading / unloading individual modules during the
    runtime / forward call. each one of these hook receives a "full_load_module" flag that indicates
    if that module is already loaded during the full_load operation or not
    """
    
    def __init__(self, model_ref: 'ModelMixin', module_name: str, full_load_module: bool = False):
        super().__init__()
        
        self.hook_type = HookType.EFFICIENT_MODULE_LOADER.value
        self.model_ref = weakref.ref(model_ref) 
        self.module_name = module_name
        self.full_load_module = full_load_module

    def pre_forward(self, module: torch.nn.Module, *args, **kwargs):
        model = self.model_ref()
        if model is None:
            return args, kwargs

        if not self.full_load_module:
            app_logger.debug(f"Loading via hook: {module.__class__.__name__}")
            load_module(
                model=model,
                module=module,
                module_name=self.module_name
            )
            
        return args, kwargs

    def post_forward(self, module: torch.nn.Module, output: Any):
        model = self.model_ref()
        if model is None: 
            return output
        
        if not self.full_load_module:
            app_logger.debug(f"Moving back {module.__class__.__name__} to cpu via hook")
            module.to(OFFLOAD_DEVICE)
            MemoryManager.clear_memory()
            
        return output
    
    
"""
There are three modes for loading the model:
1. NO_OOM   => This loads / unloads every single module for each step
2. LOW_VRAM => This initially loads a set of 'full_load' modules that remain on the VRAM
                until the inference is completed. All the rest of the modules are 
                loaded / unloaded dynamically
3. NORMAL   => This loads the entire model in one go and keeps it in VRAM for the entire inference
"""

@staticmethod
def split_model_for_loading(model: 'ModelMixin'):
    # this determines which modules can be fully loaded permanently on the vram
    # and which has to be dynamically loaded
    full_load_modules: List[Tuple[weakref.ref, str]] = []
    
    current_mem_used = 0
    available_mem = MemoryManager.available_memory(model.gpu_device) - RESERVED_MEM
    
    module_by_size = []     # contains modules sorted by size (asc)
    for m_name, m in model.named_modules():
        if m is model or not model.is_leaf_module(m):
            continue
        module_by_size.append((model._module_size(m), m_name, m))
    
    module_by_size.sort(key=lambda x: x[0])
    
    # calculating max_after, that gives the max element after the current element
    # in the sorted module_by_size array
    sizes = [s for s, _, _ in module_by_size]
    max_after = [None] * len(sizes)
    current_max = float('-inf')
    for i in reversed(range(len(sizes))):
        max_after[i] = current_max if i < len(sizes) - 1 else None
        current_max = max(current_max, sizes[i])
    
    
    for (m_size, m_name, m), max_next in zip(module_by_size, max_after):
        if m_size >= available_mem:
            raise RuntimeError(f"Single layer mem size of {m_size} exceeds the total available memory of {available_mem}")
        
        current_mem_used += m_size
        if current_mem_used < available_mem and (max_next is None or max_next < (available_mem - current_mem_used)):
            # if we can add this to the existing available mem limit + after adding this the next
            # biggest module can be safely loaded / unloaded, then we fully load this
            full_load_modules.append((weakref.ref(m), m_name))
        else:
            break
        
    return full_load_modules


def enable_efficient_loading(model: 'ModelMixin'):
    """
    This patches the forward pass of modules to dynamically load / unload them
    """
    full_load: List[Tuple[weakref.ref, str]] = []
    loading_mode = global_config.loading_mode
    
    if loading_mode == LOADING_MODE.NO_OOM.value:
        # no full_load modules in this
        full_load = []
        
    elif loading_mode == LOADING_MODE.LOW_VRAM.value:
        # full_load modules
        full_load = split_model_for_loading(model)

    else:
        # normal load, everything should be in full_load
        for m_name, m in model.named_modules():
            if m is model or not model.is_leaf_module(m):
                continue
            
            full_load.append((weakref.ref(m), m_name))
    
    for m_name, m in model.named_modules():
        if m is model or not model.is_leaf_module(m):
            continue
        
        module_hook = EfficientModuleLoaderHook(
            model_ref=model,
            module_name=m_name,
            full_load_module=any(m_name == mn for _, mn in full_load),
        )
        HookRegistry.apply_hook_to_module(m, module_hook)


    model_hook = EfficientModelLoaderHook(full_load=full_load)
    HookRegistry.apply_hook_to_module(model, model_hook)
    