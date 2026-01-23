import torch

import weakref
from typing import Callable, List, Any, Tuple

from .common import prepare_and_cache_cpu_state, restore_cpu_state
from .hook_registry import FeatureType, HookLocation, HookRegistry, HookType, ModelHook, register_hook_method
from ..model_utils.helper_methods import estimate_peak_activation_size
from ..utils.logging import app_logger
from ..utils.device import OFFLOAD_DEVICE, RESERVED_MEM, MemoryManager
from ..config import BYTES_IN_MB, LOADING_MODE, global_config


def move_module_or_params(model, module, target_device, module_name=None):
    from ..model_utils.helper_methods import move_module, move_immediate_params
    
    # if it has params then only the params are moved (as children are moved through their own hook)
    # or else the entire module is moved (leaf/children)
    if model.is_leaf_module(module):
        move_module(model=model, module=module, module_name=module_name, target_device=target_device)
    elif model.has_orphan_params(module):
        move_immediate_params(module=module, device=target_device)
        
def should_preload(model, module):
    # we only preload modules that are either leaves or they have params that needs to be moved as well
    return model.is_leaf_module(module) or model.has_orphan_params(module)

class EfficientModelLoaderHook(ModelHook):
    """
    this hook loads the initial set of full_load modules before the main model's forward call
    """
    
    def __init__(self, full_load: List[Tuple[weakref.ref, str]]):
        self.hook_type = HookType.EFFICIENT_MODEL_LOADER.value
        self.hook_location = HookLocation.FORWARD.value
        self.full_load = full_load
    
    # module here is the main model
    def execute(self, module, original_fn: Callable, *args, **kwargs):
        # ------ pre forward hook
        app_logger.info(f"*****##### Loading {module.__class__.__name__}")
        app_logger.debug(f"full load modules count: {len(self.full_load)}")
        self._full_load(module)
        return original_fn(*args, **kwargs)
    
    def _full_load(self, model):
        
        total = 0
        if getattr(model, "_fully_loaded", False): return
        # load initial full_load modules
        for m_ref, m_name in self.full_load:
            m = m_ref()
            s = model._module_size(m)
            total += s
            app_logger.debug(f"Loading via full load: {m_name} , size: {s}, total: {total}")
            move_module_or_params(model=model, module=m, target_device=model.gpu_device, module_name=m_name)

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
        self.hook_location = HookLocation.FORWARD.value
        
        self.model_ref = weakref.ref(model_ref) 
        self.module_name = module_name
        self.full_load_module = full_load_module

    def execute(self, module: torch.nn.Module, original_fn: Callable, *args, **kwargs):
        model = self.model_ref()
        if model is None:
            return original_fn(*args, **kwargs)

        cpu_cache = None
        if not self.full_load_module:
            cpu_cache = prepare_and_cache_cpu_state(module)
                
            app_logger.debug(f"Loading via hook: {self.module_name}")
            move_module_or_params(
                model=model,
                module=module,
                target_device=model.gpu_device,
                module_name=self.module_name
            )
        
        # lora patching
        delta = None
        if getattr(module, "weight", None) is not None:
            current_step = getattr(model, "current_time_step", -1)
            model_key = f"{self.module_name}.weight"
            delta = model.get_combined_delta(
                model_key=model_key,
                timestep=current_step,
                target_device=module.weight.device,
                target_dtype=module.weight.dtype
            )
        
        if delta is not None:
            app_logger.debug("---- patching lora weights delta")
            module.weight.data.add_(delta)
            self._applied_delta = delta
        
        try:
            output = original_fn(*args, **kwargs)
        finally:
            # ----- post forward unpatch lora
            if getattr(self, "_applied_delta", None) is not None:
                app_logger.debug("---- UNpatching lora weights delta")
                module.weight.data.sub_(self._applied_delta)
                self._applied_delta = None
            
            if cpu_cache is not None:
                restore_cpu_state(module, cpu_cache)

        return output
    
"""
There are three modes for loading the model:
1. NO_OOM   => This loads / unloads every single module for each step
2. LOW_VRAM => This initially loads a set of 'full_load' modules that remain on the VRAM
                until the inference is completed. All the rest of the modules are 
                loaded / unloaded dynamically
3. NORMAL   => This loads the entire model in one go and keeps it in VRAM for the entire inference
"""

def split_model_for_loading(model: 'ModelMixin', target_shape = None):
    MemoryManager.clear_memory()
    loading_mode = getattr(model, 'force_load_mode', None) or global_config.loading_mode
    
    current_mem_used = 0
    act_mb = estimate_peak_activation_size(model.get_memory_footprint_params(), target_shape)
    runtime_mem_usage = max(RESERVED_MEM, act_mb)
    if loading_mode == LOADING_MODE.NO_OOM.value:
        # no full_load modules in this
        # returning the largest leaft module's size
        for m_name, m in model.named_modules():
            if m is model or not model.is_leaf_module(m):
                continue
            current_mem_used = max(current_mem_used, model._module_size(m)) 
        return [], current_mem_used + runtime_mem_usage
    
    elif loading_mode == LOADING_MODE.NORMAL_LOAD.value:
        # normal load, everything should be in full_load
        full_load: List[Tuple[weakref.ref, str]] = []
        for m_name, m in model.named_modules():
            if m is model or not should_preload(model, m):
                continue
            # NOTE / TODO: here we are ignoring the weights of the internal params
            # if those have large weights then this would cause problems with mem calc down the line
            current_mem_used += model._module_size(m) if model.is_leaf_module(m) else 0
            full_load.append((weakref.ref(m), m_name))
        return full_load, current_mem_used + runtime_mem_usage
    
    elif loading_mode == LOADING_MODE.LOW_VRAM.value:
        # this determines which modules can be fully loaded permanently on the vram
        # and which has to be dynamically loaded
        full_load_modules: List[Tuple[weakref.ref, str]] = []
        available_mem = MemoryManager.available_memory(model.gpu_device) - runtime_mem_usage
        
        module_by_size = []     # contains modules sorted by size (asc)
        for m_name, m in model.named_modules():
            if m is model or not should_preload(model, m):
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
            
        return full_load_modules, current_mem_used + runtime_mem_usage

def remove_efficient_loading(model: 'ModelMixin'):
    for m_name, m in model.named_modules():
        if m is model or not should_preload(model, m):
            continue
        HookRegistry.remove_hook_from_module(m, HookType.EFFICIENT_MODULE_LOADER.value)

    HookRegistry.remove_hook_from_module(model, HookType.EFFICIENT_MODEL_LOADER.value)
    

@register_hook_method(FeatureType.EFFICIENT_LOADING.value)
def enable_efficient_loading(model: 'ModelMixin', target_shape = None):
    """
    This patches the forward pass of modules to dynamically load / unload them
    """
    # cleaning hooks from the previous runs
    remove_efficient_loading(model)
    
    model._fully_loaded = False
    full_load: List[Tuple[weakref.ref, str]] = []
    full_load, _ = split_model_for_loading(model, target_shape)

    total_modules = 0
    for m_name, m in model.named_modules():
        if m is model or not should_preload(model, m):
            continue
        
        # NOTE: both leaf and has_orphan modules are given the preload hooks, but in case of has_orphan
        # only the orphan_params are loaded and not the children (as they will have their own hooks)
        module_hook = EfficientModuleLoaderHook(
            model_ref=model,
            module_name=m_name,
            full_load_module=any(m_name == mn for _, mn in full_load),
        )
        HookRegistry.apply_hook_to_module(m, module_hook)
        
        total_modules += 1

    app_logger.debug(f"Total modules found: {total_modules}")
    app_logger.debug(f"Total full load modules found: {len(full_load)}")

    model_hook = EfficientModelLoaderHook(full_load=full_load)
    HookRegistry.apply_hook_to_module(model, model_hook)
    