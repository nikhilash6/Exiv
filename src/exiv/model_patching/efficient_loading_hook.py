import torch
from torch import nn
import numpy as np

import weakref
from typing import List, Any, Tuple

from .hook_registry import HookRegistry, HookType, ModelHook
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
        self.full_load = full_load
    
    # module here is the main model
    def pre_forward(self, module, *args, **kwargs):
        app_logger.info(f"*****##### Loading {module.__class__.__name__}")
        app_logger.debug(f"full load modules count: {len(self.full_load)}")
        self._full_load(module)
        return args, kwargs
    
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
        self.model_ref = weakref.ref(model_ref) 
        self.module_name = module_name
        self.full_load_module = full_load_module

    def pre_forward(self, module: torch.nn.Module, *args, **kwargs):
        model = self.model_ref()
        if model is None:
            return args, kwargs

        if not self.full_load_module:
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
            
        return args, kwargs

    def post_forward(self, module: torch.nn.Module, output: Any):
        model = self.model_ref()
        if model is None: 
            return output
        
        # unpatch lora
        if getattr(self, "_applied_delta", None) is not None:
            app_logger.debug("---- UNpatching lora weights delta")
            module.weight.data.sub_(self._applied_delta)
            self._applied_delta = None
        
        if not self.full_load_module:
            app_logger.debug(f"Moving back {self.module_name} to cpu via hook")
            move_module_or_params(model, module, target_device=OFFLOAD_DEVICE, module_name=self.module_name)
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

def estimate_max_activation_size(model, target_shape = None):
    """
    Estimates the peak activation memory in MBs for a given input shape.
    target_shape: (B, C, T, H, W)
    """
    if not target_shape: return 0
    
    max_activation_bytes = 0
    # NOTE: Standardize Input to 5D: (Batch, Channels, Time, Height, Width)
    # will need more fixing / patching as more models are added
    # ------------------------------------------------------------------
    current_shape = list(target_shape)
    
    # image (B, C, H, W) -> insert time=1 -> (B, C, 1, H, W)
    if len(current_shape) == 4:
        current_shape.insert(2, 1)
    
    # text (B, L, D) or other 3D -> pad with 1s -> (B, L, D, 1, 1)
    # (preventing IndexError in Conv logic, though Conv is rare here)
    while len(current_shape) < 5:
        current_shape.append(1)
        
    current_feat_shape = np.array(current_shape)
    # ------------------------------------------------------------------
    
    model_dtype = getattr(model, "dtype", torch.float16)
    
    try:
        # good way to get bytes per element (e.g., float32 -> 4, float16 -> 2)
        dtype_size = torch.tensor([], dtype=model_dtype).element_size()
    except Exception:
        dtype_size = 2

    for name, m in model.named_modules():
        if not model.is_leaf_module(m):
            continue

        output_elements = 0
        
        # estimate output size
        if isinstance(m, nn.Conv3d):
            out_channels = m.out_channels
            stride = m.stride if isinstance(m.stride, tuple) else (m.stride, m.stride, m.stride)
            t_new = max(1, current_feat_shape[2] // stride[0])
            h_new = max(1, current_feat_shape[3] // stride[1])
            w_new = max(1, current_feat_shape[4] // stride[2])
            
            output_elements = current_feat_shape[0] * out_channels * t_new * h_new * w_new
            
            # update shape if this is the patch embedder
            if "patch_embedding" in name or "patch_embed" in name:
                current_feat_shape = np.array([current_feat_shape[0], out_channels, t_new, h_new, w_new])

        elif isinstance(m, nn.Linear):
            # treat volume as sequence length for Linear layers
            total_tokens = np.prod(current_feat_shape) // current_feat_shape[1]
            output_elements = total_tokens * m.out_features
            
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.RMSNorm)):
            output_elements = np.prod(current_feat_shape)

        act_bytes = output_elements * dtype_size
        if act_bytes > max_activation_bytes:
            max_activation_bytes = act_bytes

    return max_activation_bytes / BYTES_IN_MB

def split_model_for_loading(model: 'ModelMixin', target_shape = None):
    # this determines which modules can be fully loaded permanently on the vram
    # and which has to be dynamically loaded
    full_load_modules: List[Tuple[weakref.ref, str]] = []
    
    current_mem_used = 0
    act_mb = estimate_max_activation_size(model, target_shape)
    available_mem = MemoryManager.available_memory(model.gpu_device) - (RESERVED_MEM + act_mb)
    
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
        
    return full_load_modules

def enable_efficient_loading(model: 'ModelMixin', target_shape = None):
    """
    This patches the forward pass of modules to dynamically load / unload them
    """
    model._fully_loaded = False
    full_load: List[Tuple[weakref.ref, str]] = []
    loading_mode = getattr(model, 'force_load_mode', None) or global_config.loading_mode
    
    if loading_mode == LOADING_MODE.NO_OOM.value:
        # no full_load modules in this
        full_load = []
        
    elif loading_mode == LOADING_MODE.LOW_VRAM.value:
        # full_load modules
        full_load = split_model_for_loading(model, target_shape)

    else:
        # normal load, everything should be in full_load
        for m_name, m in model.named_modules():
            if m is model or not should_preload(model, m):
                continue
            
            full_load.append((weakref.ref(m), m_name))
    
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
    