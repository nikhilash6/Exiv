from heapq import heapify
import weakref
import functools

from ..utils.logging import app_logger
from ..utils.device import OFFLOAD_DEVICE, MemoryManager
from ..config import LOADING_MODE, global_config


RESERVED_MEM = 1024     # for activations, kv cache, temp tensors etc..

class EfficientModelLoader:
    """
    There are three modes for loading the model:
    1. NO_OOM   => This loads / unloads every single module for each step
    2. LOW_VRAM => This initially loads a set of 'full_load' modules that remain on the VRAM
                    until the inference is completed. All the rest of the modules are 
                    loaded / unloaded dynamically
    3. NORMAL   => This loads the entire model in one go and keeps it in VRAM for the entire inference
    """
    
    # move module to the gpu_device
    @classmethod 
    def _load_module(cls, model, module, module_name, quant_enabled=False):
        if module is None: return   # m_ref can turn out to be None
        
        app_logger.debug(f"Moving {module.__class__.__name__} to {model.gpu_device}")
        if any(p.device.type == "meta" for p in module.parameters(recurse=False)):
            module.to_empty(device=model.gpu_device)
        else:
            module.to(device=model.gpu_device)

        # - partial loads are not quantized because many quantizers 
        #   don't support offloading / swapping
        if model.quantizer is not None and quant_enabled:
            app_logger.debug(f"quant seems to be supported {module_name}")
            model.quantizer.quantize(model=model, module=module, module_name=module_name)

        app_logger.debug(f"modules rn: {[m.__class__.__name__ for mn, m in model.named_modules() if m != model]}")
        
        MemoryManager.clear_memory()

    # loading layers, doing work then moving them back to cpu
    @classmethod
    def _modified_forward(cls, input, *args, **kwargs):
        app_logger.debug("Inside the modified forward path")
        og_forward, module, module_name, model = kwargs["og_forward"], kwargs["module"], kwargs["module_name"], kwargs["model"]
        del kwargs["og_forward"]
        del kwargs["module"]
        del kwargs["model"]
        del kwargs["module_name"]

        try:
            app_logger.debug(f"Loading via partial load: {module.__class__.__name__}")
            cls._load_module(model=model, module=module, module_name=module_name)
            out = og_forward(input, *args, **kwargs)
        finally:
            # full loaded modules are not offloaded until the entire inference is complete
            if module_name not in model.full_load:
                app_logger.debug(f"Moving back {module.__class__.__name__} to cpu")
                module.to(OFFLOAD_DEVICE)
        return out
    
    @staticmethod
    def split_model_for_loading(model: 'ModelMixin'):
        # this determines which modules can be fully loaded permanently on the vram
        # and which has to be dynamically loaded
        full_load_modules = []
        
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
                full_load_modules.append(m_name)
            else:
                break
            
        return full_load_modules
            
    @classmethod
    def patch_forward_pass(cls, model: 'ModelMixin'):
        """
        This patches the forward pass of dynamically loaded modules, full_load modules
        run with the normal forward method.
        """
        model.full_load = []
        
        loading_mode = global_config.loading_mode
        
        if loading_mode == LOADING_MODE.NO_OOM.value:
            # no full_load modules in this
            model.full_load = []
            for m_name, m in model.named_modules():
                og_forward = m.forward
                m.forward = functools.partial(
                                cls._modified_forward, 
                                og_forward=og_forward, 
                                model=model,
                                module=m, 
                                module_name=m_name
                            )
            
        elif loading_mode == LOADING_MODE.LOW_VRAM.value:
            # full_load modules
            full_load_modules = EfficientModelLoader.split_model_for_loading(model)
            for m_name, m in model.named_modules():
                if m is model or not model.is_leaf_module(m):
                    continue
                
                if m_name in full_load_modules:
                    model.full_load.append((weakref.ref(m), m_name))
                else:
                    og_forward = m.forward
                    m.forward = functools.partial(
                                    cls._modified_forward, 
                                    og_forward=og_forward, 
                                    model=model,
                                    module=m, 
                                    module_name=m_name
                                )

        else:
            # normal load, everything should be in full_load
            for m_name, m in model.named_modules():
                if m is model or not model.is_leaf_module(m):
                    continue
                
                model.full_load.append((weakref.ref(m), m_name))


        def _full_load(model, *args, **kwargs):
            if not model._patched or model._fully_loaded: return None
            # load initial full_load modules
            for m_ref, m_name in model.full_load:
                m = m_ref()
                app_logger.debug(f"Loading via full load: {m_name}")
                cls._load_module(model=model, module=m, module_name=m_name, quant_enabled=True)
            
            model._fully_loaded = True
            return None
        
        app_logger.debug(f"full load modules: {[m_name for _, m_name in model.full_load]}")
        model._patched = True
        model.register_forward_pre_hook(_full_load)
        