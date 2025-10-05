import weakref
import functools

from ..utils.logging import app_logger
from ..utils.device import MemoryManager, ProcDevice

# TODO: this is somewhat messy, will be refined as things progress
class ModelPatcher:
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
    # PONDER: is this better done through register_forward_pre_hook and register_forward_hook ?
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
            app_logger.debug(f"Moving back {module.__class__.__name__} to cpu")
            module.to(ProcDevice.CPU.value)
        return out
            
    @classmethod
    def patch_forward_pass(cls, model: 'ModelMixin'):
        current = 0
        model.full_load = []

        def _full_load(model, *args, **kwargs):
            if not model._patched or model._fully_loaded: return None
            # load initial full_load modules
            for m_ref, m_name in model.full_load:
                m = m_ref()
                app_logger.debug(f"Loading via full load: {m_name}")
                cls._load_module(model=model, module=m, module_name=m_name, quant_enabled=True)
            
            model._fully_loaded = True
            return None

        available_mem = MemoryManager.available_memory(model.gpu_device)
        for m_name, m in model.named_modules():
            if m is model or not model.is_leaf_module(m):
                continue
            print("layer: ", m_name)
            current += model._module_size(m)
            if model._module_size(m) >= available_mem:
                raise RuntimeError(f"Single layer mem size of {model._module_size(m)} exceeds the total available memory of {available_mem}")
            
            if current < available_mem - 50:  # 50 MB buffer
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
            model._patched = True
        
        model.register_forward_pre_hook(_full_load)
        