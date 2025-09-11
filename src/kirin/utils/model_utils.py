import os
import weakref
import functools
import torch
import torch.nn as nn
import psutil
import safetensors

from .device import DEFAULT_DEVICE, MemoryManager, ProcDevice
from .file import ensure_model_available
from .logging import app_logger
from ..constants import ALWAYS_SAFE_LOAD, DISABLE_MMAP, LOW_VRAM_MODE, BYTES_IN_MB
from ..quantizers.base import Quantizer


# bypassing weight creation at model init
class ModuleMeta(type(nn.Module)):
    def __call__(cls, *args, **kwargs):
        # zero init weight load
        with torch.device("meta"):
            instance = super().__call__(*args, **kwargs)
            if isinstance(instance, ModelMixin):    # mainly for safety
                instance.patch_forward_pass()
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
    def __init__(self, device: str = None, quantizer: Quantizer = None):
        super().__init__()
        self.gpu_device = device or DEFAULT_DEVICE
        self.model_path = None
        self._patched = False
        self._fully_loaded = False
        
        self.quantizer = quantizer
        
    def patch_forward_pass(self):
        current = 0
        self.full_load = []
        
        # move module to the gpu_device
        def _load_module(module):
            if module is None: return   # m_ref can turn out to be None
            
            app_logger.debug(f"Moving {module.__class__.__name__} to {self.gpu_device}")
            if any(p.device.type == "meta" for p in module.parameters(recurse=False)):
                module.to_empty(self.gpu_device)
            else:
                module.to(device=self.gpu_device)
            
            # NOTE: torchao does inplace quantization but others may not do it
            if self.quantizer is not None:
                app_logger.debug("quant seems to be supported")
                self.quantizer.quantize(module)
                
        
        def _full_load(self, *args, **kwargs):
            if not self._patched or self._fully_loaded: return None
            # load initial full_load modules
            for m_ref in self.full_load:
                m = m_ref()
                app_logger.debug(f"Loading via full load: {m.__class__.__name__}")
                _load_module(m)
            
            self._fully_loaded = True
            return None

        app_logger.debug("******** patching modules")
        # loading layers, doing work then moving them back to cpu
        # PONDER: is this better done through register_forward_pre_hook and register_forward_hook ?
        def _modified_forward(obj, *args, **kwargs):
            app_logger.debug("Inside the modified forward path")
            og_forward, module = kwargs["og_forward"], kwargs["module"]
            del kwargs["og_forward"]
            del kwargs["module"]

            try:
                app_logger.debug(f"Loading via partial load: {m.__class__.__name__}")
                _load_module(module)
                out = og_forward(obj, *args, **kwargs)
            finally:
                app_logger.debug(f"Moving back {module.__class__.__name__} to cpu")
                module.to(ProcDevice.CPU.value)
            return out

        available_mem = MemoryManager.available_memory(self.gpu_device)
        for m in self.modules():
            if m is self or not self.is_leaf_module(m):
                continue
            
            current += self._module_size(m)
            if self._module_size(m) >= available_mem:
                raise RuntimeError("Single layer exceeds total available memory, tf")
            
            if current < available_mem - 50:  # 50 MB buffer
                self.full_load.append(weakref.ref(m))
            else:
                og_forward = m.forward
                m.forward = functools.partial(_modified_forward, og_forward=og_forward, module=m)
            self._patched = True
        
        self.register_forward_pre_hook(_full_load)
        
        
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
        model_path,                     # path or url
        device = None,                  # device to load this on
        torch_dtype = "fp32",           # 'auto' or 'torch.dtype'
        force_download=False,           # re_download models
        download_path=None,             # defaults to folder util
        force_low_vram=False,
    ):
        assert model_path is not None, "model_path is required"
        
        self.gpu_device = device or self.gpu_device
        model_path = ensure_model_available(model_path, download_path, force_download)
        self.load_state_dict(self.get_state_dict(model_path), assign=True)
    
    # code adapted from ComfyUI
    def get_state_dict(self, model_path, device=torch.device("cpu")):
        file_extension = os.path.basename(model_path).split(".")[-1]
        if file_extension in ["safetensors", "sft"]:
            try:
                # safetensor's zero copy loading (pt - pytorch)
                kwargs = {"framework": "pt"}
                # safetensors only support cpu and cuda, and doesn't take cpu as param
                if device.type == "cuda": kwargs["device"] = device
                with safetensors.safe_open(model_path, **kwargs) as f:
                    sd = {}
                    for k in f.keys():
                        tensor = f.get_tensor(k)    # loading one key at a time; low mem pressure
                        if DISABLE_MMAP:
                            # moving to device (no zero copying)
                            tensor = tensor.to(device=device, copy=True)
                        sd[k] = tensor
            except Exception as e:
                app_logger.error(str(e))
                raise e
        else:   # ckpt, pth, pt
            torch_args = {}
            # using simple flags rn, will fix later
            if not DISABLE_MMAP: torch_args["mmap"] = True
            if ALWAYS_SAFE_LOAD: torch_args["weights_only"] = True
            
            sd = torch.load(model_path, map_location=device, **torch_args)
            if "state_dict" in sd:  
                sd = sd["state_dict"]   # loading state_dict if available
            elif len(sd) == 1:          # loading the first key (if it's a dict)
                val = next(iter(sd.values()))
                sd = val if isinstance(val, dict) else sd
                
        return sd