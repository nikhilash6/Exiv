import os
import weakref
import functools
import torch
import torch.nn as nn
import psutil
import safetensors

from .memory import mem_manager
from .file import ensure_model_available
from .logging import app_logger
from ..constants import DISABLE_MMAP, LOW_VRAM_MODE


# bypassing weight creation at model init
class ModuleMeta(type):
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
        
    def patch_forward_pass(self):
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
            if current < mem_manager.available_memory(self.device) - 50_000_000:  # 50 MB buffer
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
    
    def load_model(
        self,
        model_path,                     # path or url
        device: torch.device,           # device to load this on
        torch_dtype,                    # 'auto' or 'torch.dtype'
        force_download=False,           # re_download models
        download_path=None,             # defaults to folder util
    ):
        assert model_path is not None, "model_path is required"
        model_path = ensure_model_available(model_path, download_path, force_download)
        self.load_state_dict(self.get_state_dict(model_path, device=device), assign=True)
    
    # code adapted from ComfyUI
    def get_state_dict(model_path, device):
        # loading to cpu, then loading it on demand on gpu
        if LOW_VRAM_MODE: device = torch.device("cpu")
        # torch.load(model_path, map_location=device)
        file_extension = os.path.basename(model_path).split(".")[-1]
        if file_extension in ["safetensors", "sft"]:
            try:
                # safetensor's zero copy loading (pt - pytorch)
                with safetensors.safe_open(model_path, framework="pt", device=device) as f:
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
        else:
            torch_args = {}
            # using the same mmap flag here, will see how it pans out
            if not DISABLE_MMAP: torch_args["mmap"] = True

            ...