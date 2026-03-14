import torch
import torch.nn as nn
from abc import abstractmethod

from .helper_methods import get_state_dict, set_module_tensor_to_device
from ..utils.common import get_module_from_name
from ..utils.dtype import cast_to
from ..utils.device import VRAM_DEVICE, ProcDevice
from ..utils.logging import app_logger
from ..quantizers.base import QuantType, Quantizer, get_quantizer
from ..model_patching.hook_registry import HookLocation
from ..utils.file import ensure_model_availability

# TODO: keeping separate from the diffusion meta, maybe these will be merged in the future
class ARModuleMeta(type(nn.Module)):
    def __call__(cls, *args, **kwargs):
        model_dtype = kwargs.pop("dtype", torch.float32)
        quant_type = kwargs.get("quant_type", None)
        quant_config = kwargs.get("quant_config", None)
        force_load_mode = kwargs.get("force_load_mode", None)
        original_dtype = torch.get_default_dtype()
        
        try:
            torch.set_default_dtype(model_dtype)
            
            # zero init weight load
            with torch.device("meta"):
                instance = super().__call__(*args, **kwargs)
                quantizer: Quantizer = get_quantizer(quant_type=quant_type, quant_config=quant_config)
                instance.quantizer = quantizer
                if isinstance(instance, ARModelMixin):    # mainly for safety
                    instance.force_load_mode = getattr(instance, "force_load_mode", None) or force_load_mode
                    if not getattr(instance, 'dtype', None):
                        instance.dtype = model_dtype
                        
        finally:
            torch.set_default_dtype(original_dtype)
        
        return instance

class ARModelArchConfig:
    def __init__(self, model_type=None):
        self.model_type = model_type
        # TODO: add more specific attributes and methods

class ARModelMixin(nn.Module):
    """
    - (TODO) support quantizers
    - (TODO) support GGUF loading
    """
    
    def __init__(self, device: str = None, quant_type: QuantType = None, model_path: str = None, dtype = torch.float32, **kwargs):
        super().__init__()
        
        self.gpu_device = device or VRAM_DEVICE
        self.model_path = model_path
        self.model_arch_config = None
        
    def __call__(self, *args, **kwargs):
        """
        Must return: (logits: torch.Tensor, updated_kv_cache: any)
        - logits shape: [Batch, Sequence_Len, Vocab_Size]
        - cond: dictionary containing extra conditioning (e.g., {"speaker_cond": tensor})
        """
        def original_call(*args, **kwargs):
            with torch.inference_mode():
                app_logger.debug(f"moving the inputs to {self.gpu_device} and dtype {self.dtype}")
                new_args = tuple(cast_to(a, device=self.gpu_device, dtype=self.dtype) if torch.is_tensor(a) and idx == 0 else a for idx, a in enumerate(args))
                new_kwargs = {k: (cast_to(v, device=self.gpu_device, dtype=self.dtype) if torch.is_tensor(v) else v) for k, v in kwargs.items()}
                
                return super(ARModelMixin, self).__call__(*new_args, **new_kwargs)
            
        registry = getattr(self, "hook_registry", None)
        if registry and registry.head.next_hook != registry.tail:
            wrapped_call = registry.get_wrapped_fn(original_call, location=HookLocation.MODEL_RUN.value)
            return wrapped_call(*args, **kwargs)
        else:
            return original_call(*args, **kwargs)
        
    def load_model(
        self,
        model_path = None,
        force_download = False,
        download_url = None,
        dtype = None,
        model_type = None
    ):
        model_path = model_path or self.model_path
        assert model_path is not None, "model_path is required"
        device = ProcDevice.CPU.value
        self.dtype = dtype or self.dtype
        
        # TODO: add gguf and other quantizers support
        
        model_path = ensure_model_availability(model_path, download_url, force_download)
        state_dict = get_state_dict(model_path, model_type=model_type)
        model_state_dict = self.state_dict()
        for param_name, param in state_dict.items():
            if param_name not in model_state_dict: 
                app_logger.warning(f"Skipping param {param_name} as it's not present in the model definition")
                continue
            
            parent_module, leaf_name = get_module_from_name(self, param_name)
            param = param.to(self.dtype)    # can be removed
            old_param = getattr(parent_module, leaf_name, None)
            # skipping buffer / non-loadable entities
            if not isinstance(old_param, (torch.nn.Parameter, torch.Tensor)):
                old_param = None
                
            if old_param is not None:
                if self.dtype is None:
                    param = param.to(old_param.dtype)
                    
                if old_param.is_contiguous():
                    param = param.contiguous()
                    
            if model_state_dict[param_name].shape != param.shape:
                raise ValueError(
                        f"Cannot load {model_path} because {param_name} expected shape {model_state_dict[param_name].shape}, but got {param.shape}."
                    )

            set_module_tensor_to_device(self, param_name, device, value=param, dtype=self.dtype)
        
        del state_dict
        app_logger.info("Autoregressive Model successfully loaded to CPU.")
        
        