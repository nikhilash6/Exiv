import torch
import torch.nn as nn

import os
import functools
import safetensors
from typing import Optional, Union

from ..utils.dev import print_memory_usage
from ..utils.dtype import cast_to
from ..utils.device import VRAM_DEVICE, MemoryManager, ProcDevice, is_same_device
from ..utils.file import ensure_model_available
from ..utils.logging import app_logger
from ..config import global_config, BYTES_IN_MB
from ..quantizers.base import QuantType, Quantizer, get_quantizer
from ..model_patching.efficient_loading_hook import enable_efficient_loading


# bypassing weight creation at model init
class ModuleMeta(type(nn.Module)):
    def __call__(cls, *args, **kwargs):
        model_dtype = kwargs.pop("dtype", torch.float32)
        quant_type = kwargs.get("quant_type", None)
        original_dtype = torch.get_default_dtype()
        
        try:
            torch.set_default_dtype(model_dtype)
            
            # zero init weight load
            with torch.device("meta"):
                instance = super().__call__(*args, **kwargs)
                quantizer: Quantizer = get_quantizer(quant_type=quant_type)
                instance.quantizer = quantizer
                if quantizer is not None:
                    quantizer.validate_environment()
                    quantizer.process_model_before_weight_loading(model=instance)
                
                if isinstance(instance, ModelMixin):    # mainly for safety
                    enable_efficient_loading(instance)  # kinda default hook
                    if not getattr(instance, 'dtype', None):
                        instance.dtype = model_dtype
                        
        finally:
            torch.set_default_dtype(original_dtype)
        
        return instance


class ModelMixin(nn.Module, metaclass=ModuleMeta):
    '''
    Adds additional feature to the base model
    
    - (TODO) telemetry / stats
    - (TODO) better / modular patch system
    - zero init loading
    - (TODO) multi gpu sharding
    - (TODO) implement low cpu mem usage feature
    - auto block swapping during low memory
    - (TODO) priority swapping
    - (TODO) cuda streams for offloading
    - (TODO) support GGUF loading
    - quantization support
    - safetensor support
    - URL download support
    '''
    def __init__(self, device: str = None, quant_type: QuantType = None, model_path: str = None, dtype = torch.float32):     # quant_type, dtype is used by the meta class
        super().__init__()
        self.gpu_device = device or VRAM_DEVICE
        self.model_path = model_path
    
    def clear_cache(self):
        # add other cleanup in future
        self.__class__.clear_cls_cache()

    @classmethod
    def clear_cls_cache(cls):
        cls._module_size.cache_clear()
        cls.is_leaf_module.cache_clear()
    
    @staticmethod
    @functools.lru_cache(maxsize=None)
    def is_leaf_module(module: nn.Module) -> bool:
        # TODO: this needs major fixing. Rn we are considering any module with
        # a parameter as the leaf module, so we don't have to load the parameters separately,
        # but this means that modules with multiple sub modules and even just one parameter
        # count as leaf (and will be significantly heavy than a leaf), thus increasing the min mem required
        if len(list(module.parameters(recurse=False))) > 0:
            return True
        
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
            app_logger.debug(f"moving the inputs to {self.gpu_device}")
            # new_args = tuple(a.to(self.gpu_device, non_blocking=False) if torch.is_tensor(a) else a for a in args)
            # new_kwargs = {k: (v.to(self.gpu_device, non_blocking=False) if torch.is_tensor(v) else v) for k, v in kwargs.items()}
            new_args = tuple(cast_to(a, device=self.gpu_device, dtype=self.dtype) if torch.is_tensor(a) else a for a in args)
            new_kwargs = {k: (cast_to(v, device=self.gpu_device, dtype=self.dtype) if torch.is_tensor(v) else v) for k, v in kwargs.items()}

            return super().__call__(*new_args, **new_kwargs)

    # code adapted from Huggingface Diffusers
    def load_model(
        self,
        model_path = None,              # model file path (override for flexibility)
        force_download=False,           # re_download models
        download_url=None,              # file url (optional)
        dtype=None                      # TODO: hardware specific dtype
    ):
        model_path = model_path or self.model_path
        assert model_path is not None, "model_path is required"
        # loading everything on the CPU, then modularly offloading to the GPU
        device = ProcDevice.CPU.value
        self.dtype = dtype or self.dtype
        
        model_path = ensure_model_available(model_path, download_url, force_download)
        print_memory_usage("State dict load started")
        
        state_dict = ModelMixin.get_state_dict(model_path)
        model_state_dict = self.state_dict()
        
        print_memory_usage("State dict loaded in the variable")
        
        for param_name, param in state_dict.items():
            if param_name not in model_state_dict: 
                app_logger.warning(f"skipping the param {param_name} as its not present in the model definition")
                continue
            
            if self.dtype is not None:
                if self.quantizer is not None:
                    pass    # not overiding dtype of quantized models
                else:
                    param = param.to(self.dtype)
            
            # For compatibility with PyTorch load_state_dict which converts state dict dtype to existing dtype in model, and which
            # uses `param.copy_(input_param)` that preserves the contiguity of the parameter in the model
            # Reference: https://github.com/pytorch/pytorch/blob/db79ceb110f6646523019a59bbd7b838f43d4a86/torch/nn/modules/module.py#L2040C29-L2040C29
            old_param = self
            splits = param_name.split(".")
            for split in splits:
                # recursively drill down: model.down_blocks[0].attentions[0].proj_in.weight
                old_param = getattr(old_param, split)
            
            # param_name might be for a buffer or something not loadable, skip it
            if not isinstance(old_param, (torch.nn.Parameter, torch.Tensor)):
                old_param = None
                
            if old_param is not None:
                if self.dtype is None:
                    param = param.to(old_param.dtype)
                    
                if old_param.is_contiguous():
                    param = param.contiguous()
            
            # bnb params are flattened.
            # gguf quants have a different shape based on the type of quantization applied
            if model_state_dict[param_name].shape != param.shape:
                if self.quantizer is not None:
                    self.quantizer.check_quantized_param_shape(param_name, model_state_dict[param_name], param)
                else:
                    raise ValueError(
                        f"Cannot load {model_path} because {param_name} expected shape {model_state_dict[param_name].shape}, but got {param.shape}."
                    )
            
            # final assignment
            if self.quantizer is not None and self.quantizer.check_if_quantized_param(
                self, param, param_name, state_dict, dtype=dtype
            ):
                self.quantizer.create_quantized_param(
                    self,
                    param,
                    param_name,
                    device,
                    state_dict,
                    dtype=dtype
                )
            else:
                set_module_tensor_to_device(self, param_name, device, value=param, dtype=dtype)
                
        print_memory_usage("State dict loaded in the model dict")

    # code adapted from ComfyUI
    @staticmethod
    def get_state_dict(model_path, device=torch.device("cpu")):
        if isinstance(device, str):
            device = torch.device(device)
        
        file_extension = os.path.basename(model_path).split(".")[-1]
        if file_extension in ["safetensors", "sft"]:
            try:
                # safetensor's zero copy loading (pt - pytorch)
                kwargs = {"framework": "pt"}
                # safetensors only support cpu and cuda, and doesn't take cpu as param
                if device.type == "cuda": kwargs["device"] = device.type
                with safetensors.safe_open(model_path, **kwargs) as f:
                    sd = {}
                    for k in f.keys():
                        tensor = f.get_tensor(k)    # loading one key at a time; low mem pressure
                        if global_config.disable_mmap:
                            # moving to device (no zero copying)
                            tensor = tensor.to(device=device, copy=True)
                        sd[k] = tensor
            except Exception as e:
                app_logger.error(str(e))
                raise e
        else:   # ckpt, pth, pt
            torch_args = {}
            # using simple flags rn, will fix later
            if not global_config.disable_mmap: torch_args["mmap"] = True
            if global_config.always_safe_load: torch_args["weights_only"] = True
            
            sd = torch.load(model_path, map_location=device, **torch_args)
            if "state_dict" in sd:  
                sd = sd["state_dict"]   # loading state_dict if available
            elif len(sd) == 1:          # loading the first key (if it's a dict)
                val = next(iter(sd.values()))
                sd = val if isinstance(val, dict) else sd
                
        return sd

def move_model(model, device):
    # handling device movement through our custom logic
    for name, module in model.named_modules():
        move_module(
            model,
            module,
            name, 
            target_device=device,
        )

    return model

c = 0
# TODO: dtype and non_blocking params are not handled as of now
def move_module(model, module, module_name, target_device=None):
    """
    This contains the centralized logic for moving different module types 
    between devices
    """
    if module is None: return   # m_ref can turn out to be None
    
    global c
    app_logger.info(f"Loading the current module: {c}")
    c += 1
    
    target_device = target_device or model.gpu_device
    app_logger.debug(f"Moving {module_name} to {target_device}")
    
    module_class_name = module.__class__.__name__
    is_bnb_module = module_class_name in ["Linear8bitLt", "Linear4bit"]

    if any(p.device.type == "meta" for p in module.parameters(recurse=False)):
        module.to_empty(device=target_device)
    
    elif is_bnb_module:
        device_index = torch.device(target_device).index
        if device_index is None:
             device_index = torch.cuda.current_device() # Get default index if "cuda"
        
        # .cuda(device_index) / to is overridden by bnb
        module.to(target_device)
        
        # handling the movement of linear8bit
        # after the first forward, quant weights are stored in the state
        if hasattr(module, "weight") and getattr(module.weight, "CB", None) is not None:
            module.weight.CB = module.weight.CB.to(target_device)
            
        if hasattr(module, "weight") and getattr(module.weight, "SCB", None) is not None:
            module.weight.SCB = module.weight.SCB.to(target_device)
        
        if hasattr(module, "state") and getattr(module.state, "CB", None) is not None:
            module.state.CB = module.state.CB.to(target_device)
            
        if hasattr(module, "state") and getattr(module.state, "SCB", None) is not None:
            module.state.SCB = module.state.SCB.to(target_device)

        
    else:
        # standard .to() for all other regular modules
        module.to(device=target_device)

    # app_logger.debug(f"modules rn: {[m.__class__.__name__ for mn, m in model.named_modules() if m != model]}")
    # MemoryManager.clear_memory()


# lots of checks that can be skipped
def set_module_tensor_to_device(
    module: nn.Module,
    tensor_name: str,
    device: Union[int, str, torch.device],
    value: Optional[torch.Tensor] = None,
    dtype: Optional[Union[str, torch.dtype]] = None,
    non_blocking: bool = False,
):
    # traverse the nested modules using the '.' in the tensor name (e.g., 'encoder.layer.0.weight')
    if "." in tensor_name:
        splits = tensor_name.split(".")
        for split in splits[:-1]:
            new_module = getattr(module, split)
            if new_module is None:
                raise ValueError(f"{module} has no attribute {split}.")
            module = new_module
        tensor_name = splits[-1]

    # ensure the tensor name corresponds to an existing parameter or buffer
    if tensor_name not in module._parameters and tensor_name not in module._buffers:
        raise ValueError(f"{module} does not have a parameter or a buffer named {tensor_name}.")
    is_buffer = tensor_name in module._buffers
    old_value = getattr(module, tensor_name)

    if old_value.device == torch.device("meta") and device not in ["meta", torch.device("meta")] and value is None:
        raise ValueError(f"{tensor_name} is on the meta device, we need a `value` to put in on {device}.")

    param = module._parameters[tensor_name] if tensor_name in module._parameters else None
    param_cls = type(param)

    
    if value is not None:
        if dtype is None:
            # For compatibility with PyTorch load_state_dict which converts state dict dtype to existing dtype in model
            value = value.to(old_value.dtype, non_blocking=non_blocking)
        elif not str(value.dtype).startswith(("torch.uint", "torch.int", "torch.bool")):
            value = value.to(dtype, non_blocking=non_blocking)

    device_quantization = None
    with torch.no_grad():
        # temporarily set the device to 'cpu' to handle quantization correctly before the final move.
        # if it's currently not on gpu then it needs to processed first before moving to gpu (if its the target)
        if (
            param is not None
            and param.device.type not in ("cuda", "xpu")
            and torch.device(device).type in ("cuda", "xpu")
            and param_cls.__name__ in ["Int8Params", "FP4Params", "Params4bit"]
        ):
            device_quantization = device 
            device = "cpu"

        if isinstance(value, torch.Tensor):
            new_value = value.to(device, non_blocking=non_blocking)
        else:
            new_value = torch.tensor(value, device=device)

        # revert the target device to the original GPU
        if device_quantization is not None:
            device = device_quantization

        # --- final assignment
        # simple assignment for the buffer
        if is_buffer:
            module._buffers[tensor_name] = new_value
        
        # update if a new value was provided OR if the device actually changed.
        elif value is not None or not is_same_device(torch.device(device), module._parameters[tensor_name].device):
            param_cls = type(module._parameters[tensor_name])
            kwargs = module._parameters[tensor_name].__dict__
            
            # special handling for low-precision/quantized parameter classes (e.g., bitsandbytes)
            if param_cls.__name__ in ["Int8Params", "FP4Params", "Params4bit"]:
                # cast to fp16 for 8-bit serialization/compatibility if needed
                if param_cls.__name__ == "Int8Params" and new_value.dtype == torch.float32:
                    new_value = new_value.to(torch.float16, non_blocking=non_blocking)
                
                # quantize the weights on the GPU first then move to the CPU
                if device == "cpu" and param_cls.__name__ == "Int8Params":
                    new_value = param_cls(new_value, requires_grad=old_value.requires_grad, **kwargs).to(0).to("cpu")
                    new_value.CB = new_value.CB.to("cpu")
                    new_value.SCB = new_value.SCB.to("cpu")
                else:
                    # re-wrap the tensor using its specialized class and move to the final device
                    new_value = param_cls(new_value, requires_grad=old_value.requires_grad, **kwargs).to(
                        device, non_blocking=non_blocking
                    )

            # other known quantized tensor types (affine one is from torchao)
            elif param_cls.__name__ in ["QTensor", "QBitsTensor", "AffineQuantizedTensor"]:
                new_value = torch.nn.Parameter(new_value, requires_grad=old_value.requires_grad).to(
                    device, non_blocking=non_blocking
                )
            
            # default
            else:
                new_value = param_cls(new_value, requires_grad=old_value.requires_grad).to(
                    device, non_blocking=non_blocking
                )

            module._parameters[tensor_name] = new_value

    # freeing old_value (safety check)
    # MemoryManager.clear_memory()

