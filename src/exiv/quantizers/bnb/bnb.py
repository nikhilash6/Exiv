import torch
from torch import nn
from transformers import is_bitsandbytes_available

from typing import Dict, List, Any, Optional

from ..base import Quantizer
from ...utils.common import get_module_from_name
from ...utils.device import OFFLOAD_DEVICE, VRAM_DEVICE, is_cuda_available, is_xpu_available

# this code has been adapted from Huggingface Diffusers

class BnB4BitQuantizer(Quantizer):
    """
    4-bit quantization from bitsandbytes.py quantization method:
        before loading: converts transformer layers into Linear4bit during loading: load 16bit weight and pass to the
        layer object after: quantizes individual weights in Linear4bit into 4bit at the first .cuda() call saving:
            from state dict, as usual; saves weights and `quant_state` components
        loading:
            need to locate `quant_state` components and pass to Param4bit constructor
    """
    
    def __init__(self, quantization_config, **kwargs):
        super().__init__(quantization_config, **kwargs)

        if self.quantization_config.llm_int8_skip_modules is not None:
            self.modules_to_not_convert = self.quantization_config.llm_int8_skip_modules

    def validate_environment(self, *args, **kwargs):
        if not (is_cuda_available or is_xpu_available):
            raise RuntimeError("No GPU found. A GPU is needed for quantization.")
  
        if not is_bitsandbytes_available():
            raise ImportError(
                "Using `bitsandbytes` 4-bit quantization requires the latest version of bitsandbytes: `pip install -U bitsandbytes`"
            )
            
    def is_state_dict_quantized(self, state_dict: Dict[str, Any]) -> bool:
        # soft check
        if hasattr(self, "is_prequantized"): return self.is_prequantized
        for key in state_dict.keys():
            # 4-bit keys from bnb_quantizer.py
            if "bitsandbytes__fp4" in key or "bitsandbytes__nf4" in key:
                self.is_prequantized = True
                return self.is_prequantized
                
        self.is_prequantized = False
        return self.is_prequantized

    def create_quantized_param(
        self,
        model: "ModelMixin",
        param_value: "torch.Tensor",
        param_name: str,
        target_device: "torch.device",
        state_dict: Dict[str, Any],
        unexpected_keys: Optional[List[str]] = None,
        **kwargs,
    ):
        import bitsandbytes as bnb

        module, tensor_name = get_module_from_name(model, param_name)

        if tensor_name not in module._parameters:
            raise ValueError(f"{module} does not have a parameter or a buffer named {tensor_name}.")

        old_value = getattr(module, tensor_name)

        # biases are not quantized, setting bias and returning
        if tensor_name == "bias":
            if param_value is None:
                new_value = old_value.to(target_device)
            else:
                new_value = param_value.to(target_device)

            new_value = torch.nn.Parameter(new_value, requires_grad=old_value.requires_grad)
            module._parameters[tensor_name] = new_value
            return

        if not isinstance(module._parameters[tensor_name], bnb.nn.Params4bit):
            raise ValueError("this function only loads `Linear4bit components`")
        if (
            old_value.device == torch.device("meta")
            and target_device not in ["meta", torch.device("meta")]
            and param_value is None
        ):
            raise ValueError(f"{tensor_name} is on the meta device, we need a `value` to put in on {target_device}.")

        pre_quantized_weights = self.is_state_dict_quantized(state_dict)
        # construct `new_value` for the module._parameters[tensor_name]:
        if pre_quantized_weights:
            quantized_stats = {}
            for k, v in state_dict.items():
                # `startswith` to counter for edge cases where `param_name`
                # substring can be present in multiple places in the `state_dict`
                if param_name + "." in k and k.startswith(param_name):
                    quantized_stats[k] = v
                    if unexpected_keys is not None and k in unexpected_keys:
                        unexpected_keys.remove(k)

            new_value = bnb.nn.Params4bit.from_prequantized(
                data=param_value,
                quantized_stats=quantized_stats,
                requires_grad=False,
                device=target_device,
            )
        else:
            new_value = param_value.to(OFFLOAD_DEVICE)
            kwargs = old_value.__dict__
            new_value = bnb.nn.Params4bit(new_value, requires_grad=False, **kwargs).to(target_device)

        module._parameters[tensor_name] = new_value

    def check_quantized_param_shape(self, param_name, current_param, loaded_param):
        # 4 bit weights will be packed into a 8 bit byte, thus the number of elements will be halved
        current_param_shape = current_param.shape
        loaded_param_shape = loaded_param.shape

        n = current_param_shape.numel()
        inferred_shape = (n,) if "bias" in param_name else ((n + 1) // 2, 1)    # biases are not quantized
        if loaded_param_shape != inferred_shape:
            raise ValueError(
                f"Expected the flattened shape of the current param ({param_name}) to be {loaded_param_shape} but is {inferred_shape}."
            )
        else:
            return True

    def process_model_before_weight_loading(
        self,
        model: "ModelMixin",
        keep_in_fp32_modules: List[str] = [],
        **kwargs,
    ):
        from .utils import replace_with_bnb_linear
        
        # We may keep some modules such as the `proj_out` in their original dtype for numerical stability reasons
        self.modules_to_not_convert = self.quantization_config.llm_int8_skip_modules

        if not isinstance(self.modules_to_not_convert, list):
            self.modules_to_not_convert = [self.modules_to_not_convert]

        self.modules_to_not_convert.extend(keep_in_fp32_modules)

        # Purge `None`.
        # Unlike `transformers`, we don't know if we should always keep certain modules in FP32
        # in case of diffusion transformer models. For language models and others alike, `lm_head`
        # and tied modules are usually kept in FP32.
        self.modules_to_not_convert = [module for module in self.modules_to_not_convert if module is not None]

        model = replace_with_bnb_linear(
            model, modules_to_not_convert=self.modules_to_not_convert, quantization_config=self.quantization_config
        )
        
        model.is_loaded_in_4bit = True

    def _dequantize(self, model):
        from .utils import dequantize_and_replace

        old_device = model.device
        model.to(VRAM_DEVICE)

        model = dequantize_and_replace(
            model, self.modules_to_not_convert, quantization_config=self.quantization_config
        )
        
        model.to(old_device)
        return model


class BnB8BitQuantizer(Quantizer):
    """
    8-bit quantization from bitsandbytes quantization method:
        before loading: converts transformer layers into Linear8bitLt during loading: load 16bit weight and pass to the
        layer object after: quantizes individual weights in Linear8bitLt into 8bit at fitst .cuda() call
    saving:
        from state dict, as usual; saves weights and 'SCB' component
    loading:
        need to locate SCB component and pass to the Linear8bitLt object
    """

    def __init__(self, quantization_config, **kwargs):
        super().__init__(quantization_config, **kwargs)

        if self.quantization_config.llm_int8_skip_modules is not None:
            self.modules_to_not_convert = self.quantization_config.llm_int8_skip_modules

    def validate_environment(self, *args, **kwargs):
        if not (is_cuda_available or is_xpu_available):
            raise RuntimeError("No GPU found. A GPU is needed for quantization.")

        if not is_bitsandbytes_available():
            raise ImportError(
                "Using `bitsandbytes` 8-bit quantization requires the latest version of bitsandbytes: `pip install -U bitsandbytes`"
            )

    def is_state_dict_quantized(self, state_dict: Dict[str, Any]) -> bool:
        if hasattr(self, "is_prequantized"):
            return self.is_prequantized

        for key in state_dict.keys():
            # 8-bit key
            if "bitsandbytes__int8" in key:
                self.is_prequantized = True
                return True
                
            # 8-bit metadata key
            if key.endswith(".SCB"):
                self.is_prequantized = True
                return True

        self.is_prequantized = False
        return False

    def create_quantized_param(
        self,
        model: "ModelMixin",
        param_value: "torch.Tensor",
        param_name: str,
        target_device: "torch.device",
        state_dict: Dict[str, Any],
        unexpected_keys: Optional[List[str]] = None,
        **kwargs,
    ):
        import bitsandbytes as bnb

        fp16_statistics_key = param_name.replace("weight", "SCB")
        fp16_weights_format_key = param_name.replace("weight", "weight_format")

        fp16_statistics = state_dict.get(fp16_statistics_key, None)
        fp16_weights_format = state_dict.get(fp16_weights_format_key, None)

        module, tensor_name = get_module_from_name(model, param_name)
        if tensor_name not in module._parameters:
            raise ValueError(f"{module} does not have a parameter or a buffer named {tensor_name}.")

        old_value = getattr(module, tensor_name)

        if not isinstance(module._parameters[tensor_name], bnb.nn.Int8Params):
            raise ValueError(f"Parameter `{tensor_name}` should only be a `bnb.nn.Int8Params` instance.")
        if (
            old_value.device == torch.device("meta")
            and target_device not in ["meta", torch.device("meta")]
            and param_value is None
        ):
            raise ValueError(f"{tensor_name} is on the meta device, we need a `value` to put in on {target_device}.")

        pre_quantized_weights = self.is_state_dict_quantized(state_dict)
        new_value = param_value.to(OFFLOAD_DEVICE)

        kwargs = old_value.__dict__
        new_value = bnb.nn.Int8Params(new_value, requires_grad=False, **kwargs).to(target_device)

        module._parameters[tensor_name] = new_value
        if fp16_statistics is not None:
            setattr(module.weight, "SCB", fp16_statistics.to(target_device))
            if unexpected_keys is not None:
                unexpected_keys.remove(fp16_statistics_key)

        # We just need to pop the `weight_format` keys from the state dict to remove unneeded
        # messages. The correct format is correctly retrieved during the first forward pass.
        if fp16_weights_format is not None and unexpected_keys is not None:
            unexpected_keys.remove(fp16_weights_format_key)

    def process_model_before_weight_loading(
        self,
        model: "ModelMixin",
        keep_in_fp32_modules: List[str] = [],
        **kwargs,
    ):
        from .utils import replace_with_bnb_linear

        # We may keep some modules such as the `proj_out` in their original dtype for numerical stability reasons
        self.modules_to_not_convert = self.quantization_config.llm_int8_skip_modules

        if not isinstance(self.modules_to_not_convert, list):
            self.modules_to_not_convert = [self.modules_to_not_convert]

        self.modules_to_not_convert.extend(keep_in_fp32_modules)

        # Purge `None`.
        # Unlike `transformers`, we don't know if we should always keep certain modules in FP32
        # in case of diffusion transformer models. For language models and others alike, `lm_head`
        # and tied modules are usually kept in FP32.
        self.modules_to_not_convert = [module for module in self.modules_to_not_convert if module is not None]

        model = replace_with_bnb_linear(
            model, modules_to_not_convert=self.modules_to_not_convert, quantization_config=self.quantization_config
        )
        
        model.is_loaded_in_8bit = True

    def _dequantize(self, model):
        from .utils import dequantize_and_replace

        model = dequantize_and_replace(
            model, self.modules_to_not_convert, quantization_config=self.quantization_config
        )
        return model
