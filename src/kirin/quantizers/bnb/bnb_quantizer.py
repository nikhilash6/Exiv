import torch
from torch import nn
from inspect import signature
from transformers import is_bitsandbytes_available
if is_bitsandbytes_available():
    import bitsandbytes as bnb

from ...utils.model_utils import ModelMixin
from ...utils.logging import app_logger
from ..base import BNBQuantizerConfig, Quantizer


class BNBQuantizer(Quantizer):
    def __init__(self, quantization_config: BNBQuantizerConfig, **kwargs):
        super().__init__(quantization_config, **kwargs)
        
        if not is_bitsandbytes_available():
            raise ValueError("please install bitsandbytes package to use this feature")
    
    # replaces nn.Linear with bnb.nn.Linear4bit / 8bit
    def _recusive_linear_layer_replace(
        self,
        model: "ModelMixin",
        parent_key: str = "",
        quant_layers_replaced: bool = False,
    ):
        for name, module in model.named_children():
            full_key = f"{parent_key}.{name}" if parent_key else name

            # replaceable linear layer
            if isinstance(module, nn.Linear) and not self._is_excluded(full_key):
                model._modules[name] = self._make_quantized_linear(module)
                quant_layers_replaced = True

            # nested modules
            elif len(list(module.children())) > 0:
                _, child_replaced = self._recusive_linear_layer_replace(
                    module,
                    parent_key=full_key,
                    quant_layers_replaced=quant_layers_replaced,
                )
                quant_layers_replaced = quant_layers_replaced or child_replaced

        return model, quant_layers_replaced
    
    def _is_excluded(self, full_key: str) -> bool:
        exclude_list = self.quantization_config.llm_int8_skip_modules
        for key in exclude_list:
            if full_key == key or full_key.startswith(key + "."):
                return True
        return False
    
    def _make_quantized_linear(self, module: nn.Linear):
        in_features, out_features = module.in_features, module.out_features
        quantization_config = self.quantization_config  # being lazy
        if quantization_config.quantization_method() == "llm_int8":
            quantized = bnb.nn.Linear8bitLt(
                in_features,
                out_features,
                bias=module.bias is not None,
                has_fp16_weights=quantization_config.llm_int8_has_fp16_weight,
                threshold=quantization_config.llm_int8_threshold,
            )
        else:
            if module in quantization_config.llm_int8_skip_modules:
                return module  # skip quantization

            extra_kwargs = {}
            if "quant_storage" in signature(bnb.nn.Linear4bit).parameters:
                extra_kwargs["quant_storage"] = quantization_config.bnb_4bit_quant_storage

            quantized = bnb.nn.Linear4bit(
                in_features,
                out_features,
                bias=module.bias is not None,
                compute_dtype=quantization_config.bnb_4bit_compute_dtype,
                compress_statistics=quantization_config.bnb_4bit_use_double_quant,
                quant_type=quantization_config.bnb_4bit_quant_type,
                **extra_kwargs,
            )

        # preserving metadata
        quantized.source_cls = type(module)
        quantized.requires_grad_(False)
        return quantized
    
    def pre_process(
        self,
        model: "ModelMixin",
        **kwargs
    ):
        model, quant_layers_replaced = self._recusive_linear_layer_replace(model)
        if not quant_layers_replaced:
            app_logger.warning("BnB quantization not applied, no linear layers found")
        
    def post_process(
        self,
        model: "ModelMixin",
        **kwargs
    ):
        pass    # TODO: implement this