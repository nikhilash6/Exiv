import torch
from torch import nn
from inspect import signature
from transformers import is_bitsandbytes_available

from ...utils.model_utils import ModelMixin
from ...utils.logging import app_logger
from ..base import BNBQuantizerConfig, Quantizer
from ...utils.device import is_cuda_available


class BNBQuantizer(Quantizer):
    def __init__(self, quantization_config: BNBQuantizerConfig = None, **kwargs):
        if quantization_config is None: quantization_config = BNBQuantizerConfig()
        super().__init__(quantization_config, **kwargs)

        if not is_cuda_available:
            raise ImportError("bitsandbytes quant requires a CUDA machine")
        elif not is_bitsandbytes_available():
            raise ImportError("please install bitsandbytes package to use this feature")
    
    # replaces nn.Linear with bnb.nn.Linear4bit / 8bit
    def _recusive_linear_layer_replace(
        self,
        model: "ModelMixin",
        parent_key: str = "",
        quant_layers_replaced: bool = False,
    ):
        for name, module in model.named_children():
            full_key = f"{parent_key}.{name}" if parent_key else name

            # NOTE: this case is a little tricky and depends on what we want to
            # allow. do we want module level skip or layer level
            if self._is_excluded(full_key):
                continue
            
            # replaceable linear layer
            if isinstance(module, nn.Linear):
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
    
    def quantize(self, model, module, module_name):
        from bitsandbytes.nn import Int8Params, Params4bit
        import bitsandbytes as bnb
        
        if not isinstance(module, nn.Linear): return module
        
        in_features, out_features = module.in_features, module.out_features
        quantization_config = self.quantization_config  # being lazy
        
        weight_data = module.weight.data
        bias_data = module.bias.data if module.bias is not None else None
        
        if quantization_config.quantization_dtype == "llm_int8":
            if module in quantization_config.llm_int8_skip_modules:
                return module  # skip quantization
            
            quantized = bnb.nn.Linear8bitLt(
                in_features,
                out_features,
                bias=bias_data is not None,
                has_fp16_weights=quantization_config.llm_int8_has_fp16_weight,
                threshold=quantization_config.llm_int8_threshold,
            )
            
            quantized.weight = Int8Params(
                weight_data,
                has_fp16_weights=quantization_config.llm_int8_has_fp16_weight,
                requires_grad=False,
            )
        else:
            extra_kwargs = {}
            if "quant_storage" in signature(bnb.nn.Linear4bit).parameters:
                extra_kwargs["quant_storage"] = quantization_config.bnb_4bit_quant_storage

            quantized = bnb.nn.Linear4bit(
                in_features,
                out_features,
                bias=bias_data is not None,
                compute_dtype=quantization_config.bnb_4bit_compute_dtype,
                compress_statistics=quantization_config.bnb_4bit_use_double_quant,
                quant_type=quantization_config.bnb_4bit_quant_type,
                **extra_kwargs,
            )
            
            quantized.weight = Params4bit(
                weight_data,
                requires_grad=False,
                quant_type=quantization_config.bnb_4bit_quant_type,
            ).to(module.weight.device)
        
        if bias_data is not None:
            quantized.bias = module.bias
            
        if module.bias is not None:
            del module.bias
        del module.weight

        app_logger.debug(f'quantized dtype: {quantized.__class__.__name__} {quantized.weight.dtype}')
        
        *parent_path, attr_name = module_name.split('.')
        parent = model
        for part in parent_path:
            parent = getattr(parent, part)
        setattr(parent, attr_name, quantized)

        del weight_data, bias_data
        torch.cuda.empty_cache()


    def pre_process(
        self,
        model: "ModelMixin",
        **kwargs
    ):
        # model, quant_layers_replaced = self._recusive_linear_layer_replace(model)
        # if not quant_layers_replaced:
        #     app_logger.warning("BnB quantization not applied, no linear layers found")
        
        return model
        
    def post_process(
        self,
        model: "ModelMixin",
        **kwargs
    ):
        return model    # noop at this point, will update later