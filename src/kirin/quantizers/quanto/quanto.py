from torch import nn
from transformers import is_optimum_quanto_available

from ..base import QuantoConfig, Quantizer
from ...utils.model_utils import ModelMixin
from ...utils.logging import app_logger

class QuantoQuantizer(Quantizer):
    def __init__(self, quantization_config: QuantoConfig, **kwargs):
        super().__init__(quantization_config, **kwargs)
        
        if not is_optimum_quanto_available():
            raise ImportError("Quanto package not found")

    # code adapted from HF diffusers
    def _replace_with_quanto_layers(self, model, pre_quantized=False):
        from optimum.quanto import QLinear, freeze, qfloat8, qint2, qint4, qint8

        def _get_weight_type(dtype: str):
            return {"float8": qfloat8, "int8": qint8, "int4": qint4, "int2": qint2}[dtype]

        def _replace_layers(model, quantization_config, skip_modules, quant_layers_replaced):
            has_children = list(model.children())
            if not has_children:
                return model

            for name, module in model.named_children():
                _, quant_layers_replaced = _replace_layers(module, quantization_config, skip_modules)

                if name in skip_modules:
                    continue

                if isinstance(module, nn.Linear):
                    qlinear = QLinear(
                        in_features=module.in_features,
                        out_features=module.out_features,
                        bias=module.bias is not None,
                        dtype=module.weight.dtype,
                        weights=_get_weight_type(quantization_config.dtype),
                    )
                    model._modules[name] = qlinear
                    model._modules[name].source_cls = type(module)
                    model._modules[name].requires_grad_(False)
                    quant_layers_replaced = True

            return model, quant_layers_replaced

        model, quant_layers_replaced = _replace_layers(
                                            model, 
                                            self.quantization_config, 
                                            self.skip_modules, 
                                            quant_layers_replaced=False
                                        ) 

        if not quant_layers_replaced:
             app_logger.warning("Optimum quanto quantization not applied, no linear layers found")

        '''
        - training needs fp32 but inference can run with quantized weights
        - replaces normal tensor with qtensor, so as to load already quantized weights
        def freeze(self):
        qweight = self.qweight
        if qweight is not None:
            # Replace float weights by quantized weights
            self.weight = torch.nn.Parameter(qweight)
        '''
        if pre_quantized:
            freeze(model)

        return model
    
    def pre_process(
        self, 
        model: "ModelMixin", 
        **kwargs
    ):
        model, quant_layers_replaced = self._recusive_linear_layer_replace(model)
        if not quant_layers_replaced:
            app_logger.warning("BnB quantization not applied, no linear layers found")
            
    def post_process(self, model, **kwargs):
        return model    # noop