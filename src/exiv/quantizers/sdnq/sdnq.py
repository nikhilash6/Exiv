from typing import Dict, Any

from ..sdnq_lib.quantizer import SDNQQuantizer


class SDNQQuantizerRepack(SDNQQuantizer):
    def __init__(self, quantization_config, **kwargs):
        super().__init__(quantization_config, **kwargs)
        
        self.modules_to_not_convert = []
        
    def validate_environment(self, *args, **kwargs):
        return True
        
    def is_state_dict_quantized(self, state_dict: Dict[str, Any]) -> bool:
        # soft check, may need to be updated as more models are added
        if hasattr(self, "is_prequantized"):
            return self.is_prequantized

        for key in state_dict.keys():
            if key.endswith(".svd_up") or key.endswith(".svd_down"):
                self.is_prequantized = True

        self.is_prequantized = False
        return self.is_prequantized
    
    def check_if_quantized_param(self, model, param_value, param_name, *args, **kwargs):
        return super().check_if_quantized_param(model, param_value, param_name, *args, **kwargs)
    
    def create_quantized_param(self, model, param_value, param_name, target_device, *args, **kwargs):
        return super().create_quantized_param(model, param_value, param_name, target_device, *args, **kwargs)
    
    def process_model_before_weight_loading(
        self,
        model: "ModelMixin",
        **kwargs
    ):
        # TODO: not passing the keep_in_fp32_modules here
        return super()._process_model_before_weight_loading(self, model)