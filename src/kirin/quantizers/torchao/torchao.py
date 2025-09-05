from transformers import is_torchao_available
if is_torchao_available():
    # there are couple of other methods available
    # but skipping them for now
    from torchao.quantization import (
        quantize_,
        Float8DynamicActivationFloat8WeightConfig,
        Float8DynamicActivationInt4WeightConfig,
        Float8StaticActivationFloat8WeightConfig,
        Int8DynamicActivationInt4WeightConfig,
        Int4DynamicActivationInt4WeightConfig,
        Int4WeightOnlyConfig,
        Int8WeightOnlyConfig,
        Int8DynamicActivationInt8WeightConfig,
        Float8WeightOnlyConfig,
    )

from ...constants import CUDA_AVAILABLE, CUDA_COMPUTE_CAPABILITY
from ..base import Quantizer, TorchAOConfig


# NOTE: crude implementation, will polish later
# sample usage - https://docs.pytorch.org/ao/main/quick_start.html
class TorchAOQuantizer(Quantizer):
    def __init__(self, quantization_config: TorchAOConfig, **kwargs):
        super().__init__(quantization_config, **kwargs)
        
        if not is_torchao_available():
            raise ImportError("torchao package not installed")
        
        if not CUDA_AVAILABLE:
            raise RuntimeError("CUDA not found")
        
        # https://huggingface.co/docs/diffusers/quantization/torchao
        if CUDA_COMPUTE_CAPABILITY is None or \
            (CUDA_COMPUTE_CAPABILITY[0] * 10 + CUDA_COMPUTE_CAPABILITY[1] < 89):
            raise RuntimeError("Cuda compute capability should atleast be 89")

    @property
    def _config_cls(self):
        config_dict = {
            Float8DynamicActivationFloat8WeightConfig : "",     # TODO: complete this
            Float8DynamicActivationInt4WeightConfig : "",
            Float8StaticActivationFloat8WeightConfig : "",
            Int8DynamicActivationInt4WeightConfig : "",
            Int4DynamicActivationInt4WeightConfig : "",
            Int4WeightOnlyConfig : "",
            Int8WeightOnlyConfig : "",
            Int8DynamicActivationInt8WeightConfig : "",
            Float8WeightOnlyConfig : "",
        }
        
        quant_config = self.kwargs.get("quant_config", None)
        return config_dict[self.quantization_config.quant_type]() if not quant_config else \
            config_dict[self.quantization_config.quant_type](**quant_config)
    
    def pre_process(self, model, **kwargs):
        return model    # no-op
    
    # torchao quantizes the weights in place
    def post_process(self, model, **kwargs):
        for name, module in model.named_modules():
            if name not in self.quantization_config.skip_modules:
                quantize_(module, self._config_cls)
    
    