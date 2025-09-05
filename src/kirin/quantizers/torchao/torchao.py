from transformers import is_torch_available
from ..base import Quantizer, TorchAOConfig


# NOTE: crude implementation, will polish later
# good place to start - https://docs.pytorch.org/ao/main/quick_start.html
class TorchAOQuantizer(Quantizer):
    def __init__(self, quantization_config: TorchAOConfig, **kwargs):
        super().__init__(quantization_config, **kwargs)
        
        if not is_torch_available():
            raise ImportError("torchao package not installed")