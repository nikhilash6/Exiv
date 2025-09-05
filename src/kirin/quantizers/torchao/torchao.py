from transformers import is_torchao_available
if is_torchao_available():
    from torchao.quantization import quantize_

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
        # compute capability 89 and above is needed
        major_ver, minor_ver = CUDA_COMPUTE_CAPABILITY
        if (major_ver == minor_ver == None) or \
            (major_ver * 10 + minor_ver < 89):
            raise RuntimeError("Cuda compute capability should atleast be 89")
    
    def pre_process(self, model, **kwargs):
        return model    # no-op
    
    # torchao quantizes the weights in place
    def post_process(self, model, **kwargs):
        for name, module in model.named_modules():
            if name not in self.quantization_config.skip_modules:
                quantize_(module, self.quantization_config.get_config_cls())