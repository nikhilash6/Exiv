import torch
from torch import nn
from transformers.utils.import_utils import is_torchao_available
if is_torchao_available():
    from torchao.quantization import quantize_

from ..base import Quantizer, TorchAOConfig
from ...utils.device import is_cuda_available, CUDA_COMPUTE_CAPABILITY
from ...utils.common import split_module_key


# sample usage - https://docs.pytorch.org/ao/main/quick_start.html
class TorchAOQuantizer(Quantizer):
    def __init__(self, quantization_config: TorchAOConfig = None, **kwargs):
        if quantization_config is None: quantization_config = TorchAOConfig()
        super().__init__(quantization_config, **kwargs)
        
        if not is_torchao_available():
            raise ImportError("torchao package not installed")
        
        if not is_cuda_available:
            raise RuntimeError("CUDA not found")
        
        # https://huggingface.co/docs/diffusers/quantization/torchao
        # compute capability 89 and above is needed
        major_ver, minor_ver = CUDA_COMPUTE_CAPABILITY
        if (major_ver == minor_ver == None) or \
            (major_ver * 10 + minor_ver < 89):
            raise RuntimeError("Cuda compute capability should atleast be 89")
    
    def is_quant_supported_val(self, model, key):
        for skip_key in self.quantization_config.skip_modules:
            if (skip_key + "." in key) or (skip_key == key):
                return False
            
        # supports quant of linear layer weights
        module, tensor_name = split_module_key(model, key)
        return isinstance(module, nn.Linear) and tensor_name == "weight"
    
    def quantize_val(self, model, key, value, device):
        module, tensor_name = split_module_key(model, key)
        module._parameters[tensor_name] = torch.nn.Parameter(value).to(device=device)
        quantize_(module, self.quantization_config.get_config_cls())
    
    def pre_process(self, model, **kwargs):
        model = super().pre_process(model, **kwargs)
        return model    # no-op
    
    # torchao quantizes the weights in place
    def post_process(self, model, **kwargs):
        model = super().post_process(model, **kwargs)
        return model    # no-op