import torch
from torch import nn
from transformers.utils.import_utils import is_torchao_available
if is_torchao_available():
    from torchao.quantization import quantize_

from ..base import Quantizer, TorchAOConfig
from ...utils.device import is_cuda_available, CUDA_CC
from ...utils.common import split_module_key
from ...utils.logging import app_logger

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
        if (CUDA_CC < 89):
            raise RuntimeError(f"Cuda compute capability should atleast be 89, current capability {CUDA_CC}")
    
    def is_quant_supported_val(self, mod, *args):
        for skip_key in self.quantization_config.skip_modules:
            if skip_key == mod or hasattr(mod, skip_key): 
                return False

        # NOTE: the internal code of torchao has a much more comprehensive
        # set of checks, wonder if that is needed here as well
        return isinstance(mod, nn.Linear) and hasattr(mod, 'weight')
    
    def quantize(self, module):
        '''
        - many quantized layers (such as AffineQuantizedTensor) return the dtype
            of the original high precision tensor
        - 
        '''
        app_logger.debug("*** quantizing the module")
        quantize_(
            model=module, 
            config=self.quantization_config.get_config_cls(), 
            filter_fn=self.is_quant_supported_val
        )
    
    def pre_process(self, model, **kwargs):
        model = super().pre_process(model, **kwargs)
        return model    # no-op
    
    # torchao quantizes the weights in place
    def post_process(self, model, **kwargs):
        model = super().post_process(model, **kwargs)
        return model    # no-op