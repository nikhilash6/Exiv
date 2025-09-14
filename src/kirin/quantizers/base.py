import torch

from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Any, List, Dict
from functools import partial

from ..utils.enum import QuantizationMethod
from ..utils.common import validate_type


class QuantizationConfig(ABC):
    def __init__(self, *args, **kwargs):
        pass
    
    def __post_init__(self):
        for field_name, field_def in self.__dataclass_fields__.items():
            value = getattr(self, field_name)
            expected_type = field_def.type
            validate_type(value, expected_type, field_name)
            
    @property
    @abstractmethod
    def quantization_method(self):
        pass

    @property
    @abstractmethod
    def quantization_dtype(self):
        pass

@dataclass
class TorchAOConfig(QuantizationConfig):
    kwargs: Dict[str, Any] = field(default_factory=dict)         # https://github.com/pytorch/ao/blob/main/torchao/quantization/quant_api.py
    skip_modules: List[str] = field(default_factory=list)
    quant_type: str = "fp8wo_e4m3"
    
    # not sure if this should be present here..
    def get_config_cls(self):
        from transformers.utils.import_utils import is_torchao_available
        if is_torchao_available():
            # there are couple of other methods available
            # but skipping them for now
            from torchao.quantization import (
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
            
        # NOTE: all these methods have their own set of params
        # plus can be applied at different granularity but those are
        # not added yet (i guess by default its PerTensor)
        # more info on granularity - https://github.com/pytorch/ao/blob/0f6bae5288d50251ea52f72a5e53c1ef7618a7ca/torchao/quantization/granularity.py
        config_dict = {
            # float dynamic
            "fp8dq_e4m3": Float8DynamicActivationFloat8WeightConfig,
            # (skipping for now, needs additional package)
            # "fp8dq_int4": Float8DynamicActivationInt4WeightConfig,
            # float static (TODO: skipping for now)
            # "fp8s_e4m3": Float8StaticActivationFloat8WeightConfig
            # int dynamic
            "int8dq_int4": Int8DynamicActivationInt4WeightConfig,
            "int4dq_int4": Int4DynamicActivationInt4WeightConfig,
            "int8dq_int8": Int8DynamicActivationInt8WeightConfig,
            # int only
            # (skipping, not ideal for inference on consumer cards)
            # "int4": Int4WeightOnlyConfig,
            "int8": Int8WeightOnlyConfig,
            # float only
            "fp8wo_e4m3": partial(Float8WeightOnlyConfig, weight_dtype=torch.float8_e4m3fn),  # 'fn': finite numbers (no NaN, Inf..)
            "fp8wo_e5m2": partial(Float8WeightOnlyConfig, weight_dtype=torch.float8_e5m2),
        }
        
        quant_config = self.kwargs.get("quant_config", None)
        return config_dict[self.quant_type]() if not quant_config else \
            config_dict[self.quant_type](**quant_config)

    @property
    def quantization_dtype(self):
        return self.dtype
    
    @property
    def quantization_method(self):
        return QuantizationMethod.TORCHAO.value

@dataclass
class BNBQuantizerConfig(QuantizationConfig):
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    llm_int8_has_fp16_weight: bool = False      # keeps an extra copy of fp16 weights for the 
                                                # backward pass and weight updation
                                                # more info - https://arxiv.org/pdf/2208.07339
    llm_int8_threshold: float = 6.0      # int8 handles values ~6 (99.9%), outside this
                                        # range the ops are done in fp16 (empirical data for LLMs)
    llm_int8_skip_modules: List[str] = field(default_factory=list)   # these won't be quantized
    bnb_4bit_quant_type: str = "fp4"    # supported - fp4 , nf4
    bnb_4bit_compute_dtype: Any = torch.float32     # internal computation dtype
    bnb_4bit_use_double_quant: bool = False     # weights ~ quant_val * scale + offset
                                                # this also quantizes scale, offset
                                                # which are normally in higher precision
    bnb_4bit_quant_storage: Any = torch.uint8   # naturally computers store 1 byte = 8 bits at 
                                                                    # minimum, so setting it to torch.uint4 will pack
                                                                    # stored weights tightly in a single byte
    
    @property
    def quantization_method(self):
        return QuantizationMethod.BITSANDBYTES.value
    
    @property
    def quantization_dtype(self):
        if self.load_in_8bit:
            return "llm_int8"
        elif self.load_in_4bit and self.bnb_4bit_quant_type == "fp4":
            return "fp4"
        elif self.load_in_4bit and self.bnb_4bit_quant_type == "nf4":
            return "nf4"
        else:
            return None

class Quantizer(ABC):
    def __init__(self, quantization_config: QuantizationConfig, **kwargs):
        self.quantization_config = quantization_config
        self.kwargs = kwargs
    
    # given a value (from the state dict), this determines if that
    # value/tensor can be quantized by this quantizer
    def is_quant_supported_val(self, key):
        pass
    
    def quantize(self, model, module, module_name):
        pass
    
    # runs before loading the weights
    def pre_process(self, model, **kwargs):
        return model
    
    # runs after loading the weights
    # noop for - bnb, quanto, torchao
    def post_process(self, model, **kwargs):
        return model