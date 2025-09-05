from ast import Dict
import torch

from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Any, List
from sympy import Union

from ..utils.enum import QuantizationMethod


class QuantizationConfig(ABC):
    def __init__(self, *args, **kwargs):
        pass
    
    def __post_init__(self):
        # quick data type checker
        for field in self.__dataclass_fields__:
            value = getattr(self, field)
            expected_type = self.__dataclass_fields__[field].type
            if not isinstance(value, expected_type):
                raise TypeError(f"Variable '{field}' must be of type {expected_type.__name__}, but got {type(value).__name__}")

    @property
    @abstractmethod
    def quantization_dtype(self):
        pass

@dataclass
class TorchAOConfig(QuantizationConfig):
    kwargs: Dict[str, Any]          # https://github.com/pytorch/ao/tree/main/torchao/quantization#other-available-quantization-techniques
    skip_modules: List[str] = []
    quant_type: str = "float8wo"    # https://huggingface.co/docs/diffusers/en/quantization/torchao#supported-quantization-types

    @property
    def quantization_dtype(self):
        return self.dtype
    
    @property
    def quantization_method(self):
        return QuantizationMethod.TORCHAO.value

@dataclass
class QuantoConfig(QuantizationConfig):
    quant_type: str = "int8"        # basically dtype to quant in
    skip_modules: List[str] = []    # these won't be quantized
    
    @property
    def quantization_dtype(self):
        return self.quant_type
    
    @property
    def quantization_method(self):
        return QuantizationMethod.QUANTO.value

@dataclass
class BNBQuantizerConfig(QuantizationConfig):
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    llm_int8_threshold: bool = 6.0      # int8 handles values ~6 (99.9%), outside this
                                        # range the ops are done in fp16 (empirical data for LLMs)
    llm_int8_skip_modules: List[str] = []   # these won't be quantized
    bnb_4bit_quant_type: str = "fp4"    # supported - fp4 , nf4
    bnb_4bit_compute_dtype: Union[torch.dtype, str] = torch.float32     # internal computation dtype
    bnb_4bit_use_double_quant: bool = False     # weights ~ quant_val * scale + offset
                                                # this also quantizes scale, offset
                                                # which are normally in higher precision
    bnb_4bit_quant_storage: Union[torch.dtype, str] = torch.uint8   # naturally computers store 1 byte = 8 bits at 
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
    
    # runs before loading the weights
    def pre_process(self, model, **kwargs):
        pass
    
    # runs after loading the weights
    # noop for - bnb, quanto
    def post_process(self, model, **kwargs):
        pass