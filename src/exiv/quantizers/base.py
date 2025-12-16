import json
import torch

from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Any, List, Dict, Optional
from functools import partial
from safetensors import safe_open

from .sdnq.sdnq import SDNQQuantizerRepack
from .sdnq_lib.common import use_torch_compile, dtype_dict, sdnq_version, \
    accepted_weight_dtypes, accepted_matmul_dtypes
from ..utils.enum import ExtendedEnum, QuantizationMethod
from ..utils.common import validate_type
from ..utils.logging import app_logger


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

# NOTE: code for this is removed as of now, no longer in use
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
        

@dataclass
class SDNQQuantizerConfig(QuantizationConfig):
    def __init__(
        self,
        weights_dtype: str = "int8",
        quantized_matmul_dtype: str = None,
        group_size: int = 0,
        svd_rank: int = 32,
        svd_steps: int = 8,
        use_svd: bool = False,
        use_grad_ckpt: bool = True,
        quant_conv: bool = False,
        use_quantized_matmul: bool = False,
        use_quantized_matmul_conv: bool = False,
        use_static_quantization: bool = True,
        use_stochastic_rounding: bool = False,
        dequantize_fp32: bool = False,
        non_blocking: bool = False,
        add_skip_keys: bool = True,
        quantization_device: Optional[torch.device] = None,
        return_device: Optional[torch.device] = None,
        modules_to_not_convert: Optional[List[str]] = None,
        modules_dtype_dict: Optional[Dict[str, List[str]]] = None,
        is_training: bool = False,
        **kwargs, # pylint: disable=unused-argument
    ):
        self.weights_dtype = weights_dtype
        self.quantized_matmul_dtype = quantized_matmul_dtype
        self.is_training = is_training
        if self.is_training:
            self.quant_method = QuantizationMethod.SDNQ_TRAINING
        else:
            self.quant_method = QuantizationMethod.SDNQ
        self.group_size = group_size
        self.svd_rank = svd_rank
        self.svd_steps = svd_steps
        self.use_svd = use_svd
        self.use_grad_ckpt = use_grad_ckpt
        self.quant_conv = quant_conv
        self.use_quantized_matmul = use_quantized_matmul
        self.use_quantized_matmul_conv = use_quantized_matmul_conv
        self.use_static_quantization = use_static_quantization
        self.use_stochastic_rounding = use_stochastic_rounding
        self.dequantize_fp32 = dequantize_fp32
        self.non_blocking = non_blocking
        self.add_skip_keys = add_skip_keys
        self.quantization_device = quantization_device
        self.return_device = return_device
        self.modules_to_not_convert = modules_to_not_convert
        self.modules_dtype_dict = modules_dtype_dict
        self.is_integer = dtype_dict[self.weights_dtype]["is_integer"]
        self.sdnq_version = sdnq_version
        self.verify_integrity()

    def verify_integrity(self):
        r"""
        Safety checker that arguments are correct
        """
        if self.use_quantized_matmul and not use_torch_compile:
            raise RuntimeError("SDNQ Quantized MatMul requires a working Triton install.")
        if self.weights_dtype not in accepted_weight_dtypes:
            raise ValueError(f"SDNQ only support weight dtypes in {accepted_weight_dtypes} but found {self.weights_dtype}")
        if self.quantized_matmul_dtype is not None and self.quantized_matmul_dtype not in accepted_matmul_dtypes:
            raise ValueError(f"SDNQ only support quantized matmul dtypes in {accepted_matmul_dtypes} but found {self.quantized_matmul_dtype}")

        if self.modules_to_not_convert is None:
            self.modules_to_not_convert = []
        elif isinstance(self.modules_to_not_convert, str):
            self.modules_to_not_convert = [self.modules_to_not_convert]
        elif isinstance(self.modules_to_not_convert, tuple):
            self.modules_to_not_convert = list(self.modules_to_not_convert)
        elif not isinstance(self.modules_to_not_convert, list):
            raise ValueError(f"modules_to_not_convert must be a list but got {type(self.modules_to_not_convert)}")

        if self.modules_dtype_dict is None:
            self.modules_dtype_dict = {}
        elif not isinstance(self.modules_dtype_dict, dict):
            raise ValueError(f"modules_dtype_dict must be a dict but got {type(self.modules_dtype_dict)}")
        elif len(self.modules_dtype_dict.keys()) > 0:
            self.modules_dtype_dict = self.modules_dtype_dict.copy()
            for key, value in self.modules_dtype_dict.items():
                if isinstance(value, str):
                    value = [value]
                    self.modules_dtype_dict[key] = value
                elif isinstance(value, tuple):
                    value = list(value)
                    self.modules_dtype_dict[key] = value
                if not isinstance(key, str) or not isinstance(value, list):
                    raise ValueError(f"modules_dtype_dict must be a dictionary of strings and lists but got {type(key)} and {type(value)}")

        self.modules_to_not_convert = self.modules_to_not_convert.copy()
        self.modules_dtype_dict = self.modules_dtype_dict.copy()

    def to_dict(self):
        dct = self.__dict__.copy() # make serializable
        dct["quantization_device"] = str(dct["quantization_device"]) if dct["quantization_device"] is not None else None
        dct["return_device"] = str(dct["return_device"]) if dct["return_device"] is not None else None
        return dct


    @property
    def quantization_method(self):
        return QuantizationMethod.SDNQ.value

    @property
    def quantization_dtype(self):
        return self.weights_dtype

class Quantizer(ABC):
    def __init__(self, quantization_config: QuantizationConfig, **kwargs):
        self.quantization_config = quantization_config
        self.kwargs = kwargs

    def quantize(self, model, module, module_name):
        pass
    
    def process_model_before_weight_loading(self, *args, **kwargs):
        pass
    
    def create_quantized_param(self, *args, **kwargs):
        pass
    
    def is_state_dict_quantized(self, state_dict):
        pass
    
    def validate_environment(self, *args, **kwargs):
        pass

class QuantType(ExtendedEnum):
    BNB_NF4     = "bnb_nf4"
    BNB_FP4     = "bnb_fp4"
    BNB_INT8    = "bnb_int8"
    GGUF        = "gguf"
    SDNQ        = "sdnq"


def load_quant_config(file_path: str, key: str = "quant_config_json"):
    # loads quant config from safetensors metadata
    
    # NOTE: this type of config loading is specific to this library and model names 
    # contain 'ec' -> embedded config, to identify these models
    
    # only support safetensors for now
    if not file_path.endswith('.safetensors'):
        return None

    try:
        with safe_open(file_path, framework="pt") as f:
            metadata = f.metadata()
            if metadata and key in metadata:
                json_string = metadata[key]
                return json.loads(json_string)
            else:
                return None
    except Exception as e:
        app_logger.warning(f"Exception while parsing state dict config: {str(e)}")
        return None


def get_quantizer(quant_type: QuantType, quant_config: Dict | None = None) -> Quantizer:
    from .bnb.bnb import BnB4BitQuantizer, BnB8BitQuantizer
    
    quantizer = None
    if quant_type == None: return quantizer
    if quant_type in [QuantType.BNB_FP4, QuantType.BNB_INT8, QuantType.BNB_NF4]:
        quant_dict = {
            QuantType.BNB_FP4.value: (BnB4BitQuantizer, {'load_in_4bit': True}),
            QuantType.BNB_NF4.value: (BnB4BitQuantizer, {'load_in_4bit': True, 'bnb_4bit_quant_type': "nf4"}),
            QuantType.BNB_INT8.value: (BnB8BitQuantizer, {'load_in_8bit': True}),
        }
        
        quant_cls, quant_config_dummy = quant_dict[quant_type.value]
        # if original config is not provided then using makeshift config
        quant_config = quant_config or quant_config_dummy                
        quantizer = quant_cls(quantization_config=BNBQuantizerConfig(**quant_config))
        
    elif quant_type == QuantType.SDNQ:
        if quant_config is None:
            raise Exception("Model safetensors doesn't contain quant config in the metadata. Aborting operation.")
        return SDNQQuantizerRepack(quantization_config=SDNQQuantizerConfig(**quant_config))
        
    else:
        raise NotImplementedError(f"{quant_type.value} not implemented yet")
        
    return quantizer