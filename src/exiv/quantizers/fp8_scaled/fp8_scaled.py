import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional
from ...utils.common import get_module_from_name
from ...utils.logging import app_logger
from ..base import Quantizer
from .layer import FP8ScaledLinear

class FP8ScaledQuantizer(Quantizer):
    def validate_environment(self, *args, **kwargs):
        if not hasattr(torch, "float8_e4m3fn"):
            raise RuntimeError("PyTorch version too old. FP8 requires newer PyTorch.")

    def check_if_quantized_param(
        self,
        model,
        param_value: "torch.Tensor",
        param_name: str,
        state_dict: Dict[str, Any],
        **kwargs,
    ) -> bool:
        module, tensor_name = get_module_from_name(model, param_name)
        return isinstance(module, FP8ScaledLinear)

    def process_model_before_weight_loading(self, model, keep_in_fp32_modules: List[str] = [], **kwargs):
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                if any(k in name for k in keep_in_fp32_modules):
                    continue
                
                parent_name = name.rsplit(".", 1)[0] if "." in name else ""
                child_name = name.split(".")[-1]
                parent = model.get_submodule(parent_name) if parent_name else model

                new_layer = FP8ScaledLinear.from_linear(module)
                setattr(parent, child_name, new_layer)

    def create_quantized_param(
        self,
        model,
        param_value: torch.Tensor,
        param_name: str,
        target_device: torch.device,
        state_dict: Dict[str, Any],
        **kwargs
    ):
        module, tensor_name = get_module_from_name(model, param_name)

        if not isinstance(module, FP8ScaledLinear):
            return

        if tensor_name == "weight":
            # CASE A: fp16/bf16 on the fly conversion (makes little sense tbh)
            if param_value.dtype in [torch.float16, torch.float32, torch.bfloat16]:
                # scale
                max_val = param_value.abs().max()
                scale = max_val / 448.0
                scale = torch.max(scale, torch.tensor(1e-6, device=scale.device))
                
                # quantize
                weight_scaled = param_value / scale
                weight_fp8 = weight_scaled.clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
                
                # set weights
                module.weight = torch.nn.Parameter(
                    weight_fp8.to(target_device), 
                    requires_grad=False
                )
                module.scale_weight = torch.nn.Parameter(
                    scale.view(1).to(torch.float32).to(target_device),
                    requires_grad=False
                )
            
            # CASE B: loading pre-quant fp8_e4m3fn
            elif param_value.dtype == torch.float8_e4m3fn:
                module.weight = torch.nn.Parameter(
                    param_value.to(target_device), 
                    requires_grad=False
                )
                scale_key = param_name.replace(".weight", ".scale_weight")
                if scale_key not in state_dict:
                    app_logger.warning(f"Warning: Loaded FP8 weight for {param_name} but found no corresponding scale_weight! Defaulting to 1.0 (Output will likely be wrong).")
                    module.scale_weight.data.fill_(1.0)
                else:
                    scale_val = state_dict[scale_key]
                    module.scale_weight = torch.nn.Parameter(
                        scale_val.to(target_device),
                        requires_grad=False
                    )
            
            # CASE C: loading pre-quant fp8_e5m2 
            elif param_value.dtype == torch.float8_e5m2:
                app_logger.debug(f"Converting {param_name} to E5M2 in a hacky way")
                scale_key = param_name.replace(".weight", ".scale_weight")
                old_scale = state_dict[scale_key].to("cpu", dtype=torch.float32) if scale_key in state_dict else 1.0
                w_e4m3, s_fp32 = convert_e5m2_to_e4m3(param_value.to("cpu"), old_scale)
                module.weight = torch.nn.Parameter(
                    w_e4m3.to(target_device), 
                    requires_grad=False
                )
                module.scale_weight = torch.nn.Parameter(
                    s_fp32.to(target_device),
                    requires_grad=False
                )
            
        elif tensor_name == "scale_weight":
            module.scale_weight = torch.nn.Parameter(
                 param_value.to(device=target_device, dtype=module.scale_weight.dtype),
                 requires_grad=False
             )

        elif tensor_name == "bias":
            if param_value is not None:
                module.bias = torch.nn.Parameter(
                    param_value.to(device=target_device, dtype=module.bias.dtype),
                    requires_grad=False
                )
                
                
def convert_e5m2_to_e4m3(t_e5m2, old_scale_fp32):
    weight_fp32 = t_e5m2.to(torch.float32)
    real_weights = weight_fp32 * old_scale_fp32
    
    # new_scale for E4M3
    max_val = real_weights.abs().max()
    new_scale = max_val / 448.0
    new_scale = torch.max(new_scale, torch.tensor(1e-6, device=new_scale.device))
    
    # main e4m3 conversion
    weights_scaled = real_weights / new_scale
    
    # clamp and cast
    weights_clamped = weights_scaled.clamp(-448.0, 448.0)
    weight_e4m3 = weights_clamped.to(torch.float8_e4m3fn)
    return weight_e4m3, new_scale
    