import torch
import torch.nn as nn
import torch.nn.functional as F

from ...utils.device import OFFLOAD_DEVICE, is_fp8_available

class FP8ScaledLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_native = is_fp8_available and (device is None or torch.device(device).type == "cuda")
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), device=device, dtype=torch.float8_e4m3fn),
            requires_grad=False
        )
        self.scale_weight = nn.Parameter(
            torch.ones((), device=device, dtype=torch.float32),
            requires_grad=False
        )
        self.bias = nn.Parameter(
            torch.empty(out_features, device=device, dtype=dtype or torch.float16),
            requires_grad=False
        ) if bias else None

    def forward(self, input):
        # native fp8 tensor cores
        if self.use_native and input.is_cuda:
            scale = (input.abs().max() / 448.0).clamp(min=1e-6)
            input_fp8 = (input / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
            out = torch._scaled_mm(input_fp8, self.weight.t(), scale, self.scale_weight)
            return out + self.bias if self.bias is not None else out
        
        # older arch
        weight_casted = self.weight.to(input.dtype)        # bit-cast expansion, relatively cheap
        scale = self.scale_weight.to(input.dtype)
        # Optimization: apply scale to the smaller tensor
        if weight_casted.numel() < input.numel():
            return F.linear(input, weight_casted * scale, self.bias)
        else:
            return F.linear(input * scale, weight_casted, self.bias)

    @classmethod
    def from_linear(cls, linear_module):
        new_layer = cls(
            linear_module.in_features,
            linear_module.out_features,
            bias=linear_module.bias is not None,
            device=linear_module.weight.device,
            dtype=linear_module.weight.dtype
        )
        return new_layer