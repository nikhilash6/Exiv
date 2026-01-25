import torch
import torch.nn as nn
import torch.nn.functional as F

from ...utils.device import OFFLOAD_DEVICE

class FP8ScaledLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), device=device, dtype=torch.float8_e4m3fn),
            requires_grad=False
        )
        self.scale_weight = nn.Parameter(
            torch.ones(1, device=device, dtype=torch.float32),
            requires_grad=False
        )
        self.bias = nn.Parameter(
            torch.empty(out_features, device=device, dtype=dtype or torch.float16),
            requires_grad=False
        ) if bias else None

    def forward(self, input):
        # TODO: enable native FP8 support
        target_dtype = input.dtype
        weight_casted = self.weight.to(target_dtype)        # bit-cast expansion, relatively cheap
        scale = self.scale_weight.to(target_dtype)
        # Optimization: apply scale to the smaller tensor
        if weight_casted.numel() < input.numel():
            weight_scaled = weight_casted * scale
            return F.linear(input, weight_scaled, self.bias)
        else:
            input_scaled = input * scale
            return F.linear(input_scaled, weight_casted, self.bias)

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