import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from ...utils.device import OFFLOAD_DEVICE, is_fp8_available

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
            torch.ones((), device=device, dtype=torch.float32),
            requires_grad=False
        )
        self.scale_input = nn.Parameter(
            torch.ones((), device=device, dtype=torch.float32),
            requires_grad=False
        )
        self.bias = nn.Parameter(
            torch.empty(out_features, device=device, dtype=dtype or torch.float16),
            requires_grad=False
        ) if bias else None

    def forward(self, input):
        # native fp8 tensor cores
        if is_fp8_available and input.is_cuda:
            # flatten inputs (Batch, Seq, Hidden) -> (Batch*Seq, Hidden)
            # torch._scaled_mm strictly requires 2D tensors
            input_shape = input.shape
            if input.dim() > 2:
                input = input.view(-1, self.in_features)
                
            scale_in_inv = (1.0 / self.scale_input).to(input.dtype)
            input_fp8 = (input * scale_in_inv).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
            if not input_fp8.is_contiguous(): input_fp8 = input_fp8.contiguous()
            out = torch._scaled_mm(
                input_fp8,
                self.weight.t(),
                scale_a=self.scale_input.float(),   # NOTE: make more robust 
                scale_b=self.scale_weight,
                bias=self.bias,
                out_dtype=input.dtype
            )

            if len(input_shape) > 2: out = out.view(*input_shape[:-1], self.out_features)   # unflatten
        else:
            # older arch
            weight_casted = self.weight.to(input.dtype)        # bit-cast expansion, relatively cheap
            scale = self.scale_weight.to(input.dtype)
            bias = self.bias.to(input.dtype)
            # Optimization: apply scale to the smaller tensor
            if weight_casted.numel() < input.numel():
                out = F.linear(input, weight_casted * scale, bias)
            else:
                out = F.linear(input * scale, weight_casted, bias)

        return out

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