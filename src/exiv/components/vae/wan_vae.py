from functools import partial
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor

from typing import Union, Tuple, Optional, List

from .base import DiagonalGaussianDistribution, VAEBase
from ..activations import get_activation
from ...config import LOADING_MODE


CACHE_T = 2

class CausalConv3d(nn.Conv3d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Union[int, Tuple[int, int, int]] = 1,
        padding: Union[int, Tuple[int, int, int]] = 0,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        # Set up causal padding
        self._padding = (self.padding[2], self.padding[2], self.padding[1], self.padding[1], 2 * self.padding[0], 0)
        self.padding = (0, 0, 0)

    def forward(self, x, cache_x=None):
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            cache_x = cache_x.to(x.device)
            x = torch.cat([cache_x, x], dim=2)
            padding[4] -= cache_x.shape[2]
        x = F.pad(x, padding)
        return super().forward(x)


class RMS_norm(nn.Module):
    def __init__(self, dim: int, channel_first: bool = True, images: bool = True, bias: bool = False) -> None:
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)

        self.channel_first = channel_first
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.0

    def forward(self, x):
        return F.normalize(x, dim=(1 if self.channel_first else -1)) * self.scale * self.gamma + self.bias


class Upsample(nn.Upsample):
    def forward(self, x):
        return super().forward(x.float()).type_as(x)


class Resample(nn.Module):
    def __init__(self, dim: int, mode: str) -> None:
        super().__init__()
        self.dim = dim
        self.mode = mode

        # layers
        if mode == "upsample2d":
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"), nn.Conv2d(dim, dim // 2, 3, padding=1)
            )
        elif mode == "upsample3d":
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"), nn.Conv2d(dim, dim // 2, 3, padding=1)
            )
            self.time_conv = CausalConv3d(dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))

        elif mode == "downsample2d":
            self.resample = nn.Sequential(nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2)))
        elif mode == "downsample3d":
            self.resample = nn.Sequential(nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2)))
            self.time_conv = CausalConv3d(dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))

        else:
            self.resample = nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        b, c, t, h, w = x.size()
        if self.mode == "upsample3d":
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = "Rep"
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, :, -CACHE_T:, :, :].clone()
                    if cache_x.shape[2] < 2 and feat_cache[idx] is not None and feat_cache[idx] != "Rep":
                        # cache last frame of last two chunk
                        cache_x = torch.cat(
                            [feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2
                        )
                    if cache_x.shape[2] < 2 and feat_cache[idx] is not None and feat_cache[idx] == "Rep":
                        cache_x = torch.cat([torch.zeros_like(cache_x).to(cache_x.device), cache_x], dim=2)
                    if feat_cache[idx] == "Rep":
                        x = self.time_conv(x)
                    else:
                        x = self.time_conv(x, feat_cache[idx])
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1

                    x = x.reshape(b, 2, c, t, h, w)
                    x = torch.stack((x[:, 0, :, :, :, :], x[:, 1, :, :, :, :]), 3)
                    x = x.reshape(b, c, t * 2, h, w)
        
        t = x.shape[2]
        x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        x = self.resample(x)
        x = x.view(b, t, x.size(1), x.size(2), x.size(3)).permute(0, 2, 1, 3, 4)

        if self.mode == "downsample3d":
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = x.clone()
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, :, -1:, :, :].clone()
                    x = self.time_conv(torch.cat([feat_cache[idx][:, :, -1:, :, :], x], 2))
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1
        return x


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout: float = 0.0,
        non_linearity: str = "silu",
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.nonlinearity = get_activation(non_linearity)

        # layers
        self.norm1 = RMS_norm(in_dim, images=False)
        self.conv1 = CausalConv3d(in_dim, out_dim, 3, padding=1)
        self.norm2 = RMS_norm(out_dim, images=False)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = CausalConv3d(out_dim, out_dim, 3, padding=1)
        self.conv_shortcut = CausalConv3d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        # Apply shortcut connection
        h = self.conv_shortcut(x)

        # First normalization and activation
        x = self.norm1(x)
        x = self.nonlinearity(x)

        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)

            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        # Second normalization and activation
        x = self.norm2(x)
        x = self.nonlinearity(x)

        # Dropout
        x = self.dropout(x)

        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)

            x = self.conv2(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv2(x)

        # Add residual connection
        return x + h


# this is shape preserving and can be dropped in diff attn. scales
class AttentionBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        # layers
        self.norm = RMS_norm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        identity = x
        batch_size, channels, time, height, width = x.size()

        x = x.permute(0, 2, 1, 3, 4).reshape(batch_size * time, channels, height, width)
        x = self.norm(x)

        # compute query, key, value
        qkv = self.to_qkv(x)
        qkv = qkv.reshape(batch_size * time, 1, channels * 3, -1)
        qkv = qkv.permute(0, 1, 3, 2).contiguous()
        q, k, v = qkv.chunk(3, dim=-1)

        # apply attention
        x = F.scaled_dot_product_attention(q, k, v)

        x = x.squeeze(1).permute(0, 2, 1).reshape(batch_size * time, channels, height, width)

        # output projection
        x = self.proj(x)

        # Reshape back: [(b*t), c, h, w] -> [b, c, t, h, w]
        x = x.view(batch_size, time, channels, height, width)
        x = x.permute(0, 2, 1, 3, 4)

        return x + identity


class MidBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.0, non_linearity: str = "silu", num_layers: int = 1):
        super().__init__()
        self.dim = dim

        resnets = [ResidualBlock(dim, dim, dropout, non_linearity)]
        attentions = []
        for _ in range(num_layers):
            attentions.append(AttentionBlock(dim))
            resnets.append(ResidualBlock(dim, dim, dropout, non_linearity))
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        self.gradient_checkpointing = False

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        # First residual block
        x = self.resnets[0](x, feat_cache, feat_idx)

        # Process through attention and residual blocks
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            if attn is not None:
                x = attn(x)

            x = resnet(x, feat_cache, feat_idx)

        return x


class Encoder3d(nn.Module):
    def __init__(
        self,
        dim=128,
        z_dim=4,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[True, True, False],
        dropout=0.0,
        non_linearity: str = "silu",
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample
        self.nonlinearity = get_activation(non_linearity)

        # dimensions
        dims = [dim * u for u in [1] + dim_mult]
        scale = 1.0

        # init block
        self.conv_in = CausalConv3d(3, dims[0], 3, padding=1)

        # downsample blocks
        self.down_blocks = nn.ModuleList([])
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # residual (+attention) blocks
            for _ in range(num_res_blocks):
                self.down_blocks.append(ResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    self.down_blocks.append(AttentionBlock(out_dim))
                in_dim = out_dim

            # downsample block
            if i != len(dim_mult) - 1:
                mode = "downsample3d" if temperal_downsample[i] else "downsample2d"
                self.down_blocks.append(Resample(out_dim, mode=mode))
                scale /= 2.0

        # middle blocks
        self.mid_block = MidBlock(out_dim, dropout, non_linearity, num_layers=1)

        # output blocks
        self.norm_out = RMS_norm(out_dim, images=False)
        self.conv_out = CausalConv3d(out_dim, z_dim, 3, padding=1)

        self.gradient_checkpointing = False

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                # cache last frame of last two chunk
                cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
            x = self.conv_in(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv_in(x)

        ## downsamples
        for layer in self.down_blocks:
            if feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        ## middle
        x = self.mid_block(x, feat_cache, feat_idx)

        ## head
        x = self.norm_out(x)
        x = self.nonlinearity(x)
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                # cache last frame of last two chunk
                cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
            x = self.conv_out(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv_out(x)
        return x


class UpBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_res_blocks: int,
        dropout: float = 0.0,
        upsample_mode: Optional[str] = None,
        non_linearity: str = "silu",
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # Create layers list
        resnets = []
        # Add residual blocks and attention if needed
        current_dim = in_dim
        for _ in range(num_res_blocks + 1):
            resnets.append(ResidualBlock(current_dim, out_dim, dropout, non_linearity))
            current_dim = out_dim

        self.resnets = nn.ModuleList(resnets)

        # Add upsampling layer if needed
        self.upsamplers = None
        if upsample_mode is not None:
            self.upsamplers = nn.ModuleList([Resample(out_dim, mode=upsample_mode)])

        self.gradient_checkpointing = False

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        for resnet in self.resnets:
            if feat_cache is not None:
                x = resnet(x, feat_cache, feat_idx)
            else:
                x = resnet(x)

        if self.upsamplers is not None:
            if feat_cache is not None:
                x = self.upsamplers[0](x, feat_cache, feat_idx)
            else:
                x = self.upsamplers[0](x)
        return x


class Decoder3d(nn.Module):
    def __init__(
        self,
        dim=128,
        z_dim=4,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_upsample=[False, True, True],
        dropout=0.0,
        non_linearity: str = "silu",
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_upsample = temperal_upsample

        self.nonlinearity = get_activation(non_linearity)

        # dimensions
        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        scale = 1.0 / 2 ** (len(dim_mult) - 2)

        # init block
        self.conv_in = CausalConv3d(z_dim, dims[0], 3, padding=1)

        # middle blocks
        self.mid_block = MidBlock(dims[0], dropout, non_linearity, num_layers=1)

        # upsample blocks
        self.up_blocks = nn.ModuleList([])
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # residual (+attention) blocks
            if i > 0:
                in_dim = in_dim // 2

            # Determine if we need upsampling
            upsample_mode = None
            if i != len(dim_mult) - 1:
                upsample_mode = "upsample3d" if temperal_upsample[i] else "upsample2d"

            # Create and add the upsampling block
            up_block = UpBlock(
                in_dim=in_dim,
                out_dim=out_dim,
                num_res_blocks=num_res_blocks,
                dropout=dropout,
                upsample_mode=upsample_mode,
                non_linearity=non_linearity,
            )
            self.up_blocks.append(up_block)

            # Update scale for next iteration
            if upsample_mode is not None:
                scale *= 2.0

        # output blocks
        self.norm_out = RMS_norm(out_dim, images=False)
        self.conv_out = CausalConv3d(out_dim, 3, 3, padding=1)

        self.gradient_checkpointing = False

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        ## conv1
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                # cache last frame of last two chunk
                cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
            x = self.conv_in(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv_in(x)

        ## middle
        x = self.mid_block(x, feat_cache, feat_idx)

        ## upsamples
        for up_block in self.up_blocks:
            x = up_block(x, feat_cache, feat_idx)

        ## head
        x = self.norm_out(x)
        x = self.nonlinearity(x)
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                # cache last frame of last two chunk
                cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
            x = self.conv_out(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv_out(x)
        return x

class Wan21VAE(VAEBase):
    def __init__(
        self,
        base_dim: int = 96,                     # rgb (3) -> base_dim (first convolution)
        z_dim: int = 16,                        # num. of channels in the latent
        dim_mult: Tuple[int] = [1, 2, 4, 4],
        num_res_blocks: int = 2,
        attn_scales: List[float] = [],
        temporal_downsample: List[bool] = [False, True, True],
        use_tiling: bool = True,
        max_batch_size: Union[int, None] = 4,
    ):
        super().__init__()
        
        self.z_dim = z_dim
        self.temporal_downsample = temporal_downsample
        self.temperal_upsample = temporal_downsample[::-1]
        
        self.encoder = Encoder3d(
            base_dim,
            z_dim * 2,
            dim_mult,
            num_res_blocks,
            attn_scales,
            self.temporal_downsample,
        )
        
        self.quant_conv = CausalConv3d(z_dim * 2, z_dim * 2, 1)
        self.post_quant_conv = CausalConv3d(z_dim, z_dim, 1)
        
        self.decoder = Decoder3d(
            base_dim, 
            z_dim, 
            dim_mult, 
            num_res_blocks,
            attn_scales, 
            self.temperal_upsample, 
        )
        
        # every downsample layer does spatial compression
        self.spatial_compression_ratio = 2 ** len(self.temporal_downsample)
        
        # slicing config
        self.use_slicing = max_batch_size != None and max_batch_size >= 1
        self.slice_batch_size = max(max_batch_size, 1)
        
        # tiling config
        self.use_tiling = use_tiling
    
        # number of causal convolution layers
        self._cached_conv_counts = {
            "decoder": sum(isinstance(m, CausalConv3d) for m in self.decoder.modules()) if self.decoder is not None else 0,
            "encoder": sum(isinstance(m, CausalConv3d) for m in self.encoder.modules()) if self.encoder is not None else 0,
        }
        
    def reset_causal_cache(self):
        self._conv_num = self._cached_conv_counts["decoder"]
        self._conv_idx = [0]
        self._feat_map = [None] * self._conv_num

        self._enc_conv_num = self._cached_conv_counts["encoder"]
        self._enc_conv_idx = [0]
        self._enc_feat_map = [None] * self._enc_conv_num

    @torch.inference_mode
    def encode(self, x: Tensor):
        B, C, T, H, W = x.shape
        # (T, H, W, C) -> (W, H, T)
        tile_width, tile_height, tile_temporal, overlap_width, overlap_height = self.get_tiling_config(input_shape=(W, H, T))
        encode_fn = partial(
            self.tiled_encode_3d, 
            tile_width=tile_width, 
            tile_height=tile_height, 
            tile_temporal=tile_temporal, 
            overlap_width=overlap_width, 
            overlap_height=overlap_height
        )   # this can be written in a much shorter way, but writing like this for readability 
        
        if self.use_slicing and x.shape[0] > self.slice_batch_size:
            encoded_slices = [encode_fn(x_slice) for x_slice in x.split(self.slice_batch_size)]
            h = torch.cat(encoded_slices)
        else:
            h = encode_fn(x)

        posterior = DiagonalGaussianDistribution(h)

        return posterior.sample()
    
    @torch.inference_mode
    def decode(self, z: Tensor, input_shape: Tuple):
        tile_width, tile_height, tile_temporal, overlap_width, overlap_height = self.get_tiling_config(input_shape=input_shape)
        decode_fn = partial(
            self.tiled_decode_3d, 
            tile_width=tile_width, 
            tile_height=tile_height, 
            tile_temporal=tile_temporal, 
            overlap_width=overlap_width, 
            overlap_height=overlap_height
        )   # this can be written in a much shorter way, but writing like this for readability 
        
        if self.use_slicing and z.shape[0] > 1:
            decoded_slices = [decode_fn(z_slice) for z_slice in z.split(1)]
            decoded = torch.cat(decoded_slices)
        else:
            decoded = decode_fn(z)

        return decoded
