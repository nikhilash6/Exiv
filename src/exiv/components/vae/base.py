import torch

import math
import numpy as np

from typing import Optional, Tuple

from ..enum import VAEType
from ...utils.tensor import random_tensor
from ...utils.device import VRAM_DEVICE
from ...model_utils.model_mixin import ModelMixin

class VAEBase(ModelMixin):
    # ------- these methods must be overriden in the child ---------
    # children must also define 'encoder' and 'decoder' properties (obviously)
    def get_tiling_config(self, input_shape, tile_x=256, tile_y=256, tile_z=4, overlap_x=64, overlap_y=64):
        assert input_shape is not None, "input shape is required to calc tile dims"
        
        # input_shape: (W, H, T)
        tile_x = min(tile_x, input_shape[0])
        tile_y = min(tile_y, input_shape[1])
        tile_z = min(tile_z, input_shape[2])
        
        # overlap shouldn't exceed 25% of the tile size
        overlap_x = min(overlap_x, tile_x // 4)
        overlap_y = min(overlap_y, tile_y // 4)

        return tile_x, tile_y, tile_z, overlap_x, overlap_y
    
    def reset_causal_cache(self):
        pass
    # ---------------------------------------------------------------

    # TODO: make these methods generalized as more and more models are added
    # TODO: handle the case of no tiling (but temporal chunking)
    def tiled_encode_3d(self, x: torch.Tensor, tile_width: int, tile_height: int, tile_temporal:int = 4, overlap_width=64, overlap_height=64) -> torch.Tensor:
        # (bs, channels, num_frames, height, width)
        _, _, num_frames, height, width = x.shape
        
        # final latent dims
        latent_height = height // self.spatial_compression_ratio
        latent_width = width // self.spatial_compression_ratio
        
        # latent tile and blend calc.
        og_stride_height = tile_height - overlap_height
        og_stride_width = tile_width - overlap_width
    
        blend_width = overlap_width // self.spatial_compression_ratio
        blend_height = overlap_height // self.spatial_compression_ratio
        latent_tile_size_height = tile_height // self.spatial_compression_ratio
        latent_tile_size_width = tile_width // self.spatial_compression_ratio
        latent_tile_stride_height = og_stride_height // self.spatial_compression_ratio
        latent_tile_stride_width = og_stride_width // self.spatial_compression_ratio

        # split and encode separately
        temporal_latent_rows = []
        for i in range(0, height, og_stride_height):
            cur_temporal_row = []
            for j in range(0, width, og_stride_width):
                self.reset_causal_cache()
                
                temporal_tile = []
                frame_range = 1 + (num_frames - 1) // tile_temporal
                for k in range(frame_range):
                    if k == 0:
                        # only process the first frame for the first iteration
                        # this is called with None cache (unlike the other frames)
                        tile = x[
                            :, 
                            :, 
                            :1, 
                            i : i + tile_height, 
                            j : j + tile_width
                        ]
                    else:
                        # k = 1 => 1 to 5, k = 2 => 5 to 9
                        tile = x[
                            :,
                            :,
                            1 + tile_temporal * (k - 1) : 1 + tile_temporal * k,
                            i : i + tile_height,
                            j : j + tile_width,
                        ]
                    
                    self._enc_conv_idx = [0]    # resetting each itr, all the layers of conv are passed again
                    tile = tile.to(VRAM_DEVICE)
                    res_tile = self.encoder(tile, feat_cache=self._enc_feat_map, feat_idx=self._enc_conv_idx)
                    res_tile = self.quant_conv(res_tile)
                    temporal_tile.append(res_tile)                           # complete temporal latent given a starting x,y tile idx
                    del tile
                cur_temporal_row.append(torch.cat(temporal_tile, dim=2))     # complete temporal latent given a starting x and all y tile idx
            temporal_latent_rows.append(cur_temporal_row)                    # complete temporal latents for all x,y variants

        # reassemble and blend the latent tiles
        result_temporal_latent_rows = []
        for i, cur_temporal_row in enumerate(temporal_latent_rows):
            result_row = []
            for j, temporal_tile in enumerate(cur_temporal_row):
                # wish i could add diagrams to code comments
                if i > 0:
                    # blending with the blocks above (starting from the second row)
                    temporal_tile = self.blend_v(temporal_latent_rows[i - 1][j], temporal_tile, blend_height)
                if j > 0:
                    # blending with the blocks on the left
                    temporal_tile = self.blend_h(cur_temporal_row[j - 1], temporal_tile, blend_width)
                result_row.append(temporal_tile[:, :, :, :latent_tile_stride_height, :latent_tile_stride_width])
            result_temporal_latent_rows.append(torch.cat(result_row, dim=-1))

        enc = torch.cat(result_temporal_latent_rows, dim=3)[:, :, :, :latent_height, :latent_width]
        return enc

    def tiled_decode_3d(self, z: torch.Tensor, tile_width: int, tile_height: int, tile_temporal:int = 4,  overlap_width=64, overlap_height=64) -> torch.Tensor:
        # (bs, channels, num_frames, height, width)
        _, _, num_frames, height, width = z.shape
        
        # final sample dims
        sample_height = height * self.spatial_compression_ratio
        sample_width = width * self.spatial_compression_ratio

        # doing the exact same calc. as the encoder
        # latent tile and blend calc.
        og_stride_height = tile_height - overlap_height
        og_stride_width = tile_width - overlap_width
    
        blend_width = overlap_width // self.spatial_compression_ratio
        blend_height = overlap_height // self.spatial_compression_ratio
        latent_tile_size_height = tile_height // self.spatial_compression_ratio
        latent_tile_size_width = tile_width // self.spatial_compression_ratio
        latent_tile_stride_height = og_stride_height // self.spatial_compression_ratio
        latent_tile_stride_width = og_stride_width // self.spatial_compression_ratio

        # split and decode (same as encode)
        temporal_latent_rows = []
        for i in range(0, height, latent_tile_stride_height):
            cur_temporal_row = []
            for j in range(0, width, latent_tile_stride_width):
                self.reset_causal_cache()
                time = []
                for k in range(num_frames):
                    self._conv_idx = [0]
                    tile = z[:, :, k : k + 1, i : i + latent_tile_size_height, j : j + latent_tile_size_width]
                    tile = self.post_quant_conv(tile)
                    decoded = self.decoder(tile, feat_cache=self._feat_map, feat_idx=self._conv_idx)
                    time.append(decoded)
                cur_temporal_row.append(torch.cat(time, dim=2))
            temporal_latent_rows.append(cur_temporal_row)
        self.reset_causal_cache()

        result_temporal_latent_rows = []
        for i, row in enumerate(temporal_latent_rows):
            result_row = []
            for j, tile in enumerate(row):
                # blend the above tile and the left tile
                # to the current tile and add the current tile to the result row
                if i > 0:
                    tile = self.blend_v(temporal_latent_rows[i - 1][j], tile, blend_height)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_width)
                result_row.append(tile[:, :, :, :og_stride_height, :og_stride_width])
            result_temporal_latent_rows.append(torch.cat(result_row, dim=-1))

        dec = torch.cat(result_temporal_latent_rows, dim=3)[:, :, :, :sample_height, :sample_width]

        return dec

    def blend_v(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        """
        Blends the bottom of tensor 'a' with the top of tensor 'b' vertically.
        It performs a linear interpolation over the `blend_extent` (overlap region).
        """
        # Ensure blend extent is not larger than the tensors themselves.
        blend_extent = min(a.shape[-2], b.shape[-2], blend_extent)
        # Iterate over each row in the vertical overlap region.
        for y in range(blend_extent):
            # Calculate the blend ratio, from 0.0 to 1.0.
            blend_ratio = y / blend_extent
            # The new pixel is a weighted average of the pixel from tensor 'a' and tensor 'b'.
            # At the start (y=0), it's 100% tensor 'a'. At the end, it's almost 100% tensor 'b'.
            b[:, :, :, y, :] = a[:, :, :, -blend_extent + y, :] * (1 - blend_ratio) + b[:, :, :, y, :] * blend_ratio
        return b

    def blend_h(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        """
        Blends the right side of tensor 'a' with the left side of tensor 'b' horizontally.
        """
        # Ensure blend extent is not larger than the tensors themselves.
        blend_extent = min(a.shape[-1], b.shape[-1], blend_extent)
        # Iterate over each column in the horizontal overlap region.
        for x in range(blend_extent):
            blend_ratio = x / blend_extent
            # Same linear interpolation logic as blend_v, but applied along the width dimension.
            b[:, :, :, :, x] = a[:, :, :, :, -blend_extent + x] * (1 - blend_ratio) + b[:, :, :, :, x] * blend_ratio
        return b

    
def get_vae(vae_type: VAEType) -> VAEBase:
    from .wan_vae import WanVAE
    
    if vae_type == VAEType.WAN:
        return WanVAE()
    
    raise Exception(f"{vae_type} vae not supported")


class DiagonalGaussianDistribution:
    def __init__(self, parameters: torch.Tensor, deterministic: bool = False):
        self.parameters = parameters
        # incase of wan vae, the encoder output has z_dim * 2 as dim 1
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)     # resolves to sqrt(var)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            # zero variance
            self.var = self.std = torch.zeros_like(
                self.mean, device=self.parameters.device, dtype=self.parameters.dtype
            )

    def sample(self, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        sample = random_tensor(
            self.mean.shape,
            generator=generator,
            device=self.parameters.device,
            dtype=self.parameters.dtype,
        )
        x = self.mean + self.std * sample
        return x

    def mode(self) -> torch.Tensor:
        return self.mean
    
    # these methods are not needed in inference
    def kl(self, other: "DiagonalGaussianDistribution" = None) -> torch.Tensor:
        if self.deterministic:
            return torch.Tensor([0.0])
        else:
            if other is None:
                return 0.5 * torch.sum(
                    torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar,
                    dim=[1, 2, 3],
                )
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var
                    - 1.0
                    - self.logvar
                    + other.logvar,
                    dim=[1, 2, 3],
                )

    def nll(self, sample: torch.Tensor, dims: Tuple[int, ...] = [1, 2, 3]) -> torch.Tensor:
        if self.deterministic:
            return torch.Tensor([0.0])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims,
        )
