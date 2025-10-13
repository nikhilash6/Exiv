import torch

import numpy as np

from typing import Optional, Tuple

from .wan_vae import WanVAE
from ..enum import VAEType
from ...model_utils.model_mixin import ModelMixin

class VAEBase(ModelMixin):
    def get_tiling_config(self, tile_size, overlap=64, temporal_size=64, temporal_overlap=8):
        # tile_size -> square tiles
        
        # overlap shouldn't exceed 25% of thetile size
        overlap = min(overlap, tile_size // 4)
        # overlap shouldn't exceed 50% tile size
        temporal_overlap = min(temporal_overlap, temporal_size // 2)

        return overlap, temporal_overlap
        
def get_vae(vae_type: VAEType) -> VAEBase:
    if vae_type == VAEType.WAN:
        return WanVAE()
    
    raise Exception(f"{vae_type} vae not supported")


class DiagonalGaussianDistribution(object):
    def __init__(self, parameters: torch.Tensor, deterministic: bool = False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(
                self.mean, device=self.parameters.device, dtype=self.parameters.dtype
            )

    # TODO: add randn_tensor and remove device movement
    def sample(self, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        sample = randn_tensor(
            self.mean.shape,
            generator=generator,
            device=self.parameters.device,
            dtype=self.parameters.dtype,
        )
        x = self.mean + self.std * sample
        return x

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

    def mode(self) -> torch.Tensor:
        return self.mean

