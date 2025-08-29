import torch

from ...utils.model_utils import ModelMixin


class LatentNoiseGenerator(ModelMixin):
    def __init__(self, device, dtype):
        super().__init__(device, dtype)
    
    def __call__(self, batch_size, channels, height, width, seed=None):
        if not seed: seed = 123
        return torch.randn(
            batch_size,
            channels,
            height,
            width,
            dtype=self.dtype,
            generator=torch.Generator(device="cpu").manual_seed(seed),
        ).to(self.device)