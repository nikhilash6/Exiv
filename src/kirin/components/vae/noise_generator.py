import torch

from ..base import IComponent


class NoiseGenerator(IComponent):
    def __init__(self, device, dtype):
        super().__init__(device, dtype)
    
    def __call__(self, dim1, dim2, dim3, dim4, seed=None):
        if not seed: seed = 123
        return torch.randn(
            dim1,       # num_samples / batch_size
            dim2,       # channels
            dim3,       # height
            dim4,       # width
            dtype=self.dtype,
            generator=torch.Generator(device="cpu").manual_seed(seed),
        ).to(self.device)