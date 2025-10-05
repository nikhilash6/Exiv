import torch

from ...utils.device import DEFAULT_DEVICE


def latent_noise_generator(
    batch_size, 
    channels, 
    height, 
    width, 
    dtype = torch.float16, 
    device = None, 
    seed = None
):
    if not seed: seed = 123
    device = device or DEFAULT_DEVICE
    return torch.randn(
        batch_size,
        channels,
        height,
        width,
        dtype=dtype,
        generator=torch.Generator(device="cpu").manual_seed(seed),
    ).to(device)