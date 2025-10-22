import torch
from torch import Tensor

from typing import List

from ..utils.device import OFFLOAD_DEVICE

class Latent:
    def __init__(self, batch_size, channels, frame_count, height, width):
        self.samples: List[Tensor] = torch.zeros([batch_size, channels, frame_count, height, width], device=OFFLOAD_DEVICE)
        batch_index: List[int] = None
        noise_mask: List[Tensor] = torch.ones([batch_size, channels, frame_count, height, width], device=OFFLOAD_DEVICE)