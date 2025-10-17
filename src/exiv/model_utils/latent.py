import torch
from torch import Tensor

from typing import List

from ..utils.device import ProcDevice

class Latent:
    def __init__(self, batch_size, channels, frame_count, height, width):
        self.samples: List[Tensor] = torch.zeros([batch_size, channels, frame_count, height, width], device=ProcDevice.CPU.value)
        batch_index: List[int] = None
        noise_mask: List[Tensor] = None