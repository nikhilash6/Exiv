import torch
from torch import Tensor

from typing import List

from dataclasses import dataclass

@dataclass
class Latent:
    samples: List[Tensor]
    batch_index: List[int]
    noise_mask: List[Tensor]