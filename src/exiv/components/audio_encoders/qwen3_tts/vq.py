# Copyright 2026 The Qwen team, Alibaba Group. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

"""
   SplitResidualVectorQuantizer  <-- Top-Level Entry Point (Receives the 8 codes)
   │
   ├── rvq_first: ResidualVectorQuantizer           <-- Handles Code #1 (Semantic: "What word is said")
   │   │
   │   ├── input_proj / output_proj                 <-- Adjusts vector dimensions if needed
   │   └── vq: ResidualVectorQuantization           <-- Core Loop logic
   │       └── layers: ModuleList
   │           └── [0] VectorQuantization           <-- Wrapper for Layer 1
   │               ├── project_out                  <-- Linear projection layer
   │               └── _codebook: EuclideanCodebook <-- The "Average Sum/Usage" Math Table
   │
   └── rvq_rest: ResidualVectorQuantizer            <-- Handles Codes #2 through #8 (Acoustic: "Who is saying it, tone, breath")
       │
       ├── input_proj / output_proj
       └── vq: ResidualVectorQuantization           <-- Loops over the remaining 7 codes and ADDS their vectors together
           └── layers: ModuleList
               ├── [0] VectorQuantization           <-- Wrapper for Layer 2
               │   └── _codebook: EuclideanCodebook
               ├── [1] VectorQuantization           <-- Wrapper for Layer 3
               │   └── _codebook: EuclideanCodebook
               ├── ...
               └── [6] VectorQuantization           <-- Wrapper for Layer 8
                   └── _codebook: EuclideanCodebook
"""



class EuclideanCodebook(nn.Module):
    def __init__(self, dim: int, codebook_size: int, epsilon: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.codebook_size = codebook_size
        self.epsilon = epsilon
        self.cluster_usage = nn.Parameter(torch.ones(codebook_size))
        self.embedding_sum = nn.Parameter(torch.zeros(codebook_size, dim))

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        embedding = self.embedding_sum / self.cluster_usage.clamp(min=self.epsilon)[:, None]
        return F.embedding(codes, embedding)


class VectorQuantization(nn.Module):
    def __init__(self, dim: int, codebook_size: int, codebook_dim: Optional[int] = None, epsilon: float = 1e-5):
        super().__init__()
        if codebook_dim is None:
            codebook_dim = dim
        self.project_out = nn.Linear(codebook_dim, dim) if codebook_dim != dim else nn.Identity()
        self.epsilon = epsilon
        self._codebook = EuclideanCodebook(dim=codebook_dim, codebook_size=codebook_size, epsilon=epsilon)
        self.codebook_size = codebook_size

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        quantized = self._codebook.decode(codes)
        quantized = self.project_out(quantized)
        return quantized.transpose(1, 2)


class ResidualVectorQuantization(nn.Module):
    def __init__(self, *, num_quantizers: int, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList([VectorQuantization(**kwargs) for _ in range(num_quantizers)])

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        quantized = torch.zeros([1], device=codes.device)[0]
        for idx, layer_codes in enumerate(codes):
            quantized = quantized + self.layers[idx].decode(layer_codes)
        return quantized


class ResidualVectorQuantizer(nn.Module):
    def __init__(self, dimension: int = 128, input_dimension: Optional[int] = None, output_dimension: Optional[int] = None, n_q: int = 8, bins: int = 1024, force_projection: bool = False, **kwargs):
        super().__init__()
        self.n_q = n_q
        self.dimension = dimension
        self.input_proj = nn.Identity() if (input_dimension or dimension) == dimension and not force_projection else nn.Conv1d(input_dimension or dimension, dimension, 1, bias=False)
        self.output_proj = nn.Identity() if (output_dimension or dimension) == dimension and not force_projection else nn.Conv1d(dimension, output_dimension or dimension, 1, bias=False)
        self.vq = ResidualVectorQuantization(dim=self.dimension, codebook_size=bins, num_quantizers=self.n_q)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        codes = codes.transpose(0, 1)
        quantized = self.vq.decode(codes)
        return self.output_proj(quantized)


class SplitResidualVectorQuantizer(nn.Module):
    def __init__(self, *, n_q: int = 8, n_q_semantic: int = 1, **kwargs):
        super().__init__()
        self.n_q_semantic = n_q_semantic
        self.rvq_first = ResidualVectorQuantizer(n_q=n_q_semantic, force_projection=True, **kwargs)
        self.rvq_rest = ResidualVectorQuantizer(n_q=n_q - n_q_semantic, force_projection=True, **kwargs)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        quantized = self.rvq_first.decode(codes[:, : self.n_q_semantic])
        if codes.shape[1] > self.n_q_semantic:
            quantized += self.rvq_rest.decode(codes[:, self.n_q_semantic :])
        return quantized
