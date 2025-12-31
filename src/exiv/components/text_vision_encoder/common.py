from torch import Tensor

from typing import Any, Dict
from dataclasses import dataclass, field

# PONDER: should this and the config classes be in the same file ?
@dataclass
class TextEncoderOutput:
    # main sequence output (Batch, Seq_Len, Dim)
    last_hidden_state: Tensor
    # optional: output of the second-to-last layer (often used in SDXL/Flux)
    penultimate_hidden_state: Tensor | None = None
    # optional: pooled vector (Batch, Dim) - T5 doesn't usually have this, but CLIP Text does
    pooled_output: Tensor | None = None
    # optional: all intermediate layers if needed
    hidden_states: tuple[Tensor] | None = None
    extra: Dict[str, Any] = field(default_factory=dict)

@dataclass
class VisionEncoderOutput:
    # main sequence output (Batch, Num_Patches, Dim)
    last_hidden_state: Tensor
    # pooled/projected vector (Batch, Proj_Dim) - main embedding for style/IPAdapter
    image_embedding: Tensor
    # intermediate layers (e.g. for controlnets or hybrid architectures)
    intermediate_hidden_states: Tensor | None = None
    # special extra projections (e.g. LLaVA layers)
    multimodal_projection: Tensor | None = None
    extra: Dict[str, Any] = field(default_factory=dict)