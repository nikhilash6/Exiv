from dataclasses import dataclass, field
from typing import Optional, Any, Dict
import torch

@dataclass
class AROutput:
    """
    Unified output for all Autoregressive models in Exiv.
    Works for Main LM, Code Predictors, etc.
    """
    logits: torch.Tensor
    past_key_values: Optional[Any] = None
    
    # optional fields for research/debugging
    hidden_states: Optional[tuple[torch.Tensor]] = None
    attentions: Optional[tuple[torch.Tensor]] = None
    
    # placeholder for model-specific info (like Qwen's generation_steps)
    extra: Dict[str, Any] = field(default_factory=dict)