from dataclasses import dataclass
from typing import Optional


@dataclass
class VisionConfig:
    attention_dropout: float = 0.0
    dropout: float = 0.0
    hidden_act: str = "quick_gelu"
    hidden_size: int = 768
    image_size: int = 224
    initializer_factor: float = 1.0
    initializer_range: float = 0.02
    intermediate_size: int = 3072
    layer_norm_eps: float = 1e-05
    model_type: str = "clip_vision_model"
    num_attention_heads: int = 12
    num_channels: int = 3
    num_hidden_layers: int = 12
    patch_size: int = 32
    projection_dim: int = 512
    torch_dtype: str = "float32"
    # specific to some variants
    projector_type: Optional[str] = None

@dataclass
class CLIPViTLConfig(VisionConfig):
    hidden_act: str = "quick_gelu"
    hidden_size: int = 1024
    intermediate_size: int = 4096
    num_attention_heads: int = 16
    num_hidden_layers: int = 24
    patch_size: int = 14
    projection_dim: int = 768

@dataclass
class CLIPViTHConfig(VisionConfig):
    hidden_act: str = "gelu"
    hidden_size: int = 1280
    intermediate_size: int = 5120
    num_attention_heads: int = 16
    num_hidden_layers: int = 32
    patch_size: int = 14
    projection_dim: int = 1024

@dataclass
class CLIPVitLlavaConfig(CLIPViTLConfig):
    image_size: int = 336
    projector_type: str = "llava3"