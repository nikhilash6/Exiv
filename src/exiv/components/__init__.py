from .enum import (
    KSamplerType,
    ModelType, 
    SamplerType, 
    SchedulerType, 
    TextEncoderType, 
    VisionEncoderType,
    VAEType
)

from .extensions import Extension

def __getattr__(name):
    if name == "LatentFormat":
        from .latent_format import LatentFormat
        return LatentFormat
    elif name == "VAEBase":
        from .vae.base import VAEBase
        return VAEBase
    elif name == "KSampler":
        from .samplers.model_sampling import KSampler
        return KSampler
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    "KSamplerType",
    "ModelType",
    "SamplerType",
    "SchedulerType",
    "TextEncoderType",
    "VisionEncoderType",
    "VAEType",
    "Extension",
    "LatentFormat",
    "VAEBase",
    "KSampler",
]