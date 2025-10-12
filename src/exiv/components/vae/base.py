from .wan_vae import WanVAE
from ..enum import VAEType


class VAEBase:
    pass

def get_vae(vae_type: VAEType) -> VAEBase:
    if vae_type == VAEType.WAN:
        return WanVAE()
    
    raise Exception(f"{vae_type} vae not supported")


