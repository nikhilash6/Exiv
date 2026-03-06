import torch
import torch.nn.functional as F

from typing import Optional, Tuple

from ...utils.tensor import random_tensor

# NOTE: keeping this method outside of VAEBase for now, but will merge later
class VAEImageProcessor:
    def __init__(self, vae_scale_factor=8, crop_mode="center"):
        self.vae_scale_factor = vae_scale_factor
        self.crop_mode = crop_mode

    def process_image(self, image, width=None, height=None):
        # Normalize to [N, C, H, W] regardless of input (Video or Image)
        packed_image, info = self._pack_to_nchw(image)

        if width and height:
            packed_image = self.resize(packed_image, width, height)

        packed_image = self.ensure_rgb_channels(packed_image)
        packed_image = self.crop_to_multiples(packed_image)

        return self._unpack_from_nchw(packed_image, info)

    # PONDER: this maybe useful in other parts of the code 
    def _pack_to_nchw(self, image):
        from PIL import Image
        import numpy as np
        # Standardize Input Types
        if isinstance(image, Image.Image):
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image).unsqueeze(0)
        elif isinstance(image, np.ndarray):
            image = torch.from_numpy(image).float()
            if image.ndim == 3: image = image.unsqueeze(0)
        
        ndim = image.ndim
        is_channels_last = image.shape[-1] in [1, 3, 4]

        # 1. Handle 5D Video
        if ndim == 5:
            # Normalize to [B, T, C, H, W]
            # Moves Channels (from 1 or -1) to index 2
            image = image.movedim(-1 if is_channels_last else 1, 2)
            
            # Capture dims safely AFTER alignment
            b, t = image.shape[0], image.shape[1]
            image = image.flatten(0, 1) # Merge B and T -> [N, C, H, W]

        # 2. Handle 4D Image
        else:
            b, t = image.shape[0], 1
            if is_channels_last:
                image = image.movedim(-1, 1) # BHWC -> BCHW

        return image, {"ndim": ndim, "clast": is_channels_last, "b": b, "t": t}

    def _unpack_from_nchw(self, image, info):
        if info["ndim"] == 5:
            # Restore Video: [N, C, H, W] -> [B, T, C, H, W]
            image = image.unflatten(0, (info["b"], info["t"]))
            # Restore Channels: Move C back to 1 (BCTHW) or -1 (BTHWC)
            image = image.movedim(2, -1 if info["clast"] else 1)
        elif info["clast"]:
            # Restore Image: BCHW -> BHWC
            image = image.movedim(1, -1)
            
        return image

    def resize(self, image, width, height):
        return F.interpolate(image, size=(height, width), mode="bilinear", align_corners=False)

    def ensure_rgb_channels(self, image):
        c = image.shape[1]
        if c == 3: return image
        # Discard Alpha or duplicate Grayscale to RGB
        if c > 3: return image[:, :3]
        return image[:, :1].repeat(1, 3, 1, 1)

    def crop_to_multiples(self, image):
        h, w = image.shape[2:]
        scale = self.vae_scale_factor
        new_h, new_w = (h // scale) * scale, (w // scale) * scale
        
        if new_h == h and new_w == w: return image
        
        # Calculate crop offsets
        y = (h - new_h) // 2 if self.crop_mode == "center" else 0
        x = (w - new_w) // 2 if self.crop_mode == "center" else 0
        
        return image[:, :, y:y+new_h, x:x+new_w]


class DiagonalGaussianDistribution:
    def __init__(self, parameters: torch.Tensor, deterministic: bool = False):
        self.parameters = parameters
        # incase of wan vae, the encoder output has z_dim * 2 as dim 1
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)     # resolves to sqrt(var)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            # zero variance
            self.var = self.std = torch.zeros_like(
                self.mean, device=self.parameters.device, dtype=self.parameters.dtype
            )

    def sample(self, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        sample = random_tensor(
            self.mean.shape,
            generator=generator,
            device=self.parameters.device,
            dtype=self.parameters.dtype,
        )
        x = self.mean + self.std * sample
        return x

    def mode(self) -> torch.Tensor:
        return self.mean
    
    # these methods are not needed in inference
    def kl(self, other: "DiagonalGaussianDistribution" = None) -> torch.Tensor:
        if self.deterministic:
            return torch.Tensor([0.0])
        else:
            if other is None:
                return 0.5 * torch.sum(
                    torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar,
                    dim=[1, 2, 3],
                )
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var
                    - 1.0
                    - self.logvar
                    + other.logvar,
                    dim=[1, 2, 3],
                )

    def nll(self, sample: torch.Tensor, dims: Tuple[int, ...] = [1, 2, 3]) -> torch.Tensor:
        import numpy as np
        if self.deterministic:
            return torch.Tensor([0.0])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims,
        )
