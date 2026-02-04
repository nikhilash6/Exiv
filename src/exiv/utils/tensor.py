import torch
from torch import Tensor

import math
import numpy as np
from typing import Union, Tuple, List, Optional

from PIL import Image

from ..model_utils.common_classes import ModelWrapper
from ..utils.logging import app_logger

# many of these methods have been borrowed from ComfyUI

def repeat_to_batch_size(tensor: torch.Tensor, batch_size: int, dim=0):
    # repeats or divides the tensor as per the given batch_size
    if tensor.shape[dim] > batch_size:
        return tensor.narrow(dim, 0, batch_size)
    elif tensor.shape[dim] < batch_size:
        return tensor.repeat(dim * [1] + [math.ceil(batch_size / tensor.shape[dim])] + [1] * (len(tensor.shape) - 1 - dim)).narrow(dim, 0, batch_size)
    return tensor

def fix_empty_latent_channels(wrapped_model: ModelWrapper, latent_image: torch.Tensor):
    # resize the empty latent image so it has the right number of channels
    latent_channels = wrapped_model.model.model_arch_config.latent_format.latent_channels
    if latent_channels != latent_image.shape[1] and torch.count_nonzero(latent_image) == 0:
        latent_image = repeat_to_batch_size(latent_image, latent_channels, dim=1)
    return latent_image

def prepare_noise(latent_image: torch.Tensor, seed: int | None = None, noise_inds: Optional[np.ndarray] = None):
    """
    Creates random noise tensors based on a latent image's shape, dtype, and layout are used
    e.g. usage, noise_inds = [0, 0, 1, 1] -> first two will share the same noise + the last two will as well
    """
    
    generator = torch.Generator(device="cpu")
    if seed is not None: generator.manual_seed(seed)

    if noise_inds is None:
        return random_tensor(
            shape=latent_image.size(),
            generator=generator,
            device=latent_image.device,
            dtype=latent_image.dtype,
            layout=latent_image.layout,
        )

    unique_inds, inverse = np.unique(noise_inds, return_inverse=True)
    noises = []

    for i in range(unique_inds[-1] + 1):
        noise = random_tensor(
            shape=(1,) + tuple(latent_image.shape[1:]),
            generator=generator,
            device=latent_image.device,
            dtype=latent_image.dtype,
            layout=latent_image.layout,
        )
        if i in unique_inds:
            noises.append(noise)

    noises = [noises[i] for i in inverse]
    noises = torch.cat(noises, dim=0)
    return noises

def random_tensor(
    shape: Union[Tuple[int], List[int]],
    generator: Optional[torch.Generator] = None,
    device: Optional[Union[str, torch.device]] = None,
    dtype: Optional[torch.dtype] = None,
    layout: Optional[torch.layout] = None,
):
    if isinstance(device, str):
        device = torch.device(device)
    device = device or torch.device("cpu")
    layout = layout or torch.strided        # standard dense format
    rand_device = device

    if generator is not None:
        gen_device_type = (
            generator[0].device.type if isinstance(generator, list) else generator.device.type
        )

        # cpu generator -> tensor (cpu) -> gpu
        if gen_device_type != device.type and gen_device_type == "cpu":
            rand_device = torch.device("cpu")

        # gpu generator -> tensor (gpu) -> cpu (movement to cpu not allowed)
        elif gen_device_type != device.type and gen_device_type == "cuda":
            raise ValueError(
                f"Cannot create a {device.type} tensor using a CUDA generator."
            )

    latents = torch.randn(
            shape, generator=generator, device=rand_device, dtype=dtype, layout=layout
        ).to(device)

    return latents


def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f'input has {x.ndim} dims but target_dims is {target_dims}, which is less')
    expanded = x[(...,) + (None,) * dims_to_append]
    # MPS will get inf values if it tries to index into the new axes, but detaching fixes this.
    # https://github.com/pytorch/pytorch/issues/84364
    return expanded.detach().clone() if expanded.device.type == 'mps' else expanded

def bislerp(samples, width, height):
    def slerp(b1, b2, r):
        '''slerps batches b1, b2 according to ratio r, batches should be flat e.g. NxC'''

        c = b1.shape[-1]

        #norms
        b1_norms = torch.norm(b1, dim=-1, keepdim=True)
        b2_norms = torch.norm(b2, dim=-1, keepdim=True)

        #normalize
        b1_normalized = b1 / b1_norms
        b2_normalized = b2 / b2_norms

        #zero when norms are zero
        b1_normalized[b1_norms.expand(-1,c) == 0.0] = 0.0
        b2_normalized[b2_norms.expand(-1,c) == 0.0] = 0.0

        #slerp
        dot = (b1_normalized*b2_normalized).sum(1)
        omega = torch.acos(dot)
        so = torch.sin(omega)

        #technically not mathematically correct, but more pleasing?
        res = (torch.sin((1.0-r.squeeze(1))*omega)/so).unsqueeze(1)*b1_normalized + (torch.sin(r.squeeze(1)*omega)/so).unsqueeze(1) * b2_normalized
        res *= (b1_norms * (1.0-r) + b2_norms * r).expand(-1,c)

        #edge cases for same or polar opposites
        res[dot > 1 - 1e-5] = b1[dot > 1 - 1e-5]
        res[dot < 1e-5 - 1] = (b1 * (1.0-r) + b2 * r)[dot < 1e-5 - 1]
        return res

    def generate_bilinear_data(length_old, length_new, device):
        coords_1 = torch.arange(length_old, dtype=torch.float32, device=device).reshape((1,1,1,-1))
        coords_1 = torch.nn.functional.interpolate(coords_1, size=(1, length_new), mode="bilinear")
        ratios = coords_1 - coords_1.floor()
        coords_1 = coords_1.to(torch.int64)

        coords_2 = torch.arange(length_old, dtype=torch.float32, device=device).reshape((1,1,1,-1)) + 1
        coords_2[:,:,:,-1] -= 1
        coords_2 = torch.nn.functional.interpolate(coords_2, size=(1, length_new), mode="bilinear")
        coords_2 = coords_2.to(torch.int64)
        return ratios, coords_1, coords_2

    orig_dtype = samples.dtype
    samples = samples.float()
    n,c,h,w = samples.shape
    h_new, w_new = (height, width)

    #linear w
    ratios, coords_1, coords_2 = generate_bilinear_data(w, w_new, samples.device)
    coords_1 = coords_1.expand((n, c, h, -1))
    coords_2 = coords_2.expand((n, c, h, -1))
    ratios = ratios.expand((n, 1, h, -1))

    pass_1 = samples.gather(-1,coords_1).movedim(1, -1).reshape((-1,c))
    pass_2 = samples.gather(-1,coords_2).movedim(1, -1).reshape((-1,c))
    ratios = ratios.movedim(1, -1).reshape((-1,1))

    result = slerp(pass_1, pass_2, ratios)
    result = result.reshape(n, h, w_new, c).movedim(-1, 1)

    #linear h
    ratios, coords_1, coords_2 = generate_bilinear_data(h, h_new, samples.device)
    coords_1 = coords_1.reshape((1,1,-1,1)).expand((n, c, -1, w_new))
    coords_2 = coords_2.reshape((1,1,-1,1)).expand((n, c, -1, w_new))
    ratios = ratios.reshape((1,1,-1,1)).expand((n, 1, -1, w_new))

    pass_1 = result.gather(-2,coords_1).movedim(1, -1).reshape((-1,c))
    pass_2 = result.gather(-2,coords_2).movedim(1, -1).reshape((-1,c))
    ratios = ratios.movedim(1, -1).reshape((-1,1))

    result = slerp(pass_1, pass_2, ratios)
    result = result.reshape(n, h_new, w_new, c).movedim(-1, 1)
    return result.to(orig_dtype)

def lanczos(samples, width, height):
    images = [Image.fromarray(np.clip(255. * image.movedim(0, -1).cpu().numpy(), 0, 255).astype(np.uint8)) for image in samples]
    images = [image.resize((width, height), resample=Image.Resampling.LANCZOS) for image in images]
    images = [torch.from_numpy(np.array(image).astype(np.float32) / 255.0).movedim(-1, 0) for image in images]
    result = torch.stack(images)
    return result.to(samples.device, samples.dtype)

def common_upscale(samples: List[Tensor] | Tensor, width, height, upscale_method = "lanczos", crop = "center") -> List[Tensor]:
    if not isinstance(samples, list): samples = [samples]
    output = []
    for sample in samples:
        while sample.ndim < 4: sample = sample.unsqueeze(0)    # getting in (B, C, H, W) form
        orig_shape = tuple(sample.shape)
        if len(orig_shape) > 4:
            sample = sample.reshape(sample.shape[0], sample.shape[1], -1, sample.shape[-2], sample.shape[-1])
            sample = sample.movedim(2, 1)
            sample = sample.reshape(-1, orig_shape[1], orig_shape[-2], orig_shape[-1])
        if crop == "center":
            old_width = sample.shape[-1]
            old_height = sample.shape[-2]
            old_aspect = old_width / old_height
            new_aspect = width / height
            x = 0
            y = 0
            if old_aspect > new_aspect:
                x = round((old_width - old_width * (new_aspect / old_aspect)) / 2)
            elif old_aspect < new_aspect:
                y = round((old_height - old_height * (old_aspect / new_aspect)) / 2)
            s = sample.narrow(-2, y, old_height - y * 2).narrow(-1, x, old_width - x * 2)
        else:
            s = sample

        if upscale_method == "bislerp":
            out = bislerp(s, width, height)
        elif upscale_method == "lanczos":
            out = lanczos(s, width, height)
        else:
            out = torch.nn.functional.interpolate(s, size=(height, width), mode=upscale_method)

        if len(orig_shape) == 4:
            output.append(out)
        else:
            out = out.reshape((orig_shape[0], -1, orig_shape[1]) + (height, width))
            out = out.movedim(2, 1).reshape(orig_shape[:-2] + (height, width))
            output.append(out)
    return output
    
def pad_to_patch_size(img, patch_size=(2, 2), padding_mode="circular"):
    # torchscript doesn't support circular padding
    if padding_mode == "circular" and (torch.jit.is_tracing() or torch.jit.is_scripting()):
        padding_mode = "reflect"

    pad = ()
    for i in range(img.ndim - 2):
        pad = (0, (patch_size[i] - img.shape[i + 2] % patch_size[i]) % patch_size[i]) + pad

    return torch.nn.functional.pad(img, pad, mode=padding_mode)

def tensor_to_parameter(x: Tensor):
    if isinstance(x, torch.nn.Parameter):
        return x
    else:
        return torch.nn.Parameter(x, requires_grad=False)
    
def get_tensor_weak_hash(tensor):
    if tensor is None: return None
    meta = (tensor.shape, tensor.dtype, tensor.device)
    # strided sampling (1D array)
    flat = tensor.view(-1)
    size = flat.numel()
    step = size // 10       # reading just 10 points
    if step == 0: step = 1
    samples = tuple(flat[::step][:10].tolist())
    return hash((meta, samples))