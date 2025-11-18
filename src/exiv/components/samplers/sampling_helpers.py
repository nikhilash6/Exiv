from typing import Any
import torch
from torch import Tensor

import collections

from ...model_utils.common_classes import ModelWrapper
from ...utils.tensor import common_upscale, repeat_to_batch_size


def preprocess_cond(conds, x_in):
    '''
    - calculates conditionals strength
    - reshapes model conditionals to match bs
    - separates controlnet conditionals
    '''
    # ---- strength calc
    strength = conds.get('strength', 1.0)
    mult = torch.ones_like(x_in) * strength

    # ---- prepare conditioning
    conditioning = {}
    model_conds = conds["model_conds"]
    for c in model_conds:
        conditioning[c] = model_conds[c].process_cond(batch_size=x_in.shape[0], device=x_in.device)

    # ---- controlnets
    control = conds.get('control', None)

    cond_obj = collections.namedtuple('cond_obj', ['input_x', 'mult', 'conditioning', 'control'])
    return cond_obj(x_in, mult, conditioning, control)


def prepare_mask(noise_mask, shape, device):
    """ensures noise mask is of proper dimensions"""
    noise_mask = torch.nn.functional.interpolate(noise_mask.reshape((-1, 1, noise_mask.shape[-2], noise_mask.shape[-1])), size=(shape[2], shape[3]), mode="bilinear")
    noise_mask = torch.cat([noise_mask] * shape[1], dim=1)
    noise_mask = repeat_to_batch_size(noise_mask, shape[0])
    noise_mask = noise_mask.to(device)
    return noise_mask


def prepare_model_conds(wrapped_model: ModelWrapper, grouped_conds: dict, noise: Tensor, latent_image: Tensor, denoise_mask: Tensor, seed: Any):
    """
    This creates the model_conds, basically a dict with all the conds in the correct format.
    In each sample step, we go through these and pick the appropriate conds to apply for that 
    particular step.
    
    grouped_conds : {'positive': [], 'negative': [], ...}
    noise_shape: tuple
    device: str / torch.device
    """
    
    device = wrapped_model.model.gpu_device
    
    for cond_group_name, cond_list in grouped_conds.items():
        if cond_list is not None:
            grouped_conds[cond_group_name] = wrapped_model.model.prepare_conds_for_model(
                cond_group_name, 
                cond_list, 
                noise, 
                spatial_compression_factor=wrapped_model.model.model_arch_config.latent_format.spatial_compression_ratio, 
                latent_image=latent_image,
                denoise_mask=denoise_mask,
                seed=seed
            )
    
    grouped_conds = process_masks(grouped_conds, noise.shape[2:], device)
    grouped_conds = prepare_controlnet(grouped_conds)
    return grouped_conds

# NOTE: not very relevant atm, but will update as more models are added
def process_masks(grouped_conds: dict, latent_dims, device):
    """
    - moves the mask tensor to the target device (e.g., GPU).
    - resizes the mask to match the dimensions of the latent image.
    """
    for cond_group_name, cond_list in grouped_conds.items():
        for cond in cond_list:
            if 'mask' in cond:
                mask = cond['mask']
                mask = mask.to(device=device)

                # adding a batch dimension
                if len(mask.shape) == len(latent_dims):
                    mask = mask.unsqueeze(0)

                if mask.shape[1:] != latent_dims:
                    if mask.ndim < 4:
                        # adding the channel dim then removing it
                        mask = common_upscale(mask.unsqueeze(1), latent_dims[-1], latent_dims[-2], 'bilinear', 'none').squeeze(1)
                    else:
                        mask = common_upscale(mask, latent_dims[-1], latent_dims[-2], 'bilinear', 'none')

                cond['mask'] = mask
                
    return grouped_conds

# TODO: test when proper controlnet support is added
def prepare_controlnet(grouped_conds: dict):
    """
    Ensures ControlNet is applied symmetrically to positive and negative conds.
    If a ControlNet guides the positive prompt (e.g., with a depth map), the 
    negative prompt also needs a corresponding "empty" ControlNet. This gives 
    the model a clean baseline to push away from, making the ControlNet's guidance 
    much more effective.
    """
    positive_conds = grouped_conds.get("positive", [])
    negative_conds = grouped_conds.get("negative", [])

    positive_controls = [
        c['control'] for c in positive_conds
        if 'control' in c and c['control'] is not None
    ]

    if not positive_controls:
        return

    neg_chunks_without_control = [
        (chunk, i) for i, chunk in enumerate(negative_conds)
        if chunk.get('control') is None
    ]

    if not neg_chunks_without_control:
        return

    # for each positive ControlNet, apply it to a corresponding negative prompt.
    for i, control_to_add in enumerate(positive_controls):
        # cycle through the available negative prompts.
        target_chunk, chunk_index = neg_chunks_without_control[i % len(neg_chunks_without_control)]
        new_chunk = target_chunk.copy()
        new_chunk['control'] = control_to_add
        grouped_conds["negative"][chunk_index] = new_chunk

    return grouped_conds