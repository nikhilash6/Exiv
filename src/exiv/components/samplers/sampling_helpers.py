import dataclasses
import torch
from torch import Tensor

import math
import collections
from typing import Any, List

from ...model_utils.common_classes import BatchedConditioning, Conditioning, ModelWrapper
from ...utils.tensor import common_upscale, repeat_to_batch_size


def prepare_mask(mask, shape, device):
    """
    Ensures the mask is of the proper dimensions.
    - Matches the number of dimensions of the target shape.
    - Interpolates the spatial dimensions (last two).
    - Adjusts the batch and channel dimensions to match the target shape.
    """
    mask = mask.to(device)
    
    # Ensure mask has the same number of dimensions as shape
    while mask.ndim < len(shape):
        mask = mask.unsqueeze(0)
    
    # Spatial interpolation on the last two dimensions
    if mask.shape[-2:] != shape[-2:]:
        mask = common_upscale(mask, shape[-1], shape[-2], upscale_method="bilinear", crop="none")

    # Adjust channel dimension (shape[1])
    if mask.shape[1] != shape[1]:
        if mask.shape[1] == 1:
            # Expand works for both 4D [B, C, H, W] and 5D [B, C, T, H, W]
            mask = mask.expand(mask.shape[0], shape[1], *mask.shape[2:])
        else:
            mask = repeat_to_batch_size(mask, shape[1], dim=1)

    # Adjust batch dimension (shape[0])
    mask = repeat_to_batch_size(mask, shape[0])
    
    # Adjust temporal dimension if 5D
    if mask.ndim == 5 and shape[2] != mask.shape[2]:
        mask = repeat_to_batch_size(mask, shape[2], dim=2)

    return mask

def filter_active_conds(
    batched_conditioning: BatchedConditioning, 
    current_progress: float
) -> BatchedConditioning:
    """
    Returns a shallow copy of BatchedConditioning containing only 
    conditions active for the current progress (0.0 start -> 1.0 end)
    """
    filtered = BatchedConditioning()
    filtered.execution_order = batched_conditioning.execution_order

    for group_name, cond_list in batched_conditioning.groups.items():
        active_list = []
        for cond in cond_list:
            start, end = cond.timestep_range
            # Check if current progress is within range (inclusive)
            if start <= current_progress <= end:
                active_list.append(cond)
        
        if active_list:
            filtered.groups[group_name] = active_list
            
    return filtered

def prepare_model_conds(
        wrapped_model: ModelWrapper, 
        batched_conditioning: BatchedConditioning, 
        noise: Tensor, 
        latent_image: Tensor, 
        denoise_mask: Tensor, 
        seed: Any
    ) -> BatchedConditioning:
    """
    (runs once)
    - adds 'model_input' to individual conditioning, basically a inference safe format
    - creates temporal mask based on frame range
    
    In each sample step, we go through these and pick the appropriate conds to apply for that 
    particular step.
    """
    
    res = BatchedConditioning()
    model = wrapped_model.model
    base_ctx = {
        "device": model.gpu_device,
        "noise": noise,
        "latent_image": latent_image,
        "denoise_mask": denoise_mask,
        "seed": seed,
        "spatial_compression_factor": model.model_arch_config.latent_format.spatial_compression_ratio
    }
    
    if len(noise.shape) >= 4:
        base_ctx["width"] = noise.shape[3] * base_ctx["spatial_compression_factor"]
        base_ctx["height"] = noise.shape[2] * base_ctx["spatial_compression_factor"]
        
    # shape is [B, C, T, H, W] for video or [B, C, H, W] for images
    has_temporal = noise.ndim == 5
    num_frames = noise.shape[2] if has_temporal else 1
    
    for cond_group_name, cond_list in batched_conditioning.get_groups_in_order():
        if cond_list is None: continue
        
        updated_conds = []
        for cond in cond_list:
            # shallow copy to avoid stale data in case of accidental re-use
            active_cond = dataclasses.replace(cond)
            
            # --- FRAME RANGE LOGIC ---
            if active_cond.frame_range is not None and has_temporal:
                f_start, f_end = active_cond.frame_range
                
                # broadcastable temporal mask: [1, 1, T, 1, 1]
                temporal_mask = torch.zeros((1, 1, num_frames, 1, 1), device=noise.device, dtype=noise.dtype)
                
                # clamping
                f_start = max(0, f_start)
                f_end = min(num_frames, f_end)
                temporal_mask[:, :, f_start:f_end, :, :] = 1.0
                
                if active_cond.mask is not None:
                    # handling spatial expansion (H,W -> T,H,W) and batching
                    existing_mask = prepare_mask(active_cond.mask, noise.shape, noise.device)
                    active_cond.mask = existing_mask * temporal_mask
                else:
                    # if no spatial mask is present, just use the temporal one
                    active_cond.mask = repeat_to_batch_size(temporal_mask, noise.shape[0])
            
            active_cond.model_input = model.format_conds(active_cond, **base_ctx)
            updated_conds.append(active_cond)
            
        res.add_cond_to_group(cond_group_name, updated_conds)
    
    # res = process_masks(res, noise.shape[2:], device)
    # res = prepare_controlnet(res)
    return res

# # NOTE: not very relevant atm, but will update as more models are added
# def process_masks(grouped_conds: dict, latent_dims, device):
#     """
#     - moves the mask tensor to the target device (e.g., GPU).
#     - resizes the mask to match the dimensions of the latent image.
#     """
#     for cond_group_name, cond_list in grouped_conds.items():
#         for cond in cond_list:
#             if 'mask' in cond:
#                 mask = cond['mask']
                
#                 # if mask is [H, W], batch becomes 1 and if its [B, H, W], batch is preserved
#                 batch_size = mask.shape[0] if mask.ndim > len(latent_dims) else 1
#                 target_shape = (batch_size, 1) + latent_dims
                
#                 # if it's [B, H, W] or [B, T, H, W], unsqueeze the channel dim 
#                 # so prepare_mask handles it as [B, C, ...] correctly
#                 if mask.ndim == len(latent_dims) + 1:
#                     mask = mask.unsqueeze(1)
                
#                 cond['mask'] = prepare_mask(mask, target_shape, device)
                
#     return grouped_conds

# # TODO: test when proper controlnet support is added
# def prepare_controlnet(grouped_conds: dict):
#     """
#     Ensures ControlNet is applied symmetrically to positive and negative conds.
#     If a ControlNet guides the positive prompt (e.g., with a depth map), the 
#     negative prompt also needs a corresponding "empty" ControlNet. This gives 
#     the model a clean baseline to push away from, making the ControlNet's guidance 
#     much more effective.
#     """
#     positive_conds = grouped_conds.get("positive", [])
#     negative_conds = grouped_conds.get("negative", [])

#     positive_controls = [
#         c['control'] for c in positive_conds
#         if 'control' in c and c['control'] is not None
#     ]

#     if not positive_controls:
#         return grouped_conds

#     neg_chunks_without_control = [
#         (chunk, i) for i, chunk in enumerate(negative_conds)
#         if chunk.get('control') is None
#     ]

#     if not neg_chunks_without_control:
#         return grouped_conds

#     # for each positive ControlNet, apply it to a corresponding negative prompt.
#     for i, control_to_add in enumerate(positive_controls):
#         # cycle through the available negative prompts.
#         target_chunk, chunk_index = neg_chunks_without_control[i % len(neg_chunks_without_control)]
#         new_chunk = target_chunk.copy()
#         new_chunk['control'] = control_to_add
#         grouped_conds["negative"][chunk_index] = new_chunk

#     return grouped_conds