import dataclasses
import torch
from torch import Tensor

import math
import collections
from typing import Any, List

from ...utils.logging import app_logger
from ...model_utils.conditioning_mixin import ConditioningMixin
from ...model_utils.common_classes import BatchedConditioning, Conditioning, ExecutionBatch, ModelForwardInput, ModelWrapper
from ...utils.tensor import common_upscale, repeat_to_batch_size


def filter_active_conds(
    batched_conditioning: BatchedConditioning, 
    timestep: float
) -> BatchedConditioning:
    """
    Returns a shallow copy of BatchedConditioning containing only 
    conditions active for the timestep, using timestep_range and strength
    """
    filtered = BatchedConditioning()
    filtered.execution_order = batched_conditioning.execution_order

    for group_name, cond_list in batched_conditioning.groups.items():
        active_list = []
        for cond in cond_list:
            filter_out = False
            
            # timestep filtering
            if cond.timestep_range != (-1, -1):
                start, end = cond.timestep_range
                if timestep < start or timestep > end:
                    filter_out = True
                    
            # if per timestep strength is provided
            if cond.strength == 0:
                filter_out = True
            elif not filter_out and isinstance(cond.strenth, List[float]):
                try:
                    cur_timestep_strength = cond.strength[timestep - cond.timestep_range[0]]
                    if cur_timestep_strength == 0: filter_out = True
                except Exception as e:
                    app_logger.warning(f"Unable to calculate per step strength {e}")
                    
            if not filter_out: active_list.append(cond)
        
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
    - filters out model conds that are not supported by the model
    - adds 'model_input' to individual conditioning, basically an inference safe format
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
        
    for cond_group_name, cond_list in batched_conditioning.get_groups_in_order():
        cond_list = wrapped_model.model.filter_conditionings(cond_list)
        if cond_list is None: continue
        
        updated_conds = []
        for cond in cond_list:
            # shallow copy to avoid stale data in case of accidental re-use
            active_cond = dataclasses.replace(cond)
            active_cond.model_input = model.prepare_model_input(active_cond, **base_ctx)
            updated_conds.append(active_cond)
            
        res.set_group_cond(cond_group_name, updated_conds, replace=True)

    return res

def collate_inputs(inputs: List[ModelForwardInput]):
    """Merges a list of ModelForwardInputs into a single batched input."""
    if not inputs: return {}
    
    # TODO: assumes same makeup for pos/neg batches, will fail for heterogenous stuff like IPA
    keys = inputs[0].to_dict().keys()
    
    collated = {}
    for k in keys:
        values = [getattr(inp, k) for inp in inputs]
        if hasattr(values[0], 'concat'):
            collated[k] = values[0].concat(values[1:])
        elif isinstance(values[0], torch.Tensor):
            collated[k] = torch.cat(values, dim=0)
            
    return collated

def prepare_per_frame_timestep(timestep, num_tasks, denoise_mask):
    # NOTE: assuming denoise_mask shape to be [Batch, Channels, Frames, Height, Width]
    if denoise_mask is None:
        return timestep.repeat(num_tasks)
    
    # 1. Create a Per-Frame Mask
    # We average over Channels(1), Height(3), and Width(4).
    # Result: A tensor of shape [Batch, 1, Frames, 1, 1]
    # Value is 1.0 for "Generation Frames" and 0.0 for "Context Frames"
    # if ANY pixel in a frame is masked, the entire frame get the full timestep t
    frame_mask = torch.amax(denoise_mask, dim=(1, 3, 4), keepdim=True)
    
    # 2. Reshape the global timestep 't' to match the mask dimensions
    # t shape: [Batch] -> [Batch, 1, 1, 1, 1]
    t_reshaped = timestep.view(timestep.shape[0], 1, 1, 1, 1)
    
    # 3. Apply the timestep logic (The "Gate")
    # - Context Frames: 0.0 * t = 0  (Model sees them as "Clean/Finished")
    # - Gen Frames:     1.0 * t = t  (Model sees them as "Noisy/Current Step")
    per_frame_timesteps = frame_mask * t_reshaped
    
    # 4. Flatten back to [Batch, Frames] for the model
    batched_t = per_frame_timesteps.reshape(timestep.shape[0], -1)
    
    # 5. Repeat for all parallel tasks (Positive, Negative, etc.)
    batched_t = batched_t.repeat(num_tasks, 1)
    
    return batched_t


def batch_compatible_conds(active_batched_conds) -> List[ExecutionBatch]:
    """
    Group conds to run them as a single batch instead of individually. This grouping should
    be according to the memory available and if the conds are compatible with each other or not.
    """
    flat_conds = []
    group_map = []  # group name for each item in flat_conds
    
    # flat_conds - [t1, t2, t3, t4]
    # group_map  - [p,  p,  n,  n]
    for name, conds in active_batched_conds.get_groups_in_order():
        if not conds: continue
        for cond in conds:
            flat_conds.append(cond)
            group_map.append(name)
            
    if not flat_conds:
        return {}
    
    # batch inputs
    num_tasks = len(flat_conds)
    batched_inputs = collate_inputs([c.model_input for c in flat_conds])
    
    # batch standard args (repeat for each task)
    batched_x = x_in.repeat(num_tasks, *[1] * (x_in.ndim - 1))
    batched_t = prepare_per_frame_timestep(timestep, num_tasks, denoise_mask)
    

def accumulate_output(
    out_acc: Tensor, 
    weight_acc: Tensor, 
    current_output: Tensor, 
    execution_batch: ExecutionBatch
    ):
    """
    Accumulates the output per batch, applies mask and then takes their weighted average
    """
    
    if len(execution_batch.conds):
        current_output = current_output.chunk(len(execution_batch.conds))
        for pred, cond, group_name in zip(current_output, execution_batch.conds, execution_batch.group_names):
            T, _, H, W = cond.shape
            weight = cond.get_combined_mask(T) * cond.strength 
            out_acc[group_name] += pred * weight
            weight_acc[group_name] += weight
            
    return out_acc, weight_acc