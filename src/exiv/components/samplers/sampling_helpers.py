import torch
from torch import Tensor

import dataclasses
from typing import Any, Callable, Dict, List, Optional, Tuple

from ...utils.device import RESERVED_MEM, MemoryManager
from ...utils.logging import app_logger
from ...model_patching.common import profile_model_layers
from ...model_utils.helper_methods import estimate_peak_activation_size
from ...model_utils.common_classes import BatchedConditioning, Conditioning, ExecutionBatch, ModelWrapper


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

    active_list = []
    for group_name, cond_list in batched_conditioning.get_groups_in_order():
        for cond in cond_list:
            filter_out = False
            
            # timestep filtering
            start, end = cond.timestep_range
            if end == -1:
                if timestep < start:
                    filter_out = True
            else:
                if timestep < start or timestep > end:
                    filter_out = True
                    
            # if per timestep strength is provided
            if cond.strength == 0:
                filter_out = True
            elif not filter_out and isinstance(cond.strength, list):
                try:
                    cur_timestep_strength = cond.strength[timestep - cond.timestep_range[0]]
                    if cur_timestep_strength == 0: filter_out = True
                except Exception as e:
                    app_logger.warning(f"Unable to calculate per step strength {e}")
                    
            if not filter_out: active_list.append(cond)
        
    filtered.set_cond(active_list, reset=True)
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
    
    updated_conds = []
    for cond_group_name, cond_list in batched_conditioning.get_groups_in_order():
        # cond_list = wrapped_model.model.filter_conditionings(cond_list)
        # if cond_list is None: continue
        for cond in cond_list:
            # shallow copy to avoid stale data in case of accidental re-use
            active_cond = dataclasses.replace(cond)
            active_cond.model_input = model.prepare_model_input(active_cond, **base_ctx)
            updated_conds.append(active_cond)
            
    res.set_cond(updated_conds, reset=True)
    return res

# TODO: move this in the tensor file
def get_structure_size_mb(data) -> float:
    """Recursively calculates the memory size of tensors in a structure (MB)."""
    if isinstance(data, torch.Tensor):
        return (data.numel() * data.element_size()) / (1024 ** 2)
    elif isinstance(data, dict):
        return sum(get_structure_size_mb(v) for v in data.values())
    elif isinstance(data, (list, tuple)):
        return sum(get_structure_size_mb(v) for v in data)
    return 0.0

def break_cond_for_no_oom(cond: Conditioning, x_in: Tensor):
    """
    Breaks the Condition into smaller Conditions that run individually and don't cause OOMs
    """
    # TODO: not implemented yet
    return [cond]

def batch_compatible_conds(
    active_batched_conds: BatchedConditioning,
    x_in: Tensor,
    timestep: Tensor,
    max_batch_size: int = 1,
) -> List[ExecutionBatch]:

    # flatten the queue
    work_queue: List[Tuple[str, Conditioning]] = []
    for group_name, cond_list in active_batched_conds.get_groups_in_order():
        if not cond_list: continue
        for cond in cond_list:
            broken_conds = break_cond_for_no_oom(cond, x_in)
            for bc in broken_conds: work_queue.append((group_name, bc))

    # greedy approach, trying to match pairs that are compatible and OOM safe
    execution_batches: List[ExecutionBatch] = []
    is_consumed: List[bool] = [False] * len(work_queue)
    for idx, (g_name, cur_cond) in enumerate(work_queue):
        if is_consumed[idx]: continue
        is_consumed[idx] = True
        cur_execution_batch = ExecutionBatch(
            feed_x=x_in, 
            feed_t=timestep, 
            feed_input=cur_cond.model_input.to_dict() if cur_cond.model_input else {}
        )
        cur_execution_batch.add_cond(cur_cond)
        
        if idx != len(work_queue) - 1:
            for i, (g, c) in enumerate(work_queue[idx+1:]):
                # signature matches and the combination is memory safe, then batch them
                if cur_cond.signature == c.signature and \
                    len(cur_execution_batch.conds) + 1 <= max_batch_size:
                    cur_execution_batch.add_cond(c)
                    is_consumed[idx + 1 + i] = True
        
        execution_batches.append(cur_execution_batch)

    return execution_batches
    
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
            b, c, f, h, w = current_output[0].shape         # TODO: make sure this logic is generic enough for future models
            weight = cond.get_combined_mask(f) * cond.strength
            out_acc[group_name] += pred * weight
            weight_acc[group_name] += weight
            
    return out_acc, weight_acc

def get_max_bs_dict(model, shape):
    if (cached_bs_dict:=getattr(model, 'cached_bs_dict', None)) is not None:
        if shape in cached_bs_dict: return cached_bs_dict[shape]
    return None

def set_max_bs_dict(model, shape, max_bs):
    cached_bs_dict = getattr(model, 'cached_bs_dict', None)
    if cached_bs_dict is None: cached_bs_dict = {}
    cached_bs_dict[shape] = max_bs
    setattr(model, 'cached_bs_dict', cached_bs_dict)

def determine_max_batch_size(model: 'ModelMixin', target_shape):
    """
    Calculates the maximum safe batch size based on VRAM availability
    Philosophy: 'weights stay', try to pack as much weights on the VRAM as possible
    """
    single_item_shape = target_shape[1:]
    max_bs = get_max_bs_dict(model, single_item_shape)
    if max_bs is not None: return max_bs
    
    available_mem = MemoryManager.available_memory(model.gpu_device) - RESERVED_MEM
    batch_1_shape = (1, *single_item_shape)
    act_cost_b1 = estimate_peak_activation_size(model.get_memory_footprint_params(), batch_1_shape)
    
    module_by_size, total_model_size = profile_model_layers(model)
    # if batch 1 activations + largest layer (for swapping) don't fit, we may get OOM
    min_required = act_cost_b1 + (module_by_size[-1][0] if total_model_size > available_mem else 0)
    if min_required > available_mem:
        app_logger.warning("Potential OOM: VRAM insufficient for even Batch Size 1")
        max_bs = 1

    # if the WHOLE model fits, we check how many more batches we can fit (nomral mode)
    if (total_model_size + act_cost_b1) <= available_mem:
        remaining_mem = available_mem - total_model_size
        max_batch = 1
        while True:     # iteratively find max batch size
            next_shape = (max_batch + 1, *single_item_shape)
            next_cost = estimate_peak_activation_size(model.get_memory_footprint_params(), next_shape)
            
            if next_cost > remaining_mem:
                break
            max_batch += 1
        max_bs = max_batch

    # nodel too big (low vram mode)
    else:
        max_bs = 1
        
    set_max_bs_dict(model, single_item_shape, max_bs)
    return max_bs
        
    