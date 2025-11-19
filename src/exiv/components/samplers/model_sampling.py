from typing import List
import torch
from torch import Tensor

import math

from .scheduler_types import calculate_sigmas
from .sampling_helpers import preprocess_cond, prepare_model_conds, prepare_mask
from ..enum import DISCARD_PENULTIMATE_SIGMA_SAMPLERS, KSamplerType, SamplerType, SchedulerType
from .sampler_impl import Sampler, ksampler_factory
from ..conditionals import can_concat_cond, cond_cat
from ...utils.tensor import fix_empty_latent_channels, prepare_noise
from ...model_utils.common_classes import ModelWrapper
from ...model_utils.common_classes import Latent
from ...utils.device import OFFLOAD_DEVICE, VRAM_DEVICE, ProcDevice

class KSampler:
    def __init__(
        self,
        wrapped_model: ModelWrapper,
        seed: int,
        steps: int,
        cfg: float,
        sampler_name: str,
        scheduler_name: str,
        positive: torch.Tensor,
        negative: torch.Tensor,
        latent_image: Latent,
        denoise: float = 1.0,
        device = None,
    ):
        assert sampler_name in KSamplerType.value_list() + SamplerType.value_list(), f"sampler {sampler_name} not supported"
        assert scheduler_name in SchedulerType.value_list(), f"scheduler {scheduler_name} not supported"
        assert denoise >= 0.0 and denoise <= 1.0, f"denoise {denoise} out of range, should be between 0 and 1"
        
        self.device = device or VRAM_DEVICE
        
        self.wrapped_model = wrapped_model
        self.seed = seed
        self.steps = steps
        self.cfg = cfg 
        self.sampler_name = sampler_name
        self.scheduler_name = scheduler_name
        self.positive = positive
        self.negative = negative
        self.latent_image = latent_image
        self.denoise = denoise
        
        self.set_steps(steps, denoise)
    
    # or calculate_schedule
    def calculate_sigmas(self):
        steps = self.steps
        if discard_penultimate:=(self.sampler_name in DISCARD_PENULTIMATE_SIGMA_SAMPLERS):
            steps += 1
        
        sigmas = calculate_sigmas(self.wrapped_model.model_sampling, self.scheduler_name, steps)
        if discard_penultimate: sigmas = torch.cat([sigmas[:-2], sigmas[-1:]])
        return sigmas
    
    def set_steps(self, steps, denoise=None):
        self.steps = steps
        if denoise is None or denoise > 0.9999:
            self.sigmas = self.calculate_sigmas().to(self.device)
        else:
            if denoise <= 0.0:
                self.sigmas = torch.FloatTensor([])
            else:
                new_steps = int(steps/denoise)
                sigmas = self.calculate_sigmas(new_steps).to(self.device)
                self.sigmas = sigmas[-(steps + 1):]
    
    # TODO: allow for dynamic injection / overriding of variables such as sigma
    def run_sampling(self, disable_noise = False):
        latent_image = self.latent_image.samples
        latent_image = fix_empty_latent_channels(self.wrapped_model, latent_image)
        
        # decides between random noise or zero noise tensors
        if disable_noise:
            noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device=OFFLOAD_DEVICE)
        else:
            batch_inds = self.latent_image.batch_index
            noise = prepare_noise(latent_image, self.seed, batch_inds)

        # TODO: enable calculation of new sigmas as per vars injected at the runtime
        # such as last_step, start_step
        
        ksampler_cls_impl = ksampler_factory(self.sampler_name)
        return sample(
            self.wrapped_model,
            noise,
            self.positive,
            self.negative,
            self.cfg,
            self.sigmas,
            ksampler_cls_impl,
            latent_image=latent_image,
            denoise_mask=self.latent_image.noise_mask,
            callback=lambda *args, **kwargs: None,      # TODO: pass a null fn from the top
            seed=self.seed,
        )


def sample(
    wrapped_model: ModelWrapper,
    noise: Tensor,
    positive,
    negative,
    cfg,
    sigmas,
    ksampler_cls_impl: Sampler,
    latent_image,
    denoise_mask,
    callback,
    seed,
) -> Tensor:
    '''
    - processes the inputs, masks and conditionals 
    - constructs the model_sampling that is ultimately passed in the sampler's sample method
    - sampler's sample returns the output after running the sampling loop
    '''
    # returning if there are 0 steps
    if sigmas.shape[-1] == 0: return latent_image
   
    grouped_cond = {"positive": positive, "negative": negative}
        
    if denoise_mask is not None:
        denoise_mask = prepare_mask(denoise_mask, noise.shape, wrapped_model.model.gpu_device)
    
    # not scaling the blank latents
    if latent_image is not None and torch.count_nonzero(latent_image) > 0:
        latent_image = wrapped_model.process_latent_in(latent_image)
        
    conds = prepare_model_conds(wrapped_model, grouped_cond, noise, latent_image, denoise_mask, seed)
    pos_conds, neg_conds = conds.get("positive"), conds.get("negative")
    
    extra_args = {"seed":seed}
    
    def denoiser_function(x, sigma, **kwargs):
        return model_sampling(
            wrapped_model,
            x,
            sigma,
            neg_conds,
            pos_conds,
            cfg,
            seed=kwargs.get("seed")
        )
    
    samples = ksampler_cls_impl.sample(denoiser_function, wrapped_model, sigmas, extra_args, callback, noise, latent_image, denoise_mask)
    return wrapped_model.process_latent_out(samples.to(torch.float32))


def model_sampling(wrapped_model: ModelWrapper, x, timestep, uncond, cond, cond_scale, seed=None):
    '''
    Single sampling step for a given model. This has to be used inside the sampler's sample methods.
    CFG and post-CFG methods (if any) are applied here.
    '''
    # TODO: apply pre and post cfg methods here when they are added
    if math.isclose(cond_scale, 1.0) and wrapped_model.disable_cfg:
        uncond_ = None
    else:
        uncond_ = uncond

    conds = [cond, uncond_]
    out = calc_cond_batch(wrapped_model, conds, x, timestep)

    cond_pred, uncond_pred = out[0], out[1]
    
    # ------ applying cfg ---------
    args = {
        "cond": x - cond_pred, 
        "uncond": x - uncond_pred, 
        "cond_scale": cond_scale, 
        "timestep": timestep, 
        "input": x, 
        "sigma": timestep,
        "cond_denoised": cond_pred, 
        "uncond_denoised": uncond_pred, 
        "model": wrapped_model
    }
    cfg_result = wrapped_model.cfg_fn(args)

    return cfg_result


def calc_cond_batch(wrapped_model: ModelWrapper, conds: List[List], x_in: Tensor, timestep: Tensor):
    """
    It batches all conditioning (pos, neg, others etc..) together, runs the
    model once, and then returns the separated results.
    """
    # nothing to process
    if not len(conds) or all(c is None for c in conds):
        return [torch.zeros_like(x_in) for _ in conds]

    # ---- prepare conditioning
    applied_cond = []
    applied_cond_group = []

    for i, cond_list in enumerate(conds):
        if cond_list is None: continue
        for c in cond_list:
            prepared_chunk = preprocess_cond(c, x_in)
            applied_cond.append(prepared_chunk)
            applied_cond_group.append(i)

    # returning if no conditional applied
    if not applied_cond:
        return [torch.zeros_like(x_in) for _ in range(len(conds))]

    # ---- build the batch
    num_tasks = len(applied_cond)
    batched_input_x = x_in.repeat(num_tasks, 1, 1, 1)
    batched_timestep = timestep.repeat(num_tasks)
    conditioning_to_cat = [t.conditioning for t in applied_cond]

    batched_conditioning = cond_cat(conditioning_to_cat)

    controlnet_cond = next((t.control for t in applied_cond if t.control is not None), None)
    if controlnet_cond is not None:
        batched_conditioning['control'] = controlnet_cond.get_control(batched_input_x, batched_timestep, batched_conditioning, num_tasks)

    # ---- run model
    output = wrapped_model.model(batched_input_x, batched_timestep, **batched_conditioning)

    # ---- average the results
    final_outputs = [torch.zeros_like(x_in) for _ in range(len(conds))]
    final_counts = [torch.zeros_like(x_in) + 1e-37 for _ in range(len(conds))]

    output_chunks = output.chunk(num_tasks)

    for i, result_chunk in enumerate(output_chunks):
        original_index = applied_cond_group[i]
        strength_mult = applied_cond[i].mult

        final_outputs[original_index] += result_chunk * strength_mult
        final_counts[original_index] += strength_mult

    for i in range(len(final_outputs)):
        final_outputs[i] /= final_counts[i]

    final_results = []
    output_idx = 0
    for c in conds:
        if c is None: final_results.append(torch.zeros_like(x_in))
        else:
            final_results.append(final_outputs[output_idx])
            output_idx += 1

    return final_results

