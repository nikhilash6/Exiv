import torch
from torch import Tensor

from functools import partial
from typing import Dict, List

from .utils import normalize_seed
from .scheduler_types import calculate_sigmas
from .sampling_helpers import accumulate_output, batch_compatible_conds, determine_max_batch_size, filter_active_conds, prepare_model_conds
from ..enum import DISCARD_PENULTIMATE_SIGMA_SAMPLERS, KSamplerType, SamplerType, SchedulerType
from .sampler_impl import Sampler, ksampler_factory
from ...utils.tensor import fix_empty_latent_channels, prepare_noise
from ...model_utils.common_classes import BatchedConditioning, ExecutionBatch, ModelForwardInput, ModelWrapper, Latent
from ...model_utils.conditioning_mixin import ConditioningMixin
from ...utils.device import OFFLOAD_DEVICE, VRAM_DEVICE, ProcDevice
from ...utils.common import null_func
from ...utils.logging import app_logger
from ...model_patching.hook_registry import HookLocation, HookRegistry, HookType
from ...model_utils.helper_methods import get_mem_usage

class KSampler:
    def __init__(
        self,
        wrapped_model: ModelWrapper,
        seed: int,
        cfg: float,
        sampler_name: str,
        scheduler_name: str,
        batched_conditioning: BatchedConditioning,
        latent_image: Latent,
        end_step: int = None,
        start_step: int = 0,
        total_steps: int = None,
        denoise: float = 1.0,
        device = None,
    ):
        assert sampler_name in KSamplerType.value_list() + SamplerType.value_list(), f"sampler {sampler_name} not supported"
        assert scheduler_name in SchedulerType.value_list(), f"scheduler {scheduler_name} not supported"
        assert denoise >= 0.0 and denoise <= 1.0, f"denoise {denoise} out of range, should be between 0 and 1"
        assert total_steps, f"Invalid total_steps: {total_steps}"
        if end_step is None: end_step = total_steps
        assert 0 <= start_step < end_step <= total_steps, f"Invalid step range: {start_step} to {end_step} (Total: {total_steps})"
        
        self.device = device or VRAM_DEVICE
        self.wrapped_model = wrapped_model
        self.seed = normalize_seed(seed)
        self.start_step, self.end_step = start_step, end_step
        self.cfg = cfg 
        self.sampler_name = sampler_name
        self.scheduler_name = scheduler_name
        self.batched_conditioning = batched_conditioning
        self.latent_image = latent_image
        self.denoise = denoise
        
        self.set_steps(total_steps, denoise)
    
    # or calculate_schedule
    def calculate_sigmas(self):
        steps = self.total_steps
        if discard_penultimate:=(self.sampler_name in DISCARD_PENULTIMATE_SIGMA_SAMPLERS):
            steps += 1
        
        sigmas = calculate_sigmas(self.wrapped_model.model_sampling, self.scheduler_name, steps)
        if discard_penultimate: sigmas = torch.cat([sigmas[:-2], sigmas[-1:]])
        return sigmas
    
    def set_steps(self, steps, denoise=None):
        self.total_steps = steps
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
    def run_sampling(self, disable_noise = False, callback=null_func):
        latent_image = self.latent_image.samples
        latent_image = fix_empty_latent_channels(self.wrapped_model, latent_image)
        
        # decides between random noise or zero noise tensors
        if disable_noise:
            noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device=OFFLOAD_DEVICE)
        else:
            batch_inds = self.latent_image.batch_index
            noise = prepare_noise(latent_image, self.seed, batch_inds)

        if self.start_step != 0: self.sigmas = self.sigmas[self.start_step:]
        if self.end_step != self.total_steps: self.sigmas = self.sigmas[:self.end_step + 1]
        
        # main sampling loop
        ksampler_cls_impl = ksampler_factory(self.sampler_name)
        return sample(
            self.wrapped_model,
            noise,
            self.batched_conditioning,
            self.cfg,
            self.sigmas,
            ksampler_cls_impl,
            latent_image=latent_image,
            denoise_mask=self.latent_image.noise_mask,
            callback=callback,
            seed=self.seed,
        )


def sample(
    wrapped_model: ModelWrapper,
    noise: Tensor,
    batched_conditioning: BatchedConditioning,
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
    - constructs the model_sampling_step that is ultimately passed in the sampler's sample method
    - sampler's sample returns the output after running the sampling loop
    '''
    # returning if there are 0 steps
    if sigmas.shape[-1] == 0: return latent_image
    
    # not scaling the blank latents
    if latent_image is not None and torch.count_nonzero(latent_image) > 0:
        latent_image = wrapped_model.model.process_latent_in(latent_image)
    
    # modified batched conditioning
    mod_batched_conds: BatchedConditioning = prepare_model_conds(
        wrapped_model, 
        batched_conditioning, 
        noise, 
        latent_image, 
        denoise_mask, 
        seed
    )
    
    # injecting sampler hook
    registry = HookRegistry.get_hook_registry(wrapped_model.model)
    wrapped_call = registry.get_wrapped_fn(
        model_sampling_step,
        location=HookLocation.SAMPLER_STEP.value,
        hook_order=[HookType.TAYLOR_SEER_LITE_MODEL_HOOK.value],
    )
    
    def denoiser_function(x, sigma, **kwargs):
        return wrapped_call(
            wrapped_model,
            x,
            sigma,
            mod_batched_conds,
            cfg,
            denoise_mask=denoise_mask,
        )
    
    samples = ksampler_cls_impl.sample(denoiser_function, wrapped_model, sigmas, callback, noise, latent_image, denoise_mask)
    return wrapped_model.model.process_latent_out(samples.to(torch.float32))


def model_sampling_step(
    wrapped_model: ModelWrapper, 
    x: Tensor, 
    sigma: Tensor, 
    batched_conds: BatchedConditioning, 
    cond_scale: float, 
    denoise_mask=None,
):
    '''
    Single sampling step for a given model. 
    - scales input using calculate_input
    - runs the model
    - applies CFG
    - calculates denoised output (x0) using calculate_denoised
    '''
    
    # convert sigma (noise level) to the discrete timestep expected by the model
    timestep = wrapped_model.model_sampling.timestep(sigma)
    # x is the current noisy latent
    x_in = wrapped_model.model_sampling.calculate_input(sigma, x)

    # **** main model run ****
    out_groups = compute_batched_output(wrapped_model, batched_conds, x_in, timestep, denoise_mask=denoise_mask)
    
    # TODO: streamline this as more cfg methods are added
    # (defaulting to zeros if the group is missing / filtered)
    cond_pred = out_groups.get("positive", torch.zeros_like(x_in))
    uncond_pred = out_groups.get("negative", torch.zeros_like(x_in))
    
    kwargs = {
        "cond": cond_pred, 
        "uncond": uncond_pred, 
        "cond_scale": cond_scale, 
        "timestep": timestep, 
        "input": x, 
        "sigma": timestep,
        "cond_denoised": cond_pred, 
        "uncond_denoised": uncond_pred, 
        "model": wrapped_model
    }
    # we can apply cfg on raw outputs, no matter what they are 
    cfg_result = wrapped_model.cfg_func(**kwargs)
    # convert the model output (EPS, V, etc.) back to the denoised latent (x0)
    denoised = wrapped_model.model_sampling.calculate_denoised(sigma, cfg_result, x)

    return denoised


def compute_batched_output(
    wrapped_model: ModelWrapper, 
    batched_conds: BatchedConditioning, 
    x_in: Tensor, 
    timestep: Tensor, 
    denoise_mask=None
) -> Dict[str, Tensor]:
    """
    It batches all conditioning (pos, neg, others etc..) together, runs the
    model once, and then returns the separated results.
    """

    active_batched_conds = filter_active_conds(batched_conds, timestep)
    registry = HookRegistry.get_hook_registry(wrapped_model.model)
    max_bs = determine_max_batch_size(wrapped_model.model, x_in.shape)
    execution_batch_list: List[ExecutionBatch] = batch_compatible_conds(
        active_batched_conds, 
        x_in, 
        timestep,
        max_bs
    )

    # **** main model run ****
    out_acc = {k: torch.zeros_like(x_in) for k, _ in active_batched_conds.get_groups_in_order()}
    weights_acc = {k: torch.zeros_like(x_in) for k, _ in active_batched_conds.get_groups_in_order()}
    for execution_batch in execution_batch_list:
        execution_batch.expand_batched_values(timestep, denoise_mask)    # this saves us vram 
        app_logger.debug(f"Batch size this step: {len(execution_batch.conds)}")
        
        # sampler hooks are added here
        deferred_model_run = partial(run_model, wrapped_model.model)
        if registry and registry.head.next_hook != registry.tail:
            wrapped_call = registry.get_wrapped_fn(
                deferred_model_run,
                location=HookLocation.INNER_SAMPLER_STEP.value,
                hook_order=[HookType.SLIDING_CONTEXT.value],
            )
            output = wrapped_call(execution_batch.feed_x, execution_batch.feed_t, **execution_batch.feed_input)
        else:
            output = deferred_model_run(execution_batch.feed_x, execution_batch.feed_t, **execution_batch.feed_input)

        output = output.to(VRAM_DEVICE)
        out_acc, weights_acc = accumulate_output(out_acc, weights_acc, output, execution_batch)
    
    # average the accumulated outputs
    final_output = {}
    for k in out_acc:
        final_output[k] = out_acc[k] / (weights_acc[k] + 1e-5)
    
    return final_output

# NOTE: separated for debugging / testing purposes
def run_model(model, feed_x, feed_t, **feed_input):
    # from torch_tracer import TorchTracer
    # with TorchTracer("./exiv_2.pkl"):
    out = model(feed_x, feed_t, **feed_input)
    return out