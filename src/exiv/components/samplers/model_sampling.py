import torch
from torch import Tensor

from typing import Dict, List

from .utils import normalize_seed
from .scheduler_types import calculate_sigmas
from .sampling_helpers import filter_active_conds, prepare_model_conds, prepare_mask
from ..enum import DISCARD_PENULTIMATE_SIGMA_SAMPLERS, KSamplerType, SamplerType, SchedulerType
from .sampler_impl import Sampler, ksampler_factory
from ...utils.tensor import fix_empty_latent_channels, prepare_noise
from ...model_utils.common_classes import BatchedConditioning, ModelForwardInput, ModelWrapper, Latent
from ...utils.device import OFFLOAD_DEVICE, VRAM_DEVICE, ProcDevice
from ...utils.common import null_func

class KSampler:
    def __init__(
        self,
        wrapped_model: ModelWrapper,
        seed: int,
        steps: int,
        cfg: float,
        sampler_name: str,
        scheduler_name: str,
        batched_conditioning: BatchedConditioning,
        latent_image: Latent,
        denoise: float = 1.0,
        device = None,
    ):
        assert sampler_name in KSamplerType.value_list() + SamplerType.value_list(), f"sampler {sampler_name} not supported"
        assert scheduler_name in SchedulerType.value_list(), f"scheduler {scheduler_name} not supported"
        assert denoise >= 0.0 and denoise <= 1.0, f"denoise {denoise} out of range, should be between 0 and 1"
        
        self.device = device or VRAM_DEVICE
        
        self.wrapped_model = wrapped_model
        self.seed = normalize_seed(seed)
        self.steps = steps
        self.cfg = cfg 
        self.sampler_name = sampler_name
        self.scheduler_name = scheduler_name
        self.batched_conditioning = batched_conditioning
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
    def run_sampling(self, disable_noise = False, callback=null_func):
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
    
    if denoise_mask is not None:
        denoise_mask = prepare_mask(denoise_mask, noise.shape, wrapped_model.model.gpu_device)
    
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
    
    # used in filtering time based conditioning
    sigma_start = sigmas[0].item()
    sigma_end = sigmas[-1].item()
    total_range = sigma_start - sigma_end + 1e-5
    
    extra_args = {"seed":seed}
    
    def denoiser_function(x, sigma, **kwargs):
        # progres -> (0.0 -> 1.0)
        current_sigma = sigma.item()
        progress = (sigma_start - current_sigma) / total_range
        progress = max(0.0, min(1.0, progress))
        
        active_batch = filter_active_conds(mod_batched_conds, progress)
        
        return model_sampling_step(
            wrapped_model,
            x,
            sigma,
            active_batch,
            cfg,
            denoise_mask=denoise_mask,
            seed=kwargs.get("seed")
        )
    
    samples = ksampler_cls_impl.sample(denoiser_function, wrapped_model, sigmas, extra_args, callback, noise, latent_image, denoise_mask)
    return wrapped_model.model.process_latent_out(samples.to(torch.float32))


def model_sampling_step(
    wrapped_model: ModelWrapper, 
    x: Tensor, 
    sigma: Tensor, 
    active_batch: BatchedConditioning, 
    cond_scale: float, 
    denoise_mask=None, 
    seed=None
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
    out_groups = calc_cond_batch(wrapped_model, active_batch, x_in, timestep, denoise_mask=denoise_mask)
    
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

def calc_cond_batch(
    wrapped_model: ModelWrapper, 
    batched_conds: BatchedConditioning, 
    x_in: Tensor, 
    timestep: Tensor, 
    denoise_mask=None
) -> Dict[str, Tensor]:
    """
    It batches all conditioning (pos, neg, others etc..) together, runs the
    model once, and then returns the separated results. (for now)
    """

    flat_conds = []
    group_map = []  # group name for each item in flat_conds
    
    # flat_conds - [t1, t2, t3, t4]
    # group_map  - [p,  p,  n,  n]
    for name, conds in batched_conds.get_groups_in_order():
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

    # **** main model run ****
    output = wrapped_model.model(batched_x, batched_t, **batched_inputs)
    
    # un-batch and average per group
    output_chunks = output.chunk(num_tasks)
    
    results = {}
    counts = {}
    
    for i, (chunk, group_name) in enumerate(zip(output_chunks, group_map)):
        cond = flat_conds[i]
        if group_name not in results:
            results[group_name] = torch.zeros_like(x_in)
            counts[group_name] = torch.zeros_like(x_in) + 1e-37
            
        results[group_name] += chunk * cond.strength
        counts[group_name] += cond.strength

    for name in results:
        results[name] /= counts[name]
        
    return results