import torch

from .scheduler_types import calculate_sigmas
from .enum import DISCARD_PENULTIMATE_SIGMA_SAMPLERS, KSamplerType, SamplerType, SchedulerType
from .sampler_impl import ksampler_factory
from ...utils.tensor import fix_empty_latent_channels, prepare_noise
from ...model_utils.model_wrapper import ModelWrapper
from ...model_utils.latent import Latent

class KSampler:
    def __init__(
        self,
        wrapped_model: ModelWrapper,
        seed: int,
        steps: int,
        cfg: int,
        sampler_name: str,
        scheduler_name: str,
        positive: torch.Tensor,
        negative: torch.Tensor,
        latent_image: Latent,
    ):
        assert sampler_name in KSamplerType.value_list() + SamplerType.value_list(), f"sampler {sampler_name} not supported"
        assert scheduler_name in SchedulerType.value_list(), f"scheduler {scheduler_name} not supported"
        
        self.wrapped_model = wrapped_model
        self.seed = seed
        self.steps = steps
        self.cfg = cfg 
        self.sampler_name = sampler_name
        self.scheduler_name = scheduler_name
        self.positive = positive
        self.negative = negative
        self.latent_image = latent_image
        
    def calculate_sigmas(self):
        steps = self.steps
        if discard_penultimate:=(self.sampler_name in DISCARD_PENULTIMATE_SIGMA_SAMPLERS):
            steps += 1
        
        sigmas = calculate_sigmas(self.wrapped_model.model_sampling, self.scheduler, steps)
        if discard_penultimate: sigmas = torch.cat([sigmas[:-2], sigmas[-1:]])
        return sigmas
    
    def set_steps(self, steps, denoise=None):
        self.steps = steps
        if denoise is None or denoise > 0.9999:
            self.sigmas = self.calculate_sigmas(steps).to(self.device)
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
            noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
        else:
            batch_inds = self.latent_image.batch_index
            noise = prepare_noise(latent_image, self.seed, batch_inds)

        # TODO: enable calculation of new sigmas as per vars injected at the runtime
        # such as last_step, start_step
        
        sampler = ksampler_factory(self.sampler_name)
        return sample(
            self.wrapped_model,
            noise,
            self.positive,
            self.negative,
            self.cfg,
            sampler,
            self.model_options,
            latent_image=latent_image,
            denoise_mask=self.latent_image.noise_mask,
            callback=lambda *args, **kwargs: None,      # TODO: pass a null fn from the top
            seed=self.seed
        )
        
def sample(*args, **kwargs):
    # TODO: complete this
    pass

'''
denoise_mask,
callback,
disable_pbar,
seed
'''