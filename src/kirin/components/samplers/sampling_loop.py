import torch

from .enum import DISCARD_PENULTIMATE_SIGMA_SAMPLERS, KSamplerType, SamplerType, SchedulerType
from ...utils.tensor import fix_empty_latent_channels, prepare_noise

class KSampler:
    def __init__(
        self,
        model_wrapper: ModelWrapper,
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
        
        self.model_wrapper = model_wrapper
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
        
        sigmas = calculate_sigmas(self.model.get_model_object("model_sampling"), self.scheduler, steps)
        
        if discard_penultimate:
            sigmas = torch.cat([sigmas[:-2], sigmas[-1:]])
        return sigmas
        
    def run_sampling(self, disable_noise = False):
        latent_image = self.latent_image.samples
        latent_image = fix_empty_latent_channels(model_wrapper, latent_image)
        
        # decides between random noise or zero noise tensors
        if disable_noise:
            noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
        else:
            batch_inds = latent["batch_index"] if "batch_index" in latent else None
            noise = prepare_noise(latent_image, self.seed, batch_inds)

        # TODO: complete this 
        
        