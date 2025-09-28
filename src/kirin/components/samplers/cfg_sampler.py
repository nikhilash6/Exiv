import torch

from ...constants import shared_state
from ...utils.exceptions import ProcessInterrupted

# TODO: complete this 
class CFGDenoiser(torch.nn.Module):
    def __init__(self, sampler_func):
        super().__init__()
        
        self.model_wrap = None
        self.init_latent = None
        self.steps = None
        self.total_steps = None
        
        self.step = 0
        self.sampler_func = sampler_func
        
    def forward(self, x, sigma, uncond, cond, cond_scale, s_min_uncond, image_cond):
        if shared_state.stop_generation:
            raise ProcessInterrupted()
        
        # check the EDM paper (https://arxiv.org/pdf/2206.00364) Table 1 for input scaling
        # this also calculates sigmas as per k-diffusion schedule for compvis type schedules
        if self.compvis_model:
            acd = self.base_model.alphas_cumprod
            fake_sigmas = ((1 - acd) / acd) ** 0.5
            real_sigma = fake_sigmas[sigma.round().long().clip(0, int(fake_sigmas.shape[0]))]
            real_sigma_data = 1.0
            x = x * (((real_sigma ** 2.0 + real_sigma_data ** 2.0) ** 0.5)[:, None, None, None])
            sigma = real_sigma
            
        