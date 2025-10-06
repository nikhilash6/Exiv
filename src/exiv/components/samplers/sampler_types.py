import torch

import math

from .utils import make_beta_schedule
from ..enum import BetaSchedule, ModelType

# code adapted from ForgeUI and ComfyUI

# input, output and noise scaling to support k-diffusion based samplers
# check the EDM paper (https://arxiv.org/pdf/2206.00364) Table 1 for input and output scaling
# models predicting epsilon / noise
class EPS:
    def calculate_input(self, sigma, noise):
        sigma = sigma.view(sigma.shape[:1] + (1,) * (noise.ndim - 1))
        return noise / (sigma ** 2 + self.sigma_data ** 2) ** 0.5

    def calculate_denoised(self, sigma, model_output, model_input):
        sigma = sigma.view(sigma.shape[:1] + (1,) * (model_output.ndim - 1))
        return model_input - model_output * sigma

    def noise_scaling(self, sigma, noise, latent_image, max_denoise=False):
        if max_denoise:
            noise = noise * torch.sqrt(1.0 + sigma ** 2.0)
        else:
            noise = noise * sigma

        noise += latent_image
        return noise

    def inverse_noise_scaling(self, sigma, latent):
        return latent

# scaling the model ouput of these as per the EDM paper
# this same class is used for EDM and non-EDM models (empirical observation)
class V_PREDICTION(EPS):
    def calculate_denoised(self, sigma, model_output, model_input):
        sigma = sigma.view(sigma.shape[:1] + (1,) * (model_output.ndim - 1))
        return model_input * self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2) - model_output * sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2) ** 0.5

class EDM(V_PREDICTION):
    def calculate_denoised(self, sigma, model_output, model_input):
        sigma = sigma.view(sigma.shape[:1] + (1,) * (model_output.ndim - 1))
        return model_input * self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2) + model_output * sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2) ** 0.5

# constant noise (no input/output scaling)
class CONST:
    def calculate_input(self, sigma, noise):
        return noise

    def calculate_denoised(self, sigma, model_output, model_input):
        sigma = sigma.view(sigma.shape[:1] + (1,) * (model_output.ndim - 1))
        return model_input - model_output * sigma

    def noise_scaling(self, sigma, noise, latent_image, max_denoise=False):
        return sigma * noise + (1.0 - sigma) * latent_image

    def inverse_noise_scaling(self, sigma, latent):
        return latent / (1.0 - sigma)


# Methods to calculate sigma values for different model configs and training types,
# these also map sigmas to their respective timesteps for easy lookup.

# model trained on discrete timesteps
class ModelSamplingDiscrete(torch.nn.Module):
    def __init__(self, model_config={}):
        super().__init__()

        sampling_settings = model_config.get("sampling_settings", {})

        beta_schedule_type = sampling_settings.get("beta_schedule", BetaSchedule.LINEAR.value)
        linear_start = sampling_settings.get("linear_start", 0.00085)
        linear_end = sampling_settings.get("linear_end", 0.012)
        timesteps = sampling_settings.get("timesteps", 1000)

        self._register_schedule(beta_schedule_type=beta_schedule_type, timesteps=timesteps, \
            linear_start=linear_start, linear_end=linear_end, cosine_s=8e-3)
        self.sigma_data = 1.0

    def _register_schedule(self, beta_schedule_type=BetaSchedule.LINEAR.value, timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        # traditional closed form DDPM formulation
        betas = make_beta_schedule(beta_schedule_type, timesteps, linear_start=linear_start, \
            linear_end=linear_end, cosine_s=cosine_s)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        sigmas = ((1 - alphas_cumprod) / alphas_cumprod) ** 0.5
        self.set_sigmas(sigmas)

    # can be used to override sigmas at runtime
    def set_sigmas(self, sigmas):
        self.register_buffer('sigmas', sigmas.float())
        self.register_buffer('log_sigmas', sigmas.log().float())

    @property
    def sigma_min(self):
        return self.sigmas[0]

    @property
    def sigma_max(self):
        return self.sigmas[-1]

    def timestep(self, sigma):
        log_sigma = sigma.log()
        dists = log_sigma.to(self.log_sigmas.device) - self.log_sigmas[:, None]
        return dists.abs().argmin(dim=0).view(sigma.shape).to(sigma.device)

    def sigma(self, timestep):
        t = torch.clamp(timestep.float().to(self.log_sigmas.device), min=0, max=(len(self.sigmas) - 1))
        low_idx = t.floor().long()
        high_idx = t.ceil().long()
        w = t.frac()
        log_sigma = (1 - w) * self.log_sigmas[low_idx] + w * self.log_sigmas[high_idx]
        return log_sigma.exp().to(timestep.device)

    def percent_to_sigma(self, percent):
        if percent <= 0.0:
            return 999999999.9
        if percent >= 1.0:
            return 0.0
        percent = 1.0 - percent
        return self.sigma(torch.tensor(percent * 999.0)).item()

# trained on discrete steps + EDM methodology (karras noise schedule, input/output scaling etc..)
class ModelSamplingDiscreteEDM(ModelSamplingDiscrete):
    def timestep(self, sigma):
        return 0.25 * sigma.log()

    def sigma(self, timestep):
        return (timestep / 0.25).exp()

# trained to predict continuous noise values (EDM)
class ModelSamplingContinuousEDM(torch.nn.Module):
    def __init__(self, model_config={}):
        super().__init__()
        
        sampling_settings = model_config.get("sampling_settings", {})

        sigma_min = sampling_settings.get("sigma_min", 0.002)
        sigma_max = sampling_settings.get("sigma_max", 120.0)
        sigma_data = sampling_settings.get("sigma_data", 1.0)
        self.set_parameters(sigma_min, sigma_max, sigma_data)

    def set_parameters(self, sigma_min, sigma_max, sigma_data):
        self.sigma_data = sigma_data
        sigmas = torch.linspace(math.log(sigma_min), math.log(sigma_max), 1000).exp()

        self.register_buffer('sigmas', sigmas) #for compatibility with some schedulers
        self.register_buffer('log_sigmas', sigmas.log())

    @property
    def sigma_min(self):
        return self.sigmas[0]

    @property
    def sigma_max(self):
        return self.sigmas[-1]

    def timestep(self, sigma):
        return 0.25 * sigma.log()

    def sigma(self, timestep):
        return (timestep / 0.25).exp()

    def percent_to_sigma(self, percent):
        if percent <= 0.0:
            return 999999999.9
        if percent >= 1.0:
            return 0.0
        percent = 1.0 - percent

        log_sigma_min = math.log(self.sigma_min)
        return math.exp((math.log(self.sigma_max) - log_sigma_min) * percent + log_sigma_min)

# trained on continuous timesteps (tied to noise scale internally) with velocity training objective
# paper - https://arxiv.org/pdf/2202.00512 , appendix D
class ModelSamplingContinuousV(ModelSamplingContinuousEDM):
    def timestep(self, sigma):
        return sigma.atan() / math.pi * 2

    def sigma(self, timestep):
        return (timestep * math.pi / 2).tan()

def time_snr_shift(alpha, t):
    if alpha == 1.0:
        return t
    return alpha * t / (1 + (alpha - 1) * t)

class ModelSamplingDiscreteFlow(torch.nn.Module):
    def __init__(self, model_config={}):
        super().__init__()
        
        sampling_settings = model_config.get("sampling_settings", {})

        self.set_parameters(shift=sampling_settings.get("shift", 1.0), multiplier=sampling_settings.get("multiplier", 1000))

    def set_parameters(self, shift=1.0, timesteps=1000, multiplier=1000):
        self.shift = shift
        self.multiplier = multiplier
        ts = self.sigma((torch.arange(1, timesteps + 1, 1) / timesteps) * multiplier)
        self.register_buffer('sigmas', ts)

    @property
    def sigma_min(self):
        return self.sigmas[0]

    @property
    def sigma_max(self):
        return self.sigmas[-1]

    def timestep(self, sigma):
        return sigma * self.multiplier

    def sigma(self, timestep):
        return time_snr_shift(self.shift, timestep / self.multiplier)

    def percent_to_sigma(self, percent):
        if percent <= 0.0:
            return 1.0
        if percent >= 1.0:
            return 0.0
        return 1.0 - percent

def flux_time_shift(mu: float, sigma: float, t):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)

class ModelSamplingFlux(torch.nn.Module):
    def __init__(self, model_config={}):
        super().__init__()
        
        sampling_settings = model_config.get("sampling_settings", {})

        self.set_parameters(shift=sampling_settings.get("shift", 1.15))

    def set_parameters(self, shift=1.15, timesteps=10000):
        self.shift = shift
        ts = self.sigma((torch.arange(1, timesteps + 1, 1) / timesteps))
        self.register_buffer('sigmas', ts)

    @property
    def sigma_min(self):
        return self.sigmas[0]

    @property
    def sigma_max(self):
        return self.sigmas[-1]

    def timestep(self, sigma):
        return sigma

    def sigma(self, timestep):
        return flux_time_shift(self.shift, 1.0, timestep)

    def percent_to_sigma(self, percent):
        if percent <= 0.0:
            return 1.0
        if percent >= 1.0:
            return 0.0
        return 1.0 - percent


MODEL_REGISTRY = {
    ModelType.EPS: (ModelSamplingDiscrete, EPS),
    ModelType.V_PREDICTION: (ModelSamplingDiscrete, V_PREDICTION),
    ModelType.V_PREDICTION_EDM: (ModelSamplingContinuousEDM, V_PREDICTION),
    ModelType.V_PREDICTION_CONTINUOUS: (ModelSamplingContinuousV, V_PREDICTION),
    ModelType.FLOW: (ModelSamplingDiscreteFlow, CONST),
    ModelType.EDM: (ModelSamplingContinuousEDM, EDM),
    ModelType.FLUX: (ModelSamplingFlux, CONST),
}

def model_sampling(model_config, model_type):
    sampling_class, scaling_class = MODEL_REGISTRY.get(model_type)
    if not sampling_class:
        raise ValueError(f"Unknown model_type: {model_type}")

    ModelSampling = type("ModelSampling", (sampling_class, scaling_class), {})
    return ModelSampling(model_config)
