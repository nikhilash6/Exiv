import torch

import math
import numpy as np
import scipy

from .enum import SchedulerType
from ...utils.logging import app_logger
from ...utils.tensor import append_dims


# code adapter from ComfyUI

def calculate_sigmas(model_sampling, scheduler_name: str, steps: int):
    if scheduler_name == SchedulerType.KARRAS.value:
        sigmas = get_sigmas_karras(n=steps, sigma_min=float(model_sampling.sigma_min), sigma_max=float(model_sampling.sigma_max))
    elif scheduler_name == SchedulerType.EXPONENTIAL.value:
        sigmas = get_sigmas_exponential(n=steps, sigma_min=float(model_sampling.sigma_min), sigma_max=float(model_sampling.sigma_max))
    elif scheduler_name == SchedulerType.NORMAL.value:
        sigmas = normal_scheduler(model_sampling, steps)
    elif scheduler_name == SchedulerType.SIMPLE.value:
        sigmas = simple_scheduler(model_sampling, steps)
    elif scheduler_name == SchedulerType.DDIM_UNIFORM.value:
        sigmas = ddim_scheduler(model_sampling, steps)
    elif scheduler_name == SchedulerType.SGM_UNIFORM.value:
        sigmas = normal_scheduler(model_sampling, steps, sgm=True)
    elif scheduler_name == SchedulerType.BETA.value:
        sigmas = beta_scheduler(model_sampling, steps)
    else:
        app_logger.error("error invalid scheduler {}".format(scheduler_name))
    
    return sigmas


def append_zero(x):
    return torch.cat([x, x.new_zeros([1])])

# Implemented based on: https://arxiv.org/abs/2407.12173
def beta_scheduler(model_sampling, steps, alpha=0.6, beta=0.6):
    total_timesteps = (len(model_sampling.sigmas) - 1)
    ts = 1 - np.linspace(0, 1, steps, endpoint=False)
    ts = np.rint(scipy.stats.beta.ppf(ts, alpha, beta) * total_timesteps)

    sigs = []
    for t in ts:
        sigs += [float(model_sampling.sigmas[int(t)])]
    sigs += [0.0]
    return torch.FloatTensor(sigs)


def ddim_scheduler(model_sampling, steps):
    # creates a DDIM-like schedule by selecting every Nth sigma from the model's predefined list to match the desired number of steps
    s = model_sampling
    sigs = []
    x = 1
    if math.isclose(float(s.sigmas[x]), 0, abs_tol=0.00001):
        steps += 1
        sigs = []
    else:
        sigs = [0.0]

    ss = max(len(s.sigmas) // steps, 1)
    while x < len(s.sigmas):
        sigs += [float(s.sigmas[x])]
        x += ss
    sigs = sigs[::-1]
    return torch.FloatTensor(sigs)


def simple_scheduler(model_sampling, steps):
    # creates a schedule by picking sigmas at evenly spaced indices from the model's predefined list of sigmas
    s = model_sampling
    sigs = []
    ss = len(s.sigmas) / steps
    for x in range(steps):
        sigs += [float(s.sigmas[-(1 + int(x * ss))])]
    sigs += [0.0]
    return torch.FloatTensor(sigs)


def normal_scheduler(model_sampling, steps, sgm=False, floor=False):
    # interpolates in the linear timestep space and then converts back to sigmas
    s = model_sampling
    start = s.timestep(s.sigma_max)
    end = s.timestep(s.sigma_min)

    append_zero = True
    if sgm:
        timesteps = torch.linspace(start, end, steps + 1)[:-1]
    else:
        if math.isclose(float(s.sigma(end)), 0, abs_tol=0.00001):
            steps += 1
            append_zero = False
        timesteps = torch.linspace(start, end, steps)

    sigs = []
    for x in range(len(timesteps)):
        ts = timesteps[x]
        sigs.append(float(s.sigma(ts)))

    if append_zero:
        sigs += [0.0]

    return torch.FloatTensor(sigs)


def get_sigmas_karras(n, sigma_min, sigma_max, rho=7., device='cpu'):
    """Constructs the noise schedule of Karras et al. (2022)."""
    ramp = torch.linspace(0, 1, n, device=device)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return append_zero(sigmas).to(device)


def get_sigmas_exponential(n, sigma_min, sigma_max, device='cpu'):
    """Constructs an exponential noise schedule."""
    sigmas = torch.linspace(math.log(sigma_max), math.log(sigma_min), n, device=device).exp()
    return append_zero(sigmas)


def get_sigmas_polyexponential(n, sigma_min, sigma_max, rho=1., device='cpu'):
    """Constructs an polynomial in log sigma noise schedule."""
    ramp = torch.linspace(1, 0, n, device=device) ** rho
    sigmas = torch.exp(ramp * (math.log(sigma_max) - math.log(sigma_min)) + math.log(sigma_min))
    return append_zero(sigmas)


def get_sigmas_vp(n, beta_d=19.9, beta_min=0.1, eps_s=1e-3, device='cpu'):
    """Constructs a continuous VP noise schedule."""
    t = torch.linspace(1, eps_s, n, device=device)
    sigmas = torch.sqrt(torch.exp(beta_d * t ** 2 / 2 + beta_min * t) - 1)
    return append_zero(sigmas)


