import torch

import numpy as np

from ..enum import BetaSchedule

def make_beta_schedule(schedule_type, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    # added noise is proportional to sqrt(beta) and thus schedules are also defined with that in mind
    if schedule_type == BetaSchedule.LINEAR.value:
        betas = (
                torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
        )

    elif schedule_type == BetaSchedule.COSINE.value:
        timesteps = (
                torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * np.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = torch.clamp(betas, min=0, max=0.999)
    
    elif schedule_type == BetaSchedule.SQRT_LINEAR.value:
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64)
    
    elif schedule_type == BetaSchedule.SQRT.value:
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64) ** 0.5
    
    else:
        raise ValueError(f"schedule '{schedule_type}' unknown.")
    return betas