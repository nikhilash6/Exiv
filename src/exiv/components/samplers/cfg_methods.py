import torch


def default_cfg(*args, **kwargs):
    cond, uncond, cond_scale = kwargs["cond"], kwargs["uncond"], kwargs["cond_scale"]
    cfg_result = uncond + (cond - uncond) * cond_scale
    return cfg_result