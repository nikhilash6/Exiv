from ...utils.enum import ExtendedEnum


class SamplerType(ExtendedEnum):
    DDIM = "DDIM"
    DDIM_CFG_PLUS = "DDIM_CFG++"
    UNIPC = "UniPC"

class BetaSchedule(ExtendedEnum):
    LINEAR = "linear"
    COSINE = "cosine"
    SQRT_LINEAR = "sqrt_linear"
    SQRT = "sqrt"

class SchedulerBase:
    pass

class SamplerBase:
    pass