from ...utils.enum import ExtendedEnum


class SamplerType(ExtendedEnum):
    DDIM = "DDIM"
    DDIM_CFG_PLUS = "DDIM_CFG++"
    UNIPC = "UniPC"

class SchedulerBase:
    pass

class SamplerBase:
    pass