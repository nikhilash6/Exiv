from ...utils.enum import ExtendedEnum


class SamplerType(ExtendedEnum):
    DDIM = "DDIM"
    DDIM_CFG_PLUS = "DDIM_CFG++"
    UNIPC = "UniPC"

class SchedulerBase:
    pass

class SamplerBase:
    def __init__(self, sample_func):
        self.sample_func = sample_func
        
    def start_sampling(self, steps):
        pass