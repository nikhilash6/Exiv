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
    
class ModelType(ExtendedEnum):
    EPS = "eps"
    V_PREDICTION = "v_pred"
    V_PREDICTION_EDM = "v_pred_edm"
    FLOW = "flow"
    STABLE_CASCADE = "stable_cascade"
    EDM = "edm"
    V_PREDICTION_CONTINUOUS = "v_pred_cont"
    FLUX = "flux"

class SchedulerBase:
    pass

class SamplerBase:
    pass