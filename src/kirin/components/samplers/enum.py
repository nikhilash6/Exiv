from websockets import ExtensionName
from ...utils.enum import ExtendedEnum


class KSamplerType(ExtendedEnum):
    EULER = "euler"
    EULER_CFG_PP = "euler_cfg_pp"
    EULER_ANCESTRAL = "euler_ancestral"
    EULER_ANCESTRAL_CFG_PP = "euler_ancestral_cfg_pp"
    HEUN = "heun"
    HEUNPP2 = "heunpp2"
    DPM_2 = "dpm_2"
    DPM_2_ANCESTRAL = "dpm_2_ancestral"
    LMS = "lms"
    DPM_FAST = "dpm_fast"
    DPM_ADAPTIVE = "dpm_adaptive"
    DPMPP_2S_ANCESTRAL = "dpmpp_2s_ancestral"
    DPMPP_SDE = "dpmpp_sde"
    DPMPP_SDE_GPU = "dpmpp_sde_gpu"
    DPMPP_2M = "dpmpp_2m"
    DPMPP_2M_SDE = "dpmpp_2m_sde"
    DPMPP_2M_SDE_GPU = "dpmpp_2m_sde_gpu"
    DPMPP_3M_SDE = "dpmpp_3m_sde"
    DPMPP_3M_SDE_GPU = "dpmpp_3m_sde_gpu"
    DDPM = "ddpm"
    LCM = "lcm"
    IPNDM = "ipndm"
    IPNDM_V = "ipndm_v"
    DEIS = "deis"

class SamplerType(ExtendedEnum):
    DDIM = "ddim"
    UNI_PC = "uni_pc"
    UNI_PC_BH2 = "uni_pc_bh2"
    
'''
this is like a trick of the trade, a1111 issue - https://github.com/AUTOMATIC1111/stable-diffusion-webui/issues/15050
basically the last jump from very low noise to 0 noise in ksampler is numerically unstable
for certain samplers, and thus we kind of widen the jump, from a slightly bigger number to 0
'''
DISCARD_PENULTIMATE_SIGMA_SAMPLERS = [
    KSamplerType.DPM_2.value, 
    KSamplerType.DPM_2_ANCESTRAL.value, 
    SamplerType.UNI_PC.value, 
    SamplerType.UNI_PC_BH2.value,
]
    
class SchedulerType(ExtendedEnum):
    NORMAL = "normal"
    KARRAS = "karras"
    EXPONENTIAL = "exponential"
    SGM_UNIFORM = "sgm_uniform"
    SIMPLE = "simple"
    DDIM_UNIFORM = "ddim_uniform"
    BETA = "beta"
    

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