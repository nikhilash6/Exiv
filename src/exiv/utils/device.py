import importlib
import torch
import torch.nn.functional as F

import psutil

from .enum import ExtendedEnum
from .logging import app_logger
from ..config import BYTES_IN_MB

# ------------------ Device availability
# processing device
class ProcDevice(ExtendedEnum):
    CPU = "cpu"     # this might need changing to inlucde mps
    CUDA = "cuda"
    MPS = "mps"
    XLA = "xla"     # more like a backend than a device
    XPU = "xpu"
    NPU = "npu"

is_mps_available = False
is_cuda_available = False
is_xla_available = False
is_xpu_available = False
is_npu_available = False
is_cpu_available = True

if torch.cuda.is_available(): is_cuda_available = True
CUDA_CC_VERSION = (0, 0) if not is_cuda_available \
    else torch.cuda.get_device_capability()
CUDA_CC = CUDA_CC_VERSION[0] * 10 + CUDA_CC_VERSION[1]

if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): is_mps_available = True
try:
    import torch_xla.core.xla_model as xm
    if xm.xla_device(): is_xla_available = True
except ImportError:
    pass
if hasattr(torch, "xpu") and torch.xpu.is_available(): is_xpu_available = True

def npu_check(check_device=False):
    "Checks if `torch_npu` is installed and potentially if a NPU is in the environment"
    if importlib.util.find_spec("torch_npu") is None:
        return False

    # NOTE: importing torch_npu may raise error in some envs
    # e.g. inside cpu-only container with torch_npu installed
    try:
        import torch_npu  # noqa: F401
    except Exception:
        return False

    if check_device:
        try:
            # Will raise a RuntimeError if no NPU is found
            _ = torch.npu.device_count()
            return torch.npu.is_available()
        except RuntimeError:
            return False
    return hasattr(torch, "npu") and torch.npu.is_available()

is_npu_available = npu_check()

VRAM_DEVICE = ProcDevice.CPU.value
OFFLOAD_DEVICE = ProcDevice.CPU.value

if is_cuda_available:
    VRAM_DEVICE = ProcDevice.CUDA.value
elif is_mps_available:
    VRAM_DEVICE = ProcDevice.MPS.value
elif is_npu_available:
    VRAM_DEVICE = ProcDevice.NPU.value
elif is_xla_available:
    VRAM_DEVICE = ProcDevice.XLA.value
elif is_xpu_available:
    VRAM_DEVICE = ProcDevice.XPU.value
    
def is_same_device(first_device, second_device):
    if first_device.type != second_device.type:
        return False

    if first_device.type != "cpu" and first_device.index is None:
        first_device = torch.device(first_device.type, index=0)

    if second_device.type != "cpu" and second_device.index is None:
        second_device = torch.device(second_device.type, index=0)

    return first_device == second_device

# ------------------ Memory availability

RESERVED_MEM = 1024         # buffers, cache.. 

# TODO: do somekind of ttl based caching to eliminate repeated calls
class MemoryManager:
    @staticmethod
    def available_memory(device=VRAM_DEVICE):
        device = torch.device(device)

        # cuda (nvidia/rocm)
        if device.type == ProcDevice.CUDA.value:
            torch.cuda.synchronize(device)
            stats = torch.cuda.memory_stats(device)
            mem_active = stats['active_bytes.all.current']                  # actively used by tensors
            mem_reserved = stats['reserved_bytes.all.current']              # mem asked to be reserved by driver
            mem_free_from_driver, _ = torch.cuda.mem_get_info(device)       # mem not yet reserved
            mem_free_torch = mem_reserved - mem_active                      # mem reserved but not in use
            return (mem_free_from_driver + mem_free_torch) / BYTES_IN_MB
        # mps
        elif device.type == ProcDevice.MPS.value:
            torch.mps.synchronize()
            total_mem = psutil.virtual_memory().total
            driver_alloc = torch.mps.current_allocated_memory()
            return max(total_mem - driver_alloc, 0) / BYTES_IN_MB
        # xla (most probably not needed)
        elif device.type == ProcDevice.XLA.value:
            mem_info = xm.get_memory_info(device)
            return int(mem_info['kb_free']) * 1024 / BYTES_IN_MB
        # cpu
        else:
            return psutil.virtual_memory().available / BYTES_IN_MB

    @staticmethod
    def total_memory(device=ProcDevice.CPU.value):
        device = torch.device(device)

        # cuda
        if device.type == ProcDevice.CUDA.value:
            return torch.cuda.get_device_properties(device).total_memory / BYTES_IN_MB
        # mps
        elif device.type == ProcDevice.MPS.value:
            return psutil.virtual_memory().total / BYTES_IN_MB
        # xla
        elif device.type == ProcDevice.XLA.value:
            mem_info = xm.get_memory_info(device)
            return int(mem_info['kb_total']) * 1024 / BYTES_IN_MB
        # cpu
        else:
            return psutil.virtual_memory().total / BYTES_IN_MB
    
    @staticmethod
    def clear_memory():
        import gc
        gc.collect()
        if is_xpu_available:
            torch.xpu.empty_cache()
        elif is_npu_available:
            torch.npu.empty_cache()
        elif is_mps_available:
            torch.mps.empty_cache()
        elif is_cuda_available:
            torch.cuda.empty_cache()
            

def print_mem_usage(model, tag):
    app_logger.debug(f"----------------------- {tag}")
    app_logger.debug(f"model params ****")
    
    for name, p in model.named_parameters():
        app_logger.debug(f"{name} : {p.shape} : {p.dtype} : {p.__class__}")

    app_logger.debug("model buffers ****")
    for name, b in model.named_buffers():
        app_logger.debug(f"{name} : {b.shape} : {b.dtype} : {b.__class__}")
        

# -------------- ATTN availability
XFORMERS_AVAILABLE = False
try:
    import xformers
    import xformers.ops
    XFORMERS_IS_AVAILABLE = True
    try:
        XFORMERS_IS_AVAILABLE = xformers._has_cpp_library
    except:
        pass
    try:
        XFORMERS_VERSION = xformers.version.__version__
        app_logger.info("xformers version: {}".format(XFORMERS_VERSION))
        if XFORMERS_VERSION.startswith("0.0.18"):
            # bug according to comfy devs
            app_logger.warning("\nWARNING: This version of xformers has a major bug where you will get black images when generating high resolution images.")
            app_logger.warning("Please downgrade or upgrade xformers to a different version.\n")
            XFORMERS_ENABLED_VAE = False
    except:
        pass
except:
    XFORMERS_IS_AVAILABLE = False
    
SDPA_AVAILABLE = hasattr(F, 'scaled_dot_product_attention')
