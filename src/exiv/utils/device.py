import torch
import psutil

from ..constants import BYTES_IN_MB
from .enum import ExtendedEnum
from .logging import app_logger

# ------------------ Device availability
# processing device
class ProcDevice(ExtendedEnum):
    CPU = "cpu"     # this might need changing to inlucde mps
    CUDA = "cuda"
    MPS = "mps"
    XLA = "xla"     # more like a backend than a device

is_mps_available = False
is_cuda_available = False
is_xla_available = False
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

DEFAULT_DEVICE = ProcDevice.CPU.value
if is_cuda_available:
    DEFAULT_DEVICE = ProcDevice.CUDA.value
elif is_mps_available:
    DEFAULT_DEVICE = ProcDevice.MPS.value
elif is_xla_available:
    DEFAULT_DEVICE = ProcDevice.XLA.value

# ------------------ Memory availability
# TODO: do somekind of ttl based caching to eliminate repeated calls
class MemoryManager:
    @staticmethod
    def available_memory(device=ProcDevice.CPU.value):
        device = torch.device(device)

        # cuda (nvidia/rocm)
        if device.type == ProcDevice.CUDA.value:
            torch.cuda.synchronize(device)
            stats = torch.cuda.memory_stats(device)
            mem_active = stats['active_bytes.all.current']
            mem_reserved = stats['reserved_bytes.all.current']
            mem_free_from_driver, _ = torch.cuda.mem_get_info(device)
            mem_free_torch = mem_reserved - mem_active
            return (mem_free_from_driver + mem_free_torch) / BYTES_IN_MB
        # mps
        elif device.type == ProcDevice.MPS.value:
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
        if is_cuda_available:
            torch.cuda.empty_cache()
        if is_mps_available:
            torch.mps.empty_cache()
            

def print_mem_usage(model, tag):
    app_logger.debug(f"----------------------- {tag}")
    app_logger.debug(f"model params ****")
    
    for name, p in model.named_parameters():
        app_logger.debug(f"{name} : {p.shape} : {p.dtype} : {p.__class__}")

    app_logger.debug("model buffers ****")
    for name, b in model.named_buffers():
        app_logger.debug(f"{name} : {b.shape} : {b.dtype} : {b.__class__}")