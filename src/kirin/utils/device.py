import torch
import psutil

from .enum import ExtendedEnum


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

if torch.cuda.is_available: is_cuda_available = True
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
class MemoryManager:
    def available_memory(self, device=ProcDevice.CPU.value):
        device = torch.device(device)

        # cuda (nvidia/rocm)
        if device.type == ProcDevice.CUDA.value:
            stats = torch.cuda.memory_stats(device)
            mem_active = stats['active_bytes.all.current']
            mem_reserved = stats['reserved_bytes.all.current']
            mem_free_from_driver, _ = torch.cuda.mem_get_info(device)
            mem_free_torch = mem_reserved - mem_active
            return mem_free_from_driver + mem_free_torch
        # mps
        elif device.type == ProcDevice.MPS.value:
            total_mem = psutil.virtual_memory().total
            driver_alloc = torch.mps.current_allocated_memory()
            return max(total_mem - driver_alloc, 0)
        # xla (most probably not needed)
        elif device.type == ProcDevice.XLA.value:
            mem_info = xm.get_memory_info(device)
            return int(mem_info['kb_free']) * 1024
        # cpu
        else:
            return psutil.virtual_memory().available

    def total_memory(self, device="cpu"):
        device = torch.device(device)

        # cuda
        if device.type == ProcDevice.CUDA.value:
            return torch.cuda.get_device_properties(device).total_memory
        # mps
        elif device.type == ProcDevice.MPS.value:
            return psutil.virtual_memory().total
        # xla
        elif device.type == ProcDevice.XLA.value:
            mem_info = xm.get_memory_info(device)
            return int(mem_info['kb_total']) * 1024
        # cpu
        else:
            return psutil.virtual_memory().total

        
mem_manager = MemoryManager()