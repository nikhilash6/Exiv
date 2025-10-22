import torch

from ..utils.device import ProcDevice, is_cuda_available, is_xpu_available
from ..config import global_config

device_module = None
stream_context_manager = None
device_obj = None

if is_cuda_available:
    device_module = torch.cuda
    stream_context_manager = torch.cuda.stream
    device_obj = torch.device(ProcDevice.CUDA.value)
elif is_xpu_available:
    device_module = torch.xpu
    stream_context_manager = torch.xpu.stream
    device_obj = torch.device(ProcDevice.XPU.value)

def stream_context():
    return stream_context_manager

def _get_stream_internal(stream_creator_func):
    if not stream_creator_func or not device_obj:
        return None
    try:
        stream = stream_creator_func(device_obj)
        with stream_context_manager(stream):
            torch.zeros((1, 1)).to(device_obj, torch.float32)
        stream.synchronize()
        return stream
    except Exception:
        return None

def get_current_stream():
    return _get_stream_internal(getattr(device_module, 'current_stream', None))

def get_new_stream():
    return _get_stream_internal(getattr(device_module, 'Stream', None))

def should_use_stream():
    return global_config.use_multi_stream and current_stream is not None and mover_stream is not None

current_stream = get_current_stream()
mover_stream = get_new_stream()