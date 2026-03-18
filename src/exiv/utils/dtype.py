import torch
from torch import Tensor

# TODO: implement device specific dtype selection here

# TODO: add stream copying once it is implemented
def cast_to(t: Tensor, dtype=None, device=None, non_blocking=False):
    no_device_change = device is None or t.device == device
    no_dtype_change = dtype is None or t.dtype == dtype
    
    if no_device_change and no_dtype_change: return t
    
    # Don't cast integer tensors (like input_ids) to float dtype
    if dtype is not None and t.dtype in (torch.int64, torch.int32, torch.int16, torch.int8, torch.long, torch.int):
        if device is not None and t.device != device:
            return t.to(device=device, non_blocking=non_blocking)
        return t

    r = torch.empty_like(t, dtype=dtype, device=device)
    r.copy_(t, non_blocking=non_blocking)
    return r


def cast_like_reference(t: Tensor, reference: Tensor, non_blocking: bool = False):
    # casts the dtype and device of 't' to the same as the reference
    return cast_to(t=t, dtype=reference.dtype, device=reference.device, non_blocking=non_blocking)