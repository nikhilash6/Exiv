import functools
import torch

def materialize_meta_buffers(**buffer_keys_and_funcs):
    """
    Decorator to automatically materialize meta tensors in a module's buffers 
    before running the forward pass.
    
    Args:
        **buffer_keys_and_funcs: Keyword arguments where keys are the buffer attribute 
            names and values are functions that can recalculate them. The functions 
            must accept (module, device) as arguments.
            Example: @materialize_meta_buffers(inv_freq=lambda self, device: _compute_rope_inv_freq(self.config, device)[0])
    """
    def decorator(forward_fn):
        @functools.wraps(forward_fn)
        def wrapper(self, *args, **kwargs):
            # Extract the device from the first tensor argument, or fallback to module's device
            device = None
            for arg in args:
                if isinstance(arg, torch.Tensor):
                    device = arg.device
                    break
            
            if device is None:
                # Look inside kwargs for a tensor
                for v in kwargs.values():
                    if isinstance(v, torch.Tensor):
                        device = v.device
                        break
            
            if device is None:
                try:
                    device = next(self.parameters()).device
                except StopIteration:
                    device = torch.device("cpu")

            # Check and materialize each buffer
            for attr_name, compute_func in buffer_keys_and_funcs.items():
                buffer = getattr(self, attr_name, None)
                if buffer is not None and buffer.device.type == "meta":
                    # Materialize on the correct device
                    new_buffer = compute_func(self, device)
                    self.register_buffer(attr_name, new_buffer, persistent=False)
                    
            return forward_fn(self, *args, **kwargs)
        return wrapper
    return decorator