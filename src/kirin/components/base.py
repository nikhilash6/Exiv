from abc import ABC

# component interface
class IComponent(ABC):
    def __init__(self, device, dtype):
        self.device = device
        self.dtype = dtype
    
    # overridden
    def __call__(self, *args, **kwargs):
        pass