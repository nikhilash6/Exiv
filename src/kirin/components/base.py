from abc import ABC

# component interface
class ComponentMixin(ABC):
    def __init__(self, device, dtype):
        self.device = device
        self.dtype = dtype
    
    # overridden
    def __call__(self, *args, **kwargs):
        pass
    
    def move_to_device(self, device):
        self.device = device
        # TODO: trigger some kind of callback