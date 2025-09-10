import os
from functools import wraps
import torch
from torch import nn

from kirin.utils.device import MemoryManager, ProcDevice, is_mps_available
from kirin.utils.model_utils import ModelMixin

script_dir = os.path.dirname(os.path.abspath(__file__))

def check_memory_usage(expected_mem, device=ProcDevice.CPU.value, atol=50, rtol=0.01):
    def decorator(test_func):
        @wraps(test_func)
        def wrapper(*args, **kwargs):
            initial_mem_mb = MemoryManager.available_memory(device)
            test_func(*args, **kwargs)
            final_mem_mb = MemoryManager.available_memory(device)
            mem_diff = initial_mem_mb - final_mem_mb
            
            # not using rtol atm
            is_close = torch.isclose(torch.tensor(float(mem_diff)), 
                                     torch.tensor(float(expected_mem)), 
                                     atol=atol)
            
            # TODO: psutil is unreliable on mac (for cpu), find a work around
            is_mac_cpu = is_mps_available and device == ProcDevice.CPU.value
            
            if not is_close and not is_mac_cpu:
                raise AssertionError(
                    f"Memory usage difference {mem_diff:.2f} MB is not close to expected {expected_mem} MB. "
                    f"On the device {device}. "
                    f"Absolute difference: {abs(mem_diff - expected_mem):.2f} MB. "
                    f"Absolute tolerance (atol): {atol} MB, Relative tolerance (rtol): {rtol}."
                )
        return wrapper
    return decorator

# --------------- Dummy models
# TODO: put checkpoints on git lfs and have them be downloaded automatically during test run
class SimpleModel(ModelMixin):
    PTH_PATH = os.path.abspath(os.path.join(script_dir, "./assets/models/simple_model.pth"))
    SAFETENSORS_PATH = os.path.abspath(os.path.join(script_dir, "./assets/models/simple_model.safetensors"))
    SFT_PATH = os.path.abspath(os.path.join(script_dir, "./assets/models/simple_model.sft"))
    CKPT_PATH = os.path.abspath(os.path.join(script_dir, "./assets/models/simple_model.ckpt"))
    PT_PATH = os.path.abspath(os.path.join(script_dir, "./assets/models/simple_model.pt"))
    ALL_MODEL_PATHS = [PTH_PATH, SAFETENSORS_PATH, CKPT_PATH, PT_PATH, SFT_PATH]
    
    def __init__(self):
        super().__init__()
        self.input_layer = nn.Linear(1024, 2048)
        self.output_layer = nn.Linear(2048, 512)

    def forward(self, x):
        return self.output_layer(self.input_layer(x))
    
class LargeModel(ModelMixin):
    SAFETENSORS_PATH = os.path.abspath(os.path.join(script_dir, "./assets/models/large_model.safetensors"))
    ALL_MODEL_PATHS = [SAFETENSORS_PATH]
    
    def __init__(self):
        super().__init__()
        self.input_layer = nn.Linear(16384, 32768)
        self.hidden_layer = nn.Linear(32768, 16384)
        self.output_layer = nn.Linear(16384, 4096)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.hidden_layer(x)
        return self.output_layer(x)