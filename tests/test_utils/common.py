import torch
from torch import nn

import os, gc
import psutil
from functools import wraps

from exiv.config import BYTES_IN_MB
from exiv.utils.device import MemoryManager, ProcDevice, is_mps_available
from exiv.model_utils.model_mixin import ModelMixin

try:
    import torch_xla.core.xla_model as xm
    if xm.xla_device(): is_xla_available = True
except ImportError:
    pass

script_dir = os.path.dirname(os.path.abspath(__file__))


def print_top_gpu_tensors(n=5, device="cuda:0"):
    objs = gc.get_objects()
    gpu_tensors = []
    visited_ids = set()
    for obj in objs:
        try:
            if torch.is_tensor(obj) and obj.is_cuda and obj.device == torch.device(device) and id(obj) not in visited_ids:
                gpu_tensors.append((obj.element_size() * obj.nelement(), obj))
                visited_ids.add(id(obj))
        except Exception:
            pass
    
    # NOTE: the size here is theoretical, it could have been freed by gc or some other mechanism
    gpu_tensors = sorted(gpu_tensors, key=lambda x: x[0], reverse=True)
    print(f"\n Top {n} GPU tensors on {device}:")
    for size, tensor in gpu_tensors[:n]:
        size_mb = size / (1024**2)
        print(f"- {tensor.shape} | {tensor.dtype} | {size_mb:.2f} MB | requires_grad={tensor.requires_grad}")


class check_memory_usage:
    """
    This only checks the memory at the start and end of the program, so if everything is offloaded,
    this will calculate the memory usage to be 0.. Its a problem with MPS really
    """
    def __init__(self, expected_mem, device=ProcDevice.CPU.value, atol=10, rtol=0.1):
        self.expected_mem = expected_mem
        self.device = device
        self.atol = atol
        self.rtol = rtol
        self.initial_mem_mb = 0

    def __enter__(self):
        device = torch.device(self.device)
        
        if device.type == ProcDevice.CUDA.value:
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        elif device.type == ProcDevice.XLA.value:
            xm.reset_peak_memory_stats(device)
        elif device.type == ProcDevice.MPS.value:
            torch.mps.synchronize()
            # MPS: No peak tracker. Measure delta
            self.initial_mem_mb = torch.mps.current_allocated_memory() / (1024**2)
        else:
            # CPU: No VRAM
            self.initial_mem_mb = 0

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        device = torch.device(self.device)
        mem_diff = 0
        
        if device.type == ProcDevice.CUDA.value:
            torch.cuda.synchronize(device)
            mem_diff = torch.cuda.max_memory_allocated(device) / (1024**2)
        
        elif device.type == ProcDevice.XLA.value:
            mem_info = xm.get_memory_info(device)
            mem_diff = mem_info['peak_usage_bytes'] / (1024**2)

        elif device.type == ProcDevice.MPS.value:
            torch.mps.synchronize()
            # MPS: Report delta.
            final_mem_mb = torch.mps.current_allocated_memory() / (1024**2)
            mem_diff = final_mem_mb - self.initial_mem_mb
        
        is_close = torch.isclose(
                        torch.tensor(float(mem_diff)), 
                        torch.tensor(float(self.expected_mem)), 
                        rtol=self.rtol,
                        atol=self.atol
                    )
        
        if not is_close and not is_mps_available:
            print("device: ", self.device)
            print_top_gpu_tensors()
            raise AssertionError(
                f"Memory usage difference {mem_diff:.2f} MB is not close to expected {self.expected_mem} MB."
            )

    def __call__(self, test_func):
        @wraps(test_func)
        def wrapper(*args, **kwargs):
            with self:
                test_func(*args, **kwargs)
        return wrapper

# --------------- Dummy models
class SimpleModel(ModelMixin):
    PTH_PATH = os.path.abspath(os.path.join(script_dir, "./assets/models/simple_model.pth"))
    SAFETENSORS_PATH = os.path.abspath(os.path.join(script_dir, "./assets/models/simple_model.safetensors"))
    SFT_PATH = os.path.abspath(os.path.join(script_dir, "./assets/models/simple_model.sft"))
    CKPT_PATH = os.path.abspath(os.path.join(script_dir, "./assets/models/simple_model.ckpt"))
    PT_PATH = os.path.abspath(os.path.join(script_dir, "./assets/models/simple_model.pt"))
    ALL_MODEL_PATHS = [PTH_PATH, SAFETENSORS_PATH, CKPT_PATH, PT_PATH, SFT_PATH]
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_layer = nn.Linear(1024, 2048)
        self.output_layer = nn.Linear(2048, 512)

    def forward(self, x):
        return self.output_layer(self.input_layer(x))
    
class LargeModel(ModelMixin):
    SAFETENSORS_PATH = os.path.abspath(os.path.join(script_dir, "./assets/models/large_model.safetensors"))
    ALL_MODEL_PATHS = [SAFETENSORS_PATH]
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_layer = nn.Linear(16384, 32768)
        self.hidden_layer = nn.Linear(32768, 16384)
        self.output_layer = nn.Linear(16384, 4096)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.hidden_layer(x)
        return self.output_layer(x)

# instead of downloading from HF, we just create the dummy
# model at runtime
def create_large_model_file():
    from safetensors.torch import save_file
    
    # one day i'll make this relative
    model_path = "tests/test_utils/assets/models/large_model.safetensors"
    if os.path.exists(model_path): return
    with torch.no_grad():
        model = LargeModel().to_empty(device="cpu")
        # Input layer: identity-like mapping
        model.input_layer.weight.zero_()
        model.input_layer.bias.zero_()
        for i in range(16384):
            model.input_layer.weight[i, i] = 1.0  

        # Hidden layer: pass-through first 16384 dims
        model.hidden_layer.weight.zero_()
        model.hidden_layer.bias.zero_()
        for i in range(16384):
            model.hidden_layer.weight[i, i] = 1.0  

        # Output layer: sum all into first neuron
        model.output_layer.weight.zero_()
        model.output_layer.bias.zero_()
        model.output_layer.weight[0, :].fill_(1.0)
    
    state_dict = model.state_dict()
    save_file(state_dict, model_path)
            
    
        
