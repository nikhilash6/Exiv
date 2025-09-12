import os
from functools import wraps
import torch
from torch import nn

from kirin.utils.device import MemoryManager, ProcDevice, is_mps_available
from kirin.utils.model_utils import ModelMixin

script_dir = os.path.dirname(os.path.abspath(__file__))

class check_memory_usage:
    def __init__(self, expected_mem, device=ProcDevice.CPU.value, atol=5, rtol=0.1):
        self.expected_mem = expected_mem
        self.device = device
        self.atol = atol
        self.rtol = rtol
        self.initial_mem_mb = 0

    def __enter__(self):
        self.initial_mem_mb = MemoryManager.available_memory(self.device)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        final_mem_mb = MemoryManager.available_memory(self.device)
        mem_diff = self.initial_mem_mb - final_mem_mb
        
        is_close = torch.isclose(
                        torch.tensor(float(mem_diff)), 
                        torch.tensor(float(self.expected_mem)), 
                        rtol=self.rtol,
                        atol=self.atol
                    )
        
        is_mac_cpu = is_mps_available and self.device == ProcDevice.CPU.value
        
        if not is_close and not is_mac_cpu:
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
            
    
        
