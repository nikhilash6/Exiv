import torch
import os
import unittest
from safetensors.torch import save_file

from exiv.model_utils.lora_mixin import CACHED_MODEL_LORA_KEY_MAP
from tests.test_utils.common import SimpleModel
from exiv.utils.device import MemoryManager

class LoRASimpleModel(SimpleModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Set weights to 0.0 so we only measure LoRA contribution
        with torch.no_grad():
            self.input_layer.weight.fill_(0.0)
            self.input_layer.bias.fill_(0.0)
            self.output_layer.weight.fill_(0.0)
            self.output_layer.bias.fill_(0.0)

    def create_model_lora_key_map(self, state_dict):
        key_map = {}
        sd = state_dict.keys()
        for k in sd:
            if k.endswith(".weight"):
                key_lora = f"lora.{k.replace('.', '_')}.down"
                key_map[key_lora] = k
                key_map[k] = key_lora
            else:
                key_map["{}".format(k)] = k
        
        setattr(self, CACHED_MODEL_LORA_KEY_MAP, key_map)

class TestMultiStepLora(unittest.TestCase):
    LORA_PATH = "tests/test_utils/assets/models/test_lora_multistep.safetensors"
    LORA_A_PATH = "tests/test_utils/assets/models/test_lora_A.safetensors"
    LORA_B_PATH = "tests/test_utils/assets/models/test_lora_B.safetensors"
    DUMMY_MODEL_PATH = "tests/test_utils/assets/models/dummy_model.safetensors"

    def setUp(self):
        MemoryManager.clear_memory()
        # Create base LoRA (Delta = 0.16)
        self.create_lora(self.LORA_PATH, val_down=0.1, val_up=0.2)
        
        # Create LoRA A (Delta = 0.16) and B (Delta = 0.08)
        self.create_lora(self.LORA_A_PATH, val_down=0.1, val_up=0.2)
        self.create_lora(self.LORA_B_PATH, val_down=0.05, val_up=0.2)
        self.create_model()

    def tearDown(self):
        for path in [self.LORA_PATH, self.LORA_A_PATH, \
            self.LORA_B_PATH, self.DUMMY_MODEL_PATH]:
            if os.path.exists(path):
                os.remove(path)
        MemoryManager.clear_memory()
        
    def create_model(self):
        state_dict = {
            "input_layer.weight": torch.zeros(2048, 1024),
            "input_layer.bias": torch.zeros(2048),
            "output_layer.weight": torch.zeros(512, 2048),
            "output_layer.bias": torch.zeros(512)
        }

        save_file(state_dict, self.DUMMY_MODEL_PATH)

    def create_lora(self, path, val_down, val_up):
        """Generic method to create a Rank 4 LoRA with specific values"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Math:
        # Rank = 4, Alpha = 8.0 -> Scale = 2.0
        # Delta = (Rank * Down * Up) * Scale
        # For 0.1/0.2: (4 * 0.1 * 0.2) * 2.0 = 0.16
        # For 0.05/0.2: (4 * 0.05 * 0.2) * 2.0 = 0.08
        
        rank = 4
        alpha = 8.0
        tensors = {}
        
        # --- Input Layer (1024 -> 2048) ---
        tensors["lora.input_layer_weight.down"] = torch.full((rank, 1024), val_down, dtype=torch.float16)
        tensors["lora.input_layer_weight.up"] = torch.full((2048, rank), val_up, dtype=torch.float16)
        tensors["lora.input_layer_weight.alpha"] = torch.tensor(alpha, dtype=torch.float32)

        # --- Output Layer (2048 -> 512) ---
        tensors["lora.output_layer_weight.down"] = torch.full((rank, 2048), val_down, dtype=torch.float16)
        tensors["lora.output_layer_weight.up"] = torch.full((512, rank), val_up, dtype=torch.float16)
        tensors["lora.output_layer_weight.alpha"] = torch.tensor(alpha, dtype=torch.float32)
        
        save_file(tensors, path)

    def test_multistep_single_lora(self):
        steps = 5
        model = LoRASimpleModel()
        model.load_model(self.DUMMY_MODEL_PATH)
        
        model.add_lora(self.LORA_PATH, base_strength=1.0)
        model.setup_lora_schedule(total_steps=steps)

        dummy_input = torch.ones((1, 1024))
        expected_val = 53687.09

        print(f"\nRunning {steps} steps with constant LoRA schedule...")

        for i in range(steps):
            model.current_time_step = i
            output = model(dummy_input)
            mean_output = output.mean().item()
            
            print(f"Step {i}: Output Mean {mean_output:.4f} (Expected: {expected_val})")
            
            self.assertTrue(torch.isclose(
                torch.tensor(mean_output), 
                torch.tensor(expected_val), 
                rtol=1e-2
            ), f"Step {i} failed")

            self.assertEqual(model.input_layer.weight.sum().item(), 0.0)
            
    def test_variable_schedule_multi_lora(self):
        steps = 5
        model = LoRASimpleModel()
        model.load_model(self.DUMMY_MODEL_PATH)
        
        schedule_a = [0.0, 0.5, 1.0, 0.5, 0.0]
        schedule_b = [1.0, 1.0, 0.0, 0.0, 1.0]

        model.add_lora(self.LORA_A_PATH, base_strength=schedule_a)
        model.add_lora(self.LORA_B_PATH, base_strength=schedule_b)
        model.setup_lora_schedule(total_steps=steps)

        dummy_input = torch.ones((1, 1024))
        
        expected_values = [
            13421.77, # Step 0
            53687.09, # Step 1
            53687.09, # Step 2
            13421.77, # Step 3
            13421.77  # Step 4
        ]

        print(f"\nRunning {steps} steps with DYNAMIC Multi-LoRA schedule...")

        for i in range(steps):
            model.current_time_step = i
            
            output = model(dummy_input)
            mean_output = output.mean().item()
            expected = expected_values[i]
            
            print(f"Step {i} | Str A: {schedule_a[i]}, Str B: {schedule_b[i]} | "
                  f"Got: {mean_output:.2f} (Exp: {expected:.2f})")
            
            self.assertTrue(torch.isclose(
                torch.tensor(mean_output), 
                torch.tensor(expected), 
                rtol=1e-2
            ), f"Step {i} failed")

            self.assertEqual(model.input_layer.weight.sum().item(), 0.0)