import torch

import unittest
from unittest.mock import patch
from parameterized import parameterized

from exiv.model_utils.model_loader import RESERVED_MEM
from tests.test_utils.common import LargeModel, SimpleModel, check_memory_usage, create_large_model_file
from exiv.utils.device import OFFLOAD_DEVICE, MemoryManager, VRAM_DEVICE, is_mps_available
from exiv.config import global_config

# TODO: add tests to check the model split method
class ModelRunTest(unittest.TestCase):
    def setUp(self):
        create_large_model_file()
        MemoryManager.clear_memory()    
    
    def tearDown(self):
        MemoryManager.clear_memory()
    
    # model is only 12 MB but adding 20 for buffer
    LOADING_PARAMS = [
        ("no_oom",   {"no_oom": True,  "low_vram": False, "normal_load": False}, 0 if is_mps_available else 12,  OFFLOAD_DEVICE),
        ("low_vram", {"no_oom": False, "low_vram": True,  "normal_load": False}, 12, VRAM_DEVICE),
        ("normal",   {"no_oom": False, "low_vram": False, "normal_load": True},  12, VRAM_DEVICE),
    ]
    @parameterized.expand(LOADING_PARAMS)
    def test_small_model_run_variants(self, load_mode, config, expected_mem, expected_device):
        global_config.update_config(config)
        with check_memory_usage(expected_mem=expected_mem, device=VRAM_DEVICE):
            model = SimpleModel()
            model.load_model(SimpleModel.CKPT_PATH)

            x = torch.ones(1, 1024)
            out = model(x)

            self.assertEqual(out.shape, (1, 512))
            self.assertEqual(out[0, 0].item(), 1024)
            self.assertEqual(next(model.parameters()).device.type, expected_device)
            
            model.cpu()
            del x, out, model
    
    
    LOADING_PARAMS = [
        ("no_oom",   {"no_oom": True,  "low_vram": False, "normal_load": False}, 0 if is_mps_available else 4350,  OFFLOAD_DEVICE),
        ("low_vram", {"no_oom": False, "low_vram": True,  "normal_load": False}, 4350, VRAM_DEVICE),
        ("normal",   {"no_oom": False, "low_vram": False, "normal_load": True},  4350, VRAM_DEVICE),
    ]
    @parameterized.expand(LOADING_PARAMS)
    def test_large_model_run(self, load_mode, config, expected_mem, expected_device):
        global_config.update_config(config)
        
        with check_memory_usage(expected_mem=expected_mem, device=VRAM_DEVICE):
            model = LargeModel()
            model.load_model(LargeModel.SAFETENSORS_PATH)
            x = torch.ones(1, 16384)
            out = model(x)
            self.assertEqual(out.shape, (1, 4096))
            self.assertTrue(out[0, 0].item(), 16384)
            self.assertEqual(next(model.parameters()).device.type, expected_device)
            
            model.cpu()
            del x, out, model
    
    # small model low mem run
    LOADING_PARAMS = [
        ("no_oom",   {"no_oom": True,  "low_vram": False, "normal_load": False}, 0 if is_mps_available else 12,  OFFLOAD_DEVICE),
        ("low_vram", {"no_oom": False, "low_vram": True,  "normal_load": False}, 12, VRAM_DEVICE),
        ("normal",   {"no_oom": False, "low_vram": False, "normal_load": True},  12, VRAM_DEVICE),
    ]
    @parameterized.expand(LOADING_PARAMS)
    def test_low_mem_run(self, load_mode, config, expected_mem, expected_device):
        with patch.multiple(
                    MemoryManager, 
                    available_memory=lambda device='cpu': RESERVED_MEM + 50.0,
                    total_memory=lambda device='cpu': RESERVED_MEM + 100.0):
            global_config.update_config(config)
            with check_memory_usage(expected_mem=expected_mem, device=VRAM_DEVICE):
                print("current mode: ", global_config.loading_mode)
                model = SimpleModel()
                model.load_model(SimpleModel.CKPT_PATH)

                x = torch.ones(1, 1024)
                out = model(x)

                self.assertEqual(out.shape, (1, 512))
                self.assertEqual(out[0, 0].item(), 1024)
                self.assertEqual(next(model.parameters()).device.type, expected_device)
                
                model.cpu()
                del x, out, model
    
    # large model low mem run
    # this is just a dummy test for now, as low mem environment is not properly simulated
    LOADING_PARAMS = [
        ("low_vram", {"no_oom": False, "low_vram": True,  "normal_load": False}, 4350, VRAM_DEVICE),
    ]
    @parameterized.expand(LOADING_PARAMS)
    def test_low_mem_runtime_error(self, load_mode, config, expected_mem, expected_device):
        with patch.multiple(
                    MemoryManager, 
                    available_memory=lambda device='cpu': RESERVED_MEM + 50.0,
                    total_memory=lambda device='cpu': RESERVED_MEM + 100.0):
            global_config.update_config(config)
            
            # not even a single layer can fit in the memory
            with self.assertRaises(RuntimeError):
                print("current mode: ", global_config.loading_mode)
                model = LargeModel()
                model.load_model(LargeModel.SAFETENSORS_PATH)
                x = torch.ones(1, 16384)
                out = model(x)
                self.assertEqual(out.shape, (1, 4096))
                self.assertTrue(out[0, 0].item(), 16384)
                self.assertEqual(next(model.parameters()).device.type, expected_device)
                