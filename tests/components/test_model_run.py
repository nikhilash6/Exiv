import torch

import unittest
from unittest.mock import patch

from tests.test_utils.common import LargeModel, SimpleModel, check_memory_usage, create_large_model_file
from exiv.utils.device import MemoryManager, VRAM_DEVICE, print_mem_usage


class ModelRunTest(unittest.TestCase):
    def setUp(self):
        create_large_model_file()
        MemoryManager.clear_memory()    
    
    def tearDown(self):
        MemoryManager.clear_memory()
    
    # small model run
    @check_memory_usage(expected_mem=12, device=VRAM_DEVICE)
    def test_small_model_run(self):
        model = SimpleModel()
        model.load_model(SimpleModel.CKPT_PATH)
        x = torch.ones(1, 1024)
        out = model(x)
        self.assertEqual(out.shape, (1, 512))
        self.assertTrue(out[0, 0].item(), 1024)
        self.assertEqual(next(model.parameters()).device.type, VRAM_DEVICE)
    
    # large model run
    @check_memory_usage(expected_mem=4350, device=VRAM_DEVICE)
    def test_large_model_run(self):
        model = LargeModel()
        model.load_model(LargeModel.SAFETENSORS_PATH)
        x = torch.ones(1, 16384)
        out = model(x)
        self.assertEqual(out.shape, (1, 4096))
        self.assertTrue(out[0, 0].item(), 16384)
        self.assertEqual(next(model.parameters()).device.type, VRAM_DEVICE)
    
    # small model low mem run
    def test_low_mem_run(self):
        with patch.multiple(
                    MemoryManager, 
                    available_memory=lambda device='cpu': 50.0,
                    total_memory=lambda device='cpu': 100.0):
            model = SimpleModel()
            model.load_model(SimpleModel.SAFETENSORS_PATH)
            x = torch.ones(1, 1024)
            out = model(x)
            self.assertEqual(out.shape, (1, 512))
            self.assertTrue(out[0, 0].item(), 1024)
            self.assertEqual(next(model.parameters()).device.type, "cpu")
    
    # large model low mem run
    def test_low_mem_runtime_error(self):
        with patch.multiple(
                    MemoryManager, 
                    available_memory=lambda device='cpu': 50.0,
                    total_memory=lambda device='cpu': 100.0):
            # not even a single layer can fit in the memory
            with self.assertRaises(RuntimeError):
                model = LargeModel()
                model.load_model(LargeModel.SAFETENSORS_PATH)
                x = torch.ones(1, 1024)
                out = model(x)
                self.assertEqual(out.shape, (1, 512))
                self.assertTrue(out[0, 0].item(), 1024)
                self.assertEqual(next(model.parameters()).device.type, "cpu")
                