import torch

import unittest

from tests.test_utils.common import LargeModel, SimpleModel, check_memory_usage
from kirin.utils.device import MemoryManager, DEFAULT_DEVICE


class ModelRunTest(unittest.TestCase):
    def setUp(self):
        MemoryManager.clear_memory()    
    
    def tearDown(self):
        MemoryManager.clear_memory()
        
    @check_memory_usage(expected_mem=12, device=DEFAULT_DEVICE)
    def test_small_model_run(self):
        model = SimpleModel()
        model.load_model(SimpleModel.CKPT_PATH)
        x = torch.ones(1, 1024)
        out = model(x)
        self.assertEqual(out.shape, (1, 512))
        self.assertTrue(out[0, 0].item(), 1024)
        self.assertEqual(next(model.parameters()).device.type, DEFAULT_DEVICE)
    
    @check_memory_usage(expected_mem=4350, device=DEFAULT_DEVICE)
    def test_large_model_run(self):
        model = LargeModel()
        model.load_model(LargeModel.SAFETENSORS_PATH)
        x = torch.ones(1, 16384)
        out = model(x)
        self.assertEqual(out.shape, (1, 4096))
        self.assertTrue(out[0, 0].item(), 16384)
        self.assertEqual(next(model.parameters()).device.type, DEFAULT_DEVICE)