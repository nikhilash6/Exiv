import torch

import unittest

from tests.test_utils.common import SimpleModel, check_memory_usage
from kirin.utils.device import MemoryManager, DEFAULT_DEVICE


class ModelRunTest(unittest.TestCase):
    def setUp(self):
        MemoryManager.clear_memory()    
    
    def tearDown(self):
        MemoryManager.clear_memory()
        
    @check_memory_usage(expected_mem=12, device=DEFAULT_DEVICE)
    def test_model_device(self):
        model = SimpleModel()
        model.load_model(SimpleModel.CKPT_PATH)
        x = torch.ones(1, 1024)
        out = model(x)
        self.assertEqual(out.shape, (1, 512))
        self.assertTrue(torch.all(out == 0))
        self.assertEqual(next(model.parameters()).device.type, DEFAULT_DEVICE)