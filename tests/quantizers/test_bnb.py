import torch

import unittest
from unittest.mock import patch
from parameterized import parameterized

from tests.test_utils.common import LargeModel, SimpleModel, check_memory_usage, create_large_model_file
from kirin.utils.device import MemoryManager, DEFAULT_DEVICE, is_cuda_available
from kirin.utils.logging import app_logger
from kirin.utils.model_utils import ModelMixin

@unittest.skipIf(not is_cuda_available, "Only available for cuda devices")
class TorchBNBRunTest(unittest.TestCase):
    def setUp(self):
        create_large_model_file()
        MemoryManager.clear_memory()    
    
    def tearDown(self):
        MemoryManager.clear_memory()
        
    # bnb run
    @check_memory_usage(expected_mem=1100, device=DEFAULT_DEVICE)
    def test_torchao_run(self):
        from kirin.quantizers.bnb.bnb import BNBQuantizer
        from kirin.quantizers.base import BNBQuantizerConfig
        
        app_logger.info(f"quantizing: 8bit")
        quant_config = BNBQuantizerConfig(load_in_4bit=True)
        quantizer = BNBQuantizer(quantization_config=quant_config)
        model = LargeModel(quantizer=quantizer)
        model.load_model(LargeModel.SAFETENSORS_PATH)
        x = torch.ones(1, 16384)
        out = model(x)
        self.assertEqual(out.shape, (1, 4096))
        self.assertTrue(out[0, 0].item(), 16384)
        self.assertEqual(next(model.parameters()).device.type, DEFAULT_DEVICE)

   