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
    QUANT_PARAMS = [
        ("fp8", 1120),
        ("fp4", 580),
    ]
    @parameterized.expand(QUANT_PARAMS)
    def test_torchao_run(self, quant_type, expected_mem):
        from kirin.quantizers.bnb.bnb import BNBQuantizer
        from kirin.quantizers.base import BNBQuantizerConfig
        
        with check_memory_usage(expected_mem=expected_mem, device=DEFAULT_DEVICE):
            kwargs_dict = {
                "fp4": {'load_in_4bit': True},
                "fp8": {'load_in_8bit': True}
            }
            app_logger.info(f"quantizing: {quant_type}")
            quant_config = BNBQuantizerConfig(**kwargs_dict[quant_type])
            quantizer = BNBQuantizer(quantization_config=quant_config)
            model = LargeModel(quantizer=quantizer)
            model.load_model(LargeModel.SAFETENSORS_PATH)
            x = torch.ones(1, 16384)
            out = model(x)
            self.assertEqual(out.shape, (1, 4096))
            self.assertTrue(out[0, 0].item(), 16384)
            self.assertEqual(next(model.parameters()).device.type, DEFAULT_DEVICE)

   