import torch

import unittest
from unittest.mock import patch
from parameterized import parameterized

from tests.test_utils.common import LargeModel, SimpleModel, check_memory_usage, create_large_model_file
from kirin.utils.device import MemoryManager, DEFAULT_DEVICE, is_cuda_available, CUDA_CC
from kirin.utils.logging import app_logger
from kirin.utils.model_utils import ModelMixin

@unittest.skipIf(not is_cuda_available or CUDA_CC < 89, "Only available for cuda devices")
class TorchAORunTest(unittest.TestCase):
    def setUp(self):
        create_large_model_file()
        MemoryManager.clear_memory()    
    
    def tearDown(self):
        MemoryManager.clear_memory()
        
    # torchao run
    # these mem values are empirically determined
    QUANT_PARAMS = [
        ("fp8dq_e4m3", 1130),
        ("int8dq_int4", 1360),
        ("int4dq_int4", 550),
        ("int8dq_int8", 1100),
        ("int8", 1200),
        ("fp8wo_e4m3", 1130),
        ("fp8wo_e5m2", 1130),
    ]
    @parameterized.expand(QUANT_PARAMS)
    def test_torchao_run(self, quant_type, expected_mem):
        from kirin.quantizers.torchao.torchao import TorchAOQuantizer
        from kirin.quantizers.base import TorchAOConfig
        
        with check_memory_usage(expected_mem=expected_mem, device=DEFAULT_DEVICE):
            app_logger.info(f"quantizing: {quant_type}")
            quant_config = TorchAOConfig(quant_type=quant_type)
            quantizer = TorchAOQuantizer(quantization_config=quant_config)
            model = LargeModel(quantizer=quantizer)
            model.load_model(LargeModel.SAFETENSORS_PATH)
            x = torch.ones(1, 16384)
            out = model(x)
            self.assertEqual(out.shape, (1, 4096))
            self.assertTrue(out[0, 0].item(), 16384)
            self.assertEqual(next(model.parameters()).device.type, DEFAULT_DEVICE)
            
        # probably not needed since tests are being run in isolation
        # but keeping them just to be safe
        del model
        del quantizer
        del out
        del x
        del quant_config
        ModelMixin.clear_caches()
        MemoryManager.clear_memory()
        
    # torchao plus offloading
    @check_memory_usage(expected_mem=550, device=DEFAULT_DEVICE)
    def test_torchao_low_mem_run(self):
        from kirin.quantizers.torchao.torchao import TorchAOQuantizer
        
        # this offloading pattern only quantizes the full_load modules
        # that stay on the gpu, rest all are used and then offloaded
        with patch.multiple(
                    MemoryManager, 
                    available_memory=lambda device: 2500.0,
                    total_memory=lambda device: 2500.0):
            quantizer = TorchAOQuantizer()
            model = LargeModel(quantizer=quantizer)
            model.load_model(LargeModel.SAFETENSORS_PATH)
            x = torch.ones(1, 16384)
            out = model(x)
            self.assertEqual(out.shape, (1, 4096))
            self.assertTrue(out[0, 0].item(), 16384)
            self.assertEqual(next(model.parameters()).device.type, DEFAULT_DEVICE)