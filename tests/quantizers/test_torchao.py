import torch

import unittest
from unittest.mock import patch

from tests.test_utils.common import LargeModel, SimpleModel, check_memory_usage, create_large_model_file
from kirin.utils.device import MemoryManager, DEFAULT_DEVICE, print_mem_usage
from kirin.utils.logging import app_logger
from kirin.utils.model_utils import ModelMixin

class ModelRunTest(unittest.TestCase):
    def setUp(self):
        create_large_model_file()
        MemoryManager.clear_memory()    
    
    def tearDown(self):
        MemoryManager.clear_memory()
        
    # torchao run
    # @check_memory_usage(expected_mem=1130, device=DEFAULT_DEVICE)
    def test_torchao_run(self):
        from kirin.quantizers.torchao.torchao import TorchAOQuantizer
        from kirin.quantizers.base import TorchAOConfig
        
        quant_dtype_list = [
            "fp8dq_e4m3",
            "int8dq_int4",
            "int4dq_int4",
            "int8dq_int8",
            "int8",
            "fp8wo_e4m3",
            "fp8wo_e5m2",
        ]
        
        for qd in quant_dtype_list:
            app_logger.info(f"quantizing: {qd}")
            quant_config = TorchAOConfig(quant_type=qd)
            quantizer = TorchAOQuantizer(quantization_config=quant_config)
            model = LargeModel(quantizer=quantizer)
            model = quantizer.pre_process(model)
            model.load_model(LargeModel.SAFETENSORS_PATH)
            model = quantizer.post_process(model)
            x = torch.ones(1, 16384)
            out = model(x)
            self.assertEqual(out.shape, (1, 4096))
            self.assertTrue(out[0, 0].item(), 16384)
            self.assertEqual(next(model.parameters()).device.type, DEFAULT_DEVICE)
            
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
            model = quantizer.pre_process(model)
            model.load_model(LargeModel.SAFETENSORS_PATH)
            model = quantizer.post_process(model)
            x = torch.ones(1, 16384)
            out = model(x)
            self.assertEqual(out.shape, (1, 4096))
            self.assertTrue(out[0, 0].item(), 16384)
            self.assertEqual(next(model.parameters()).device.type, DEFAULT_DEVICE)