import torch

import unittest
from unittest.mock import patch
from parameterized import parameterized

from tests.test_utils.common import LargeModel, SimpleModel, check_memory_usage, create_large_model_file
from exiv.utils.device import MemoryManager, VRAM_DEVICE, is_cuda_available, CUDA_CC
from exiv.utils.logging import app_logger
from exiv.model_utils.model_mixin import ModelMixin
from exiv.model_utils.model_loader import RESERVED_MEM

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
        ("fp8dq_e4m3", 7168.25),    # these peak mem usage is temporary, since we are doing on the fly conversion
        ("int8dq_int4", 6921.38),
        ("int4dq_int4", 6409.56),
        ("int8dq_int8", 6665.56),
        ("int8", 6665.94),
        ("fp8wo_e4m3", 7177.56),
        ("fp8wo_e5m2", 7177.56),
    ]
    @parameterized.expand(QUANT_PARAMS)
    def test_torchao_run(self, quant_type, expected_mem):
        from exiv.quantizers.torchao.torchao import TorchAOQuantizer
        from exiv.quantizers.base import TorchAOConfig
        
        with check_memory_usage(expected_mem=expected_mem, device=VRAM_DEVICE):
            app_logger.info(f"quantizing: {quant_type}")
            quant_config = TorchAOConfig(quant_type=quant_type)
            quantizer = TorchAOQuantizer(quantization_config=quant_config)
            model = LargeModel(quantizer=quantizer)
            model.load_model(LargeModel.SAFETENSORS_PATH)
            x = torch.ones(1, 16384)
            out = model(x)
            self.assertEqual(out.shape, (1, 4096))
            self.assertTrue(out[0, 0].item(), 16384)
            self.assertEqual(next(model.parameters()).device.type, VRAM_DEVICE)
            
            model.clear_cache()
            del model
            del quantizer
            del out
            del x
            del quant_config
        
        
    # torchao plus offloading
    @check_memory_usage(expected_mem=2057.44, device=VRAM_DEVICE)
    def test_torchao_low_mem_run(self):
        from exiv.quantizers.torchao.torchao import TorchAOQuantizer
        
        # this offloading pattern only quantizes the full_load modules
        # that stay on the gpu, rest all are used and then offloaded
        with patch.multiple(
                    MemoryManager, 
                    available_memory=lambda device: RESERVED_MEM + 2500.0,
                    total_memory=lambda device: RESERVED_MEM + 2500.0):
            quantizer = TorchAOQuantizer()
            model = LargeModel(quantizer=quantizer)
            model.load_model(LargeModel.SAFETENSORS_PATH)
            x = torch.ones(1, 16384)
            out = model(x)
            self.assertEqual(out.shape, (1, 4096))
            self.assertTrue(out[0, 0].item(), 16384)
            # device can be anything, depending on how the model was split
            # self.assertEqual(next(model.parameters()).device.type, VRAM_DEVICE)

            del x, out, model