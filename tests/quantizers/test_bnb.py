import torch

import unittest
from parameterized import parameterized

from exiv.quantizers.base import QuantType
from tests.test_utils.common import LargeModel, check_memory_usage, create_large_model_file
from exiv.utils.device import MemoryManager, VRAM_DEVICE, is_cuda_available
from exiv.utils.logging import app_logger
from exiv.config import global_config

@unittest.skipIf(not is_cuda_available, "Only available for cuda devices")
class TorchBNBRunTest(unittest.TestCase):
    def setUp(self):
        create_large_model_file()
        MemoryManager.clear_memory()    
    
    def tearDown(self):
        MemoryManager.clear_memory()
        
    # bnb run
    QUANT_PARAMS = [
        # (QuantType.BNB_FP8, 2176),
        (QuantType.BNB_FP4, 2336),
        (QuantType.BNB_NF4, 2336),
    ]
    @parameterized.expand(QUANT_PARAMS)
    def test_bnb_run(self, quant_type, expected_mem):
        with check_memory_usage(expected_mem=expected_mem, device=VRAM_DEVICE):
            global_config.update_config({"normal_load": True})
            app_logger.info(f"quantizing: {quant_type.value}")
            torch.cuda.set_device(0)
            model = LargeModel(quant_type=quant_type)
            model.load_model(LargeModel.SAFETENSORS_PATH)
            x = torch.ones(1, 16384)
            out = model(x)
            self.assertEqual(out.shape, (1, 4096))
            self.assertTrue(out[0, 0].item(), 16384)
            self.assertEqual(next(model.parameters()).device.type, VRAM_DEVICE)
            
            model.cpu()
            del x, out, model

   