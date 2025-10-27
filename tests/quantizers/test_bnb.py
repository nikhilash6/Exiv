import torch

import unittest
from parameterized import parameterized

from exiv.quantizers.base import QuantType
from tests.test_utils.common import LargeModel, check_memory_usage, create_large_model_file
from exiv.utils.device import MemoryManager, VRAM_DEVICE, is_cuda_available
from exiv.utils.logging import app_logger

@unittest.skipIf(not is_cuda_available, "Only available for cuda devices")
class TorchBNBRunTest(unittest.TestCase):
    def setUp(self):
        create_large_model_file()
        MemoryManager.clear_memory()    
    
    def tearDown(self):
        MemoryManager.clear_memory()
        
    # bnb run
    """
    NOTE: these tests are for on-the-fly conversion of non-quantized models. from the experiments
    OTF conversion to fp8 takes more peak mem than is needed (based on llm.int8 paper, needs the entire model to find the outliers)..
    while fp4 works correctly.
    """
    QUANT_PARAMS = [
        (QuantType.BNB_FP8, 5888.41),
        (QuantType.BNB_FP4, 2640.50),
        (QuantType.BNB_NF4, 2640.50),
    ]
    @parameterized.expand(QUANT_PARAMS)
    def test_bnb_run(self, quant_type, expected_mem):
        from exiv.quantizers.bnb.bnb import BNBQuantizer
        
        with check_memory_usage(expected_mem=expected_mem, device=VRAM_DEVICE):
            
            app_logger.info(f"quantizing: {quant_type.value}")
            model = LargeModel(quant_type=quant_type)
            model.load_model(LargeModel.SAFETENSORS_PATH)
            x = torch.ones(1, 16384)
            out = model(x)
            self.assertEqual(out.shape, (1, 4096))
            self.assertTrue(out[0, 0].item(), 16384)
            self.assertEqual(next(model.parameters()).device.type, VRAM_DEVICE)
            
            model.cpu()
            del x, out, model

   