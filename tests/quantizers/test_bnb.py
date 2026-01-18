import torch

import unittest
from parameterized import parameterized
import bitsandbytes as bnb

from exiv.quantizers.base import QuantType
from tests.test_utils.common import LargeModel, check_memory_usage, create_large_model_file
from exiv.utils.device import OFFLOAD_DEVICE, MemoryManager, VRAM_DEVICE, is_cuda_available
from exiv.utils.logging import app_logger
from exiv.config import LOADING_MODE, global_config


@unittest.skipIf(not is_cuda_available, "Only available for cuda devices")
class TorchBNBRunTest(unittest.TestCase):
    def setUp(self):
        create_large_model_file()
        MemoryManager.clear_memory()    
    
    def tearDown(self):
        MemoryManager.clear_memory()
        
    # bnb run
    QUANT_PARAMS = [
        (QuantType.BNB_INT8, 2176),
        (QuantType.BNB_FP4, 2336),
        (QuantType.BNB_NF4, 2336),
    ]
    @parameterized.expand(QUANT_PARAMS)
    def test_bnb_run(self, quant_type, expected_mem):
        with check_memory_usage(expected_mem=expected_mem, device=VRAM_DEVICE):
            global_config.update_config({"normal_load": True})
            app_logger.info(f"quantizing: {quant_type.value}")
            model = LargeModel(quant_type=quant_type)
            model.load_model(LargeModel.SAFETENSORS_PATH)
            x = torch.ones(1, 16384)
            out = model(x)
            self.assertEqual(out.shape, (1, 4096))
            self.assertTrue(out[0, 0].item(), 16384)
            self.assertEqual(next(model.parameters()).device.type, VRAM_DEVICE)

            del x, out, model
    
    # TODO: need a much bigger model for LOW_VRAM testing
    # fp32 is cast to fp16 before quant, thus halving the full precision layer size
    QUANT_PARAMS = [
        (LOADING_MODE.NO_OOM.value, 1536.75, VRAM_DEVICE, OFFLOAD_DEVICE),
        (LOADING_MODE.NORMAL_LOAD.value, 2176, VRAM_DEVICE, VRAM_DEVICE),
    ]
    @parameterized.expand(QUANT_PARAMS)
    def test_multi_step_bnb_int8(self, load_mode, expected_mem, mem_device, expected_device):
        with check_memory_usage(expected_mem=expected_mem, device=mem_device):
            global_config.update_config({load_mode: True})
            model = LargeModel(quant_type=QuantType.BNB_INT8)
            model.load_model(LargeModel.SAFETENSORS_PATH)
            x = torch.ones(1, 16384)
            with torch.no_grad():
                for _ in range(5):
                    out = model(x)
            self.assertEqual(out.shape, (1, 4096))
            self.assertTrue(out[0, 0].item(), 16384)
            self.assertEqual(next(model.parameters()).device.type, expected_device)

            del x, out, model

    @check_memory_usage(expected_mem=2336, device=VRAM_DEVICE)
    def test_save_and_load_prequantized(self):
        import os
        import tempfile
        from safetensors.torch import save_file

        # this performs the "load then quantize" workflow
        quant_type = QuantType.BNB_NF4
        print(f"Quantizing original model with {quant_type}...")
        model = LargeModel(quant_type=quant_type)
        model.load_model(LargeModel.SAFETENSORS_PATH)
        
        # the state_dict now contains the packed 4-bit/8-bit weights
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as tmp:
            save_path = tmp.name
            
        try:
            print(f"Saving quantized state dict to {save_path}...")
            save_file(model.state_dict(), save_path)
            
            # clean up the original model
            del model
            MemoryManager.clear_memory()

            # BNB quantizer detects the quantized keys (e.g. 'bitsandbytes__nf4') 
            # and uses the 'from_prequantized' loading path
            print("Loading pre-quantized model...")
            model_pre = LargeModel(quant_type=quant_type)
            model_pre.load_model(save_path)

            # verifying the output
            x = torch.ones(1, 16384).to(VRAM_DEVICE)
            out = model_pre(x)
            
            self.assertEqual(out.shape, (1, 4096))
            self.assertTrue(out[0, 0].item(), 16384)
            self.assertEqual(next(model_pre.parameters()).device.type, VRAM_DEVICE)
            print("Success: Pre-quantized model loaded and ran.")
            
        finally:
            if os.path.exists(save_path):
                os.remove(save_path)