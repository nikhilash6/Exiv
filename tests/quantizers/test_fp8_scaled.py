import torch
import unittest
import os
import tempfile
from safetensors.torch import save_file

from exiv.quantizers.base import QuantType
from tests.test_utils.common import LargeModel, check_memory_usage, create_large_model_file
from exiv.utils.device import MemoryManager, VRAM_DEVICE, is_cuda_available
from exiv.utils.logging import app_logger
from exiv.config import global_config

has_fp8_support = hasattr(torch, "float8_e4m3fn") and is_cuda_available

@unittest.skipIf(not has_fp8_support, "FP8 requires CUDA and PyTorch 2.1+")
class TorchFP8RunTest(unittest.TestCase):
    def setUp(self):
        create_large_model_file()
        MemoryManager.clear_memory()    
    
    def tearDown(self):
        MemoryManager.clear_memory()

    # TODO: fix the memory estimation
    # @check_memory_usage(expected_mem=3144, device=VRAM_DEVICE)
    def test_fp8_run_and_structure(self):
        global_config.update_config({"normal_load": True})
        quant_type = QuantType.FP8_SCALED
        app_logger.info(f"quantizing: {quant_type.value}")
        
        model = LargeModel(quant_type=quant_type)
        model.load_model(LargeModel.SAFETENSORS_PATH)
        
        # verifying the proper layers were replaced
        linear_layer = model.input_layer
        self.assertEqual(linear_layer.weight.dtype, torch.float8_e4m3fn, "Weights should be quantized to E4M3")
        self.assertTrue(hasattr(linear_layer, "scale_weight"), "Layer must have a scale_weight parameter")
        self.assertEqual(linear_layer.scale_weight.dtype, torch.float32, "Scale should be FP32")
        self.assertEqual(linear_layer.scale_weight.numel(), 1, "Scale should be a scalar (per-tensor)")

        # inference with fp16 input
        x = torch.ones(1, 16384, device=VRAM_DEVICE, dtype=torch.float16)
        out = model(x)
        
        self.assertEqual(out.shape, (1, 4096))
        self.assertNotEqual(out.abs().sum().item(), 0.0)
        self.assertFalse(torch.isnan(out).any())

        del x, out, model
        MemoryManager.clear_memory()
        print("here")

    # @check_memory_usage(expected_mem=3177, device=VRAM_DEVICE)
    # def test_save_and_load_prequantized_fp8(self):
    #     """
    #     Tests the workflow: FP16 Model -> Load & Convert to FP8 -> Save to Disk -> Load FP8 directly
    #     """
    #     quant_type = QuantType.FP8_SCALED
    #     print(f"Quantizing original model with {quant_type}...")
        
    #     # on-the-fly conversion (Case A)
    #     model = LargeModel(quant_type=quant_type)
    #     model.load_model(LargeModel.SAFETENSORS_PATH)
    #     original_scale = model.input_layer.scale_weight.item()
    #     with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as tmp:
    #         save_path = tmp.name
            
    #     try:
    #         print(f"Saving quantized state dict to {save_path}...")
    #         # this should save both 'weight' (fp8) and 'scale_weight' (fp32)
    #         save_file(model.state_dict(), save_path)
    #         del model
    #         MemoryManager.clear_memory()

    #         print("Loading pre-quantized FP8 model...")
    #         # load the saved FP8 model
    #         model_pre = LargeModel(quant_type=quant_type)
    #         model_pre.load_model(save_path)

    #         # verification
    #         linear = model_pre.input_layer
    #         self.assertEqual(linear.weight.dtype, torch.float8_e4m3fn)
    #         loaded_scale = linear.scale_weight.item()
    #         self.assertNotEqual(loaded_scale, 1.0, "Scale weight seems to have defaulted to 1.0!")
    #         self.assertAlmostEqual(original_scale, loaded_scale, places=6, 
    #                               msg="Loaded scale does not match saved scale")

    #         # inference
    #         x = torch.ones(1, 16384).to(VRAM_DEVICE)
    #         out = model_pre(x)
    #         self.assertEqual(out.shape, (1, 4096))
            
    #     finally:
    #         if os.path.exists(save_path):
    #             os.remove(save_path)