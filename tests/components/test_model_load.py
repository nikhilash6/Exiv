import torch

import unittest

from exiv.quantizers.base import QuantType
from tests.test_utils.common import LargeModel, SimpleModel, check_memory_usage, create_large_model_file
from exiv.utils.device import MemoryManager, CUDA_CC, VRAM_DEVICE, is_cuda_available, is_mps_available, is_xla_available, is_mps_available


class ModelLoadTest(unittest.TestCase):
    # clear mem / cache before n after each test
    # since we will also be measuring mem usage
    def setUp(self):
        create_large_model_file()   # TODO: will generalize this later
        MemoryManager.clear_memory()    
    
    def tearDown(self):
        MemoryManager.clear_memory()
            
    # model should be init on 'meta'
    @check_memory_usage(expected_mem=0)                          # no mem usage on the cpu
    @check_memory_usage(expected_mem=0, device=VRAM_DEVICE)      # no mem usage on the gpu
    def test_model_init(self):
        model = SimpleModel()
        self.assertEqual(next(model.parameters()).device.type, "meta")
    
    # normal load should go to the cpu (loaded into the vram during forward)
    @check_memory_usage(expected_mem=0, device="cpu")
    def test_model_device(self):
        model = SimpleModel()
        model.load_model(SimpleModel.CKPT_PATH)
        self.assertEqual(next(model.parameters()).device.type, "cpu")
    
    # test different kinds of format loading
    def test_model_formats(self):
        for model_path in SimpleModel.ALL_MODEL_PATHS:
            model = SimpleModel()
            model.load_model(model_path)
            self.assertEqual(next(model.parameters()).device.type, "cpu")
            MemoryManager.clear_memory()
            
    # test large model loading
    def test_large_model_load(self):
        for model_path in LargeModel.ALL_MODEL_PATHS:
            model = LargeModel()
            model.load_model(model_path)
            self.assertEqual(next(model.parameters()).device.type, "cpu")
            MemoryManager.clear_memory()

    # testing bnb quantization
    def test_bnb_model_quant(self):
        model = SimpleModel(quant_type=QuantType.BNB_FP4)
        model.load_model(SimpleModel.CKPT_PATH)
        self.assertEqual(next(model.parameters()).device.type, "cpu")
