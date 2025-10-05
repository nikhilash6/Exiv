import torch

import unittest

from tests.test_utils.common import LargeModel, SimpleModel, check_memory_usage, create_large_model_file
from exiv.utils.device import MemoryManager, CUDA_CC, DEFAULT_DEVICE, is_cuda_available, is_mps_available, is_xla_available, is_mps_available

class ModelLoadTest(unittest.TestCase):
    # clear mem / cache before n after each test
    # since we will also be measuring mem usage
    def setUp(self):
        create_large_model_file()   # TODO: will generalize this later
        MemoryManager.clear_memory()    
    
    def tearDown(self):
        MemoryManager.clear_memory()
            
    # model should be init on 'meta'
    @check_memory_usage(expected_mem=0)                             # no mem usage on the cpu
    @check_memory_usage(expected_mem=0, device=DEFAULT_DEVICE)      # no mem usage on the gpu
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
    @unittest.skip("not fixed")
    def test_bnb_model_quant(self):
        from exiv.quantizers.bnb.bnb import BNBQuantizer
        
        quantizer = BNBQuantizer()
        model = SimpleModel()
        model.load_model(SimpleModel.CKPT_PATH)
        self.assertEqual(next(model.parameters()).device.type, DEFAULT_DEVICE)
    
    # testing torchao quantization
    @unittest.skipIf(CUDA_CC < 89, "Compute capability > 89 required")
    def test_torchao_model_quant(self):
        from exiv.quantizers.torchao.torchao import TorchAOQuantizer
        
        quantizer = TorchAOQuantizer()
        model = SimpleModel(quantizer=quantizer)
        model.load_model(SimpleModel.CKPT_PATH)
        self.assertEqual(next(model.parameters()).device.type, "cpu")