import unittest

from tests.test_utils.common import SimpleModel, check_memory_usage
from kirin.utils.device import MemoryManager, DEFAULT_DEVICE, is_cuda_available, is_mps_available, is_xla_available, is_mps_available


class ModelMetaTest(unittest.TestCase):
    # clear mem / cache before n after each test
    # since we will also be measuring mem usage
    def setUp(self):
        MemoryManager.clear_memory()    
    
    def tearDown(self):
        MemoryManager.clear_memory()
            
    # model should be init on 'meta'
    @check_memory_usage(expected_mem=0)                             # no mem usage on the cpu
    @check_memory_usage(expected_mem=0, device=DEFAULT_DEVICE)      # no mem usage on the gpu
    def test_model_init(self):
        model = SimpleModel()
        self.assertEqual(next(model.parameters()).device.type, "meta")
    
    # normal load should go to the default device
    @check_memory_usage(expected_mem=12, device=DEFAULT_DEVICE)
    def test_model_device(self):
        model = SimpleModel()
        model.load_model(SimpleModel.SIMPLE_MODEL_CKPT_PATH)
        self.assertEqual(next(model.parameters()).device.type, DEFAULT_DEVICE)
    
    # low vram load should go to the cpu
    @check_memory_usage(expected_mem=12)
    def test_model_device_low_vram(self):
        model = SimpleModel()
        model.load_model(SimpleModel.SIMPLE_MODEL_CKPT_PATH, force_low_vram=True)
        self.assertEqual(next(model.parameters()).device.type, "cpu")

    

'''
- memory usage monitor is working correctly
- model loading
    - model loads normally
    - model loads with zero memory
- quantization
    - quantization happens + should happen in the meta (no mem usage)
    - this should speed up the perf , lower mem (check what diffusers is doing)
'''