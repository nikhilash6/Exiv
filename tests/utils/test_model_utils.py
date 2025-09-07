import torch
from torch import nn
import unittest


from tests.test_utils.models import SimpleModel
from kirin.utils.device import is_cuda_available, is_mps_available, is_xla_available, is_mps_available

class ModelMetaTest(unittest.TestCase):
    # model should be init on 'meta'
    def test_model_init(self):
        model = SimpleModel()
        self.assertEqual(next(model.parameters()).device, torch.device("meta"))
    
    # # normal load should go to the device / gpu
    # def test_model_device(self):
    #     model = SimpleModel()
    #     for param in model.parameters():
    #         self.assertEqual(param.device.type, 'mps')
    
    # # low vram load should go to the cpu
    # def test_model_device_low_vram(self):
    #     model = SimpleModel()
    #     for param in model.parameters():
    #         self.assertEqual(param.device.type, 'cpu')
            
    # def test_gpu_memory_consumption(self):
    #     if not torch.cuda.is_available():
    #         self.skipTest("No GPU available.")

    #     # Clear any existing GPU cache
    #     torch.cuda.empty_cache()
    #     torch.cuda.reset_peak_memory_stats()

    #     # Instantiate the model and check memory consumption
    #     model = SimpleModel()
    #     initial_memory = torch.cuda.memory_allocated()
    #     self.assertEqual(initial_memory, 0, "GPU memory should not be consumed at instantiation.")

    #     # Perform a forward pass and check memory consumption
    #     input_tensor = torch.randn(1, 10).to('cuda')
    #     _ = model(input_tensor)
    #     final_memory = torch.cuda.memory_allocated()
    #     self.assertGreater(final_memory, 0, "GPU memory should be consumed after forward pass.")
    

'''
- memory usage monitor is working correctly
- model loading
    - model loads normally
    - model loads with zero memory
- quantization
    - quantization happens + should happen in the meta (no mem usage)
    - this should speed up the perf , lower mem (check what diffusers is doing)
'''

'''
- instead of passing "cpu", you should pass the default device
    or take user input
- add a method to completely offload the model (something like 'complete_offload')
'''