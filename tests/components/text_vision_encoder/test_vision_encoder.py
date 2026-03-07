import torch

import unittest
from parameterized import parameterized

from exiv.components.cond_registry import get_vision_embeddings
from exiv.components.text_vision_encoder.common import VisionEncoderOutput
from exiv.components.text_vision_encoder.vision_encoder import create_vision_encoder
from exiv.utils.file import MediaProcessor
from exiv.utils.file_path import FilePathData, FilePaths
from exiv.utils.tensor import common_upscale
from exiv.utils.device import VRAM_DEVICE, MemoryManager
from exiv.config import global_config
from exiv.utils.device import is_cuda_available

from tests.test_utils.common import check_memory_usage

# TODO: extend this for other devices as well
@unittest.skipIf(not is_cuda_available, "Only available for cuda devices")
class VisionEncoderTest(unittest.TestCase):
    def setUp(self):
        global_config.auto_download = True
        MemoryManager.clear_memory()    
    
    def tearDown(self):
        MemoryManager.clear_memory()
    
    LOADING_PARAMS = [
        ("no_oom",   {"no_oom": True,  "low_vram": False, "normal_load": False}, 30,  VRAM_DEVICE),   # TODO: look into this, this should be around 600
        ("normal",   {"no_oom": False, "low_vram": False, "normal_load": True},  1206, VRAM_DEVICE),
    ]
    @parameterized.expand(LOADING_PARAMS)
    def test_clip_h_fp16_encoder(self, load_mode, config, expected_mem, expected_device):
        global_config.update_config(config)
        
        print("-- device: ", expected_device)
        with check_memory_usage(expected_mem=expected_mem, device=expected_device):
            height, width = 512, 512
            input_img = MediaProcessor.load_image_list("./tests/test_utils/assets/media/boy_anime.jpg")
            input_img = common_upscale(input_img, height, width)[0]

            cur_model = "CLIP-ViT-H-fp16.safetensors"
            model_path_data: FilePathData = FilePaths.get_path(filename=cur_model, file_type="vision_encoder")
            clip_embed: VisionEncoderOutput = get_vision_embeddings(input_data=input_img, ve_model_filename=model_path_data.path)[0]
            
            self.assertIsNotNone(clip_embed.image_embedding)
            self.assertIsNotNone(clip_embed.intermediate_hidden_states)
            del clip_embed, input_img
            # TODO: add exact output check later