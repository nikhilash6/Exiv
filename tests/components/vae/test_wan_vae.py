import torch
import torch.nn.functional as F

import unittest
from parameterized import parameterized

from exiv.components.vae.wan_vae import Wan21VAE
from exiv.model_utils.helper_methods import move_model
from exiv.utils.file import ImageProcessor, ensure_model_available
from exiv.utils.tensor import common_upscale
from exiv.utils.device import VRAM_DEVICE, MemoryManager
from exiv.config import LOADING_MODE, global_config
from exiv.utils.device import is_cuda_available
from exiv.utils.logging import app_logger

from tests.test_utils.common import check_memory_usage

# TODO: extend this for other devices as well
@unittest.skipIf(not is_cuda_available, "Only available for cuda devices")
class VisionEncoderTest(unittest.TestCase):
    def setUp(self):
        MemoryManager.clear_memory()    
    
    def tearDown(self):
        MemoryManager.clear_memory()
    
    LOADING_PARAMS = [
        ("no_oom",   {"no_oom": True,  "low_vram": False, "normal_load": False}, 2106,  VRAM_DEVICE),       # this will force revert to normal_load mode
        ("normal",   {"no_oom": False, "low_vram": False, "normal_load": True},  2106, VRAM_DEVICE),        # TODO: decoding cache increases vram, look into how this can be reduced
    ]
    @parameterized.expand(LOADING_PARAMS)
    def test_wan_vae(self, load_mode, config, expected_mem, expected_device):
        global_config.update_config(config)
        
        with check_memory_usage(expected_mem=expected_mem, device=expected_device):
            height, width, frame_count = 512, 512, 81
            input_img = ImageProcessor.load_image_list("./tests/test_utils/assets/media/test.jpg")
            input_img = common_upscale(input_img, height, width)    # B, C, H, W
            
            image = torch.ones((frame_count, height, width, input_img.shape[1]), device=input_img.device, dtype=input_img.dtype) * 0.5
            # (T, C, H, W) --> (T, H, W, C)
            input_img = input_img.permute(0, 2, 3, 1)
            image[:input_img.shape[0]] = input_img      # first conditionals are replaced by the input_img
            # (T, H, W, C) -> (1, C, T, H, W)
            image = image.permute(3, 0, 1, 2).unsqueeze(0)

            download_url = "https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P-Diffusers/resolve/main/vae/diffusion_pytorch_model.safetensors?download=true"
            model_path = "./tests/test_utils/assets/models/wan_2_1_vae.safetensors"
            model_path = ensure_model_available(model_path=model_path, download_url=download_url)
            
            wan_vae = Wan21VAE()
            wan_vae.load_model(model_path=model_path)
            move_model(wan_vae, VRAM_DEVICE)
            
            B, C, T, H, W = image.shape
            # encode the entire sequence, image: (B, C, T, H, W)
            concat_latent_image = wan_vae.encode(image)
            MemoryManager.clear_memory()
            app_logger.debug("encoding complete")
            decoded_image = wan_vae.decode(concat_latent_image, input_shape=(W, H, T))
            MemoryManager.clear_memory()
            app_logger.debug("decoding complete")
            del wan_vae
            
            image = image.to(VRAM_DEVICE)
            mse_loss = F.mse_loss(decoded_image, image)
            self.assertLess(mse_loss.item(), 1e-04)
            self.assertEqual(image.shape, decoded_image.shape)


    LOADING_PARAMS = [
        ("normal",   {"no_oom": False, "low_vram": False, "normal_load": True},  2346.53, VRAM_DEVICE),
    ]
    @parameterized.expand(LOADING_PARAMS)
    def test_wan_vae_video(self, load_mode, config, expected_mem, expected_device):
        global_config.update_config(config)
        
        with check_memory_usage(expected_mem=expected_mem, device=expected_device):
            height, width = 512, 512
            frame_count = 81 
            
            # synthetic video data: (1, 3, 81, 512, 512)
            video = torch.rand((1, 3, frame_count, height, width), device=VRAM_DEVICE, dtype=torch.float32)

            download_url = "https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P-Diffusers/resolve/main/vae/diffusion_pytorch_model.safetensors?download=true"
            model_path = "./tests/test_utils/assets/models/wan_2_1_vae.safetensors"
            model_path = ensure_model_available(model_path=model_path, download_url=download_url)
            
            wan_vae = Wan21VAE()
            wan_vae.load_model(model_path=model_path)
            move_model(wan_vae, VRAM_DEVICE)
            
            # Encode: video is (B, C, T, H, W)
            concat_latent_image = wan_vae.encode(video)
            MemoryManager.clear_memory()
            app_logger.debug("encoding complete")
            
            # Decode
            decoded_image = wan_vae.decode(concat_latent_image, input_shape=(width, height, frame_count))
            MemoryManager.clear_memory()
            app_logger.debug("decoding complete")
            del wan_vae
            
            self.assertEqual(video.shape, decoded_image.shape)
            mse_loss = F.mse_loss(decoded_image, video)
            self.assertLess(mse_loss.item(), 1e-02)