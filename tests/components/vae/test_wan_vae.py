import torch
import torch.nn.functional as F
import torchvision

import os
import unittest
from parameterized import parameterized

from exiv.components.vae.wan_vae import Wan21VAE
from exiv.components.vae.wan_vae22 import Wan22VAE
from exiv.model_utils.helper_methods import move_model
from exiv.utils.file import MediaProcessor, ensure_model_availability
from exiv.utils.file_path import FilePathData, FilePaths
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
        ("no_oom",   {"no_oom": True,  "low_vram": False, "normal_load": False}, 1840, VRAM_DEVICE, True),       # this will force revert to normal_load mode
        ("normal",   {"no_oom": False, "low_vram": False, "normal_load": True},  1840, VRAM_DEVICE, True),
        ("normal",   {"no_oom": False, "low_vram": False, "normal_load": True},  5997, VRAM_DEVICE, False),
    ]
    @parameterized.expand(LOADING_PARAMS)
    def test_wan21_vae(self, load_mode, config, expected_mem, expected_device, use_tiling):
        global_config.update_config(config)
        
        with check_memory_usage(expected_mem=expected_mem, device=expected_device):
            height, width, frame_count = 512, 512, 81
            input_img = MediaProcessor.load_image_list("./tests/test_utils/assets/media/boy_anime.jpg")
            input_img = common_upscale(input_img, height, width)[0]    # B, C, H, W
            
            image = torch.ones((frame_count, height, width, input_img.shape[1]), device=input_img.device, dtype=input_img.dtype) * 0.5
            # (T, C, H, W) --> (T, H, W, C)
            input_img = input_img.permute(0, 2, 3, 1)
            image[:input_img.shape[0]] = input_img      # first conditionals are replaced by the input_img
            # (T, H, W, C) -> (1, C, T, H, W)
            image = image.permute(3, 0, 1, 2).unsqueeze(0)

            cur_model = "wan_2_1_vae.safetensors"
            model_path_data: FilePathData = FilePaths.get_path(filename=cur_model, file_type="vae")
            model_path = ensure_model_availability(model_path=model_path_data.path, download_url=model_path_data.url)
            
            wan_vae = Wan21VAE(use_tiling=use_tiling)
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
            decoded_image = decoded_image.to(VRAM_DEVICE)
            mse_loss = F.mse_loss(decoded_image, image)
            self.assertLess(mse_loss.item(), 1e-04)
            self.assertEqual(image.shape, decoded_image.shape)

    # TODO: low_vram / no_oom should automatically trigger use_tiling = False
    LOADING_PARAMS = [
        ("no_oom",   {"no_oom": True,  "low_vram": False, "normal_load": False}, 4407, VRAM_DEVICE, True),       # this will force revert to normal_load mode
        ("normal",   {"no_oom": False, "low_vram": False, "normal_load": True},  4407, VRAM_DEVICE, True),       # TODO: decoding cache increases vram, look into how this can be reduced
        ("normal",   {"no_oom": False, "low_vram": False, "normal_load": True},  8595, VRAM_DEVICE, False),      # TODO: cross check the vram usage with the original implementation
    ]
    @parameterized.expand(LOADING_PARAMS)
    def test_wan22_vae(self, load_mode, config, expected_mem, expected_device, use_tiling):
        global_config.update_config(config)
        
        with check_memory_usage(expected_mem=expected_mem, device=expected_device):
            height, width, frame_count = 512, 512, 81
            input_img = MediaProcessor.load_image_list("./tests/test_utils/assets/media/boy_anime.jpg")
            input_img = common_upscale(input_img, height, width)[0]    # B, C, H, W
            
            image = torch.ones((frame_count, height, width, input_img.shape[1]), device=input_img.device, dtype=input_img.dtype) * 0.5
            # (T, C, H, W) --> (T, H, W, C)
            input_img = input_img.permute(0, 2, 3, 1)
            image[:input_img.shape[0]] = input_img      # first conditionals are replaced by the input_img
            # (T, H, W, C) -> (1, C, T, H, W)
            image = image.permute(3, 0, 1, 2).unsqueeze(0)

            cur_model = "wan_2_2_vae.safetensors"
            model_path_data: FilePathData = FilePaths.get_path(filename=cur_model, file_type="vae")
            model_path = ensure_model_availability(model_path=model_path_data.path, download_url=model_path_data.url)
            
            wan_vae = Wan22VAE(use_tiling=use_tiling)
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
            decoded_image = decoded_image.to(VRAM_DEVICE)
            mse_loss = F.mse_loss(decoded_image, image)
            self.assertLess(mse_loss.item(), 1e-04)
            self.assertEqual(image.shape, decoded_image.shape)


    # NOTE: used in manually / visually checking outputs
    # def test_wan_vae_video(self):
        
    #     # change the input video here
    #     input_video_path = "./512x512_4s.mp4"
    #     if not os.path.exists(input_video_path):
    #         raise FileNotFoundError("file not found bruh")
    #     save_path = "output_recon.mp4"
        
    #     # read_video returns (T, H, W, C) in [0, 255]
    #     video_frames, _, info = torchvision.io.read_video(input_video_path, pts_unit='sec')
    #     print("shape here: ", video_frames.shape)
    #     fps = info.get("video_fps", info.get("fps", 24.0))
        
    #     # Preprocess: (T, H, W, C) -> (B, C, T, H, W) and Normalize to [0, 1]
    #     video = video_frames.permute(3, 0, 1, 2).unsqueeze(0) # (1, C, T, H, W)
    #     video = video.to(device=VRAM_DEVICE, dtype=torch.float16) / 255.0
        
    #     # Resize if necessary (optional, ensures 512x512)
    #     # video = torch.nn.functional.interpolate(video, size=(video.shape[2], 512, 512))
    #     B, C, T, H, W = video.shape

    #     cur_model = "wan_2_2_vae.safetensors"
    #     model_path_data: FilePathData = FilePaths.get_path(filename=cur_model, file_type="vae")
    #     model_path = ensure_model_availability(model_path=model_path_data.path, download_url=model_path_data.url)
        
    #     wan_vae = Wan22VAE(dtype=torch.float16)
    #     wan_vae.load_model(model_path=model_path)
    #     move_model(wan_vae, VRAM_DEVICE)
    #     # Encode
    #     concat_latent_image = wan_vae.encode(video)
    #     MemoryManager.clear_memory()
    #     app_logger.debug("encoding complete")
        
    #     # Decode
    #     decoded_image = wan_vae.decode(concat_latent_image, input_shape=(W, H, T))
    #     MemoryManager.clear_memory()
    #     app_logger.debug("decoding complete")
    #     del wan_vae

    #     # Postprocess: (B, C, T, H, W) -> (T, H, W, C) and Denormalize to [0, 255]
    #     output_tensor = decoded_image.squeeze(0).permute(1, 2, 3, 0).clamp(0, 1)
    #     output_tensor = (output_tensor * 255).to(torch.uint8).cpu()
        
    #     torchvision.io.write_video(save_path, output_tensor, fps=fps)
    #     print(f"Saved reconstructed video to {os.path.abspath(save_path)}")
        
    #     # Simple shape check instead of MSE since exact pixel match is unlikely with VAE compression
    #     self.assertEqual(video.shape, decoded_image.shape)