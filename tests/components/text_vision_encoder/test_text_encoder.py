import torch

import unittest
from parameterized import parameterized

from exiv.components.text_vision_encoder.te_t5 import UMT5XXL
from exiv.components.text_vision_encoder.text_tokenizer import UMTT5XXLTokenizer
from exiv.model_patching.debug_hook import add_debug_hooks
from exiv.model_utils.model_mixin import move_model
from exiv.utils.device import VRAM_DEVICE, MemoryManager
from exiv.config import global_config
from exiv.utils.device import is_cuda_available

from tests.test_utils.common import check_memory_usage

# TODO: extend this for other devices as well
@unittest.skipIf(not is_cuda_available, "Only available for cuda devices")
class TextEncoderTest(unittest.TestCase):
    def setUp(self):
        MemoryManager.clear_memory()    
    
    def tearDown(self):
        MemoryManager.clear_memory()
    
    LOADING_PARAMS = [
        ("no_oom",   {"no_oom": True,  "low_vram": False, "normal_load": False}, 2012.01,  VRAM_DEVICE),
        # ("normal",   {"no_oom": False, "low_vram": False, "normal_load": True},  11268.5, VRAM_DEVICE),
    ]
    @parameterized.expand(LOADING_PARAMS)
    def test_t5xxl_encoder(self, load_mode, config, expected_mem, expected_device):
        global_config.update_config(config)
        
        with check_memory_usage(expected_mem=expected_mem, device=expected_device):
            # this will test both the encoder and the tokenizer
            prompt_list = [
                "a photo of a white dog and a blue bird" * 100,
                # "a photo of a (white:2) (dog:1) and a ((blue)) (bird:3)"
            ]

            res_tokens = []
            tokenizer = UMTT5XXLTokenizer()
            for p in prompt_list:
                tokens, special_tokens = tokenizer.tokenize_with_weights(p)
                res_tokens.extend(tokens)
            del tokenizer

            path = "./tests/test_utils/assets/models/umt5_xxl_fp16.safetensors"
            t5_xxl = UMT5XXL(
                model_path=path, 
                dtype=torch.float16
            )
            add_debug_hooks(t5_xxl)
            t5_xxl.load_model(
                download_url="https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp16.safetensors?download=true"
            )
            embed_output = t5_xxl.encode_token_weights(res_tokens, special_tokens)   # output, pooled, extra
            print("embed: ", len(embed_output))
            
            del t5_xxl, embed_output
            # TODO: add check for output the correctness