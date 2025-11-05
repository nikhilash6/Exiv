# import torch

# import unittest

# from exiv.components.text_vision_encoder.te_t5 import UMT5XXL
# from exiv.components.text_vision_encoder.text_tokenizer import UMTT5XXLTokenizer
# from exiv.utils.device import MemoryManager
# from exiv.utils.file import ensure_model_available

# class TextEncoderTest(unittest.TestCase):
#     def setUp(self):
#         MemoryManager.clear_memory()    
    
#     def tearDown(self):
#         MemoryManager.clear_memory()
    
#     def test_t5xxl_encoder(self):
#         # this will test both the encoder and the tokenizer
#         prompt_list = [
#             # "a photo of a white dog and a blue bird",
#             "a photo of a (white:2) (dog:1) and a ((blue)) (bird:3)"
#         ]

#         res_tokens = []
#         tokenizer = UMTT5XXLTokenizer()
#         for p in prompt_list:
#             tokens, special_tokens = tokenizer.tokenize_with_weights(p)
#             res_tokens.append(tokens)
#         del tokenizer

#         path = "../../test_utils/assets/models/umt5_xxl_fp16.safetensors"
#         t5_xxl = UMT5XXL(
#             model_path=path, 
#             dtype=torch.float16
#         )
#         t5_xxl.load_model(
#             download_url="https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp16.safetensors?download=true"
#         )
#         embed_output = t5_xxl.encode_token_weights(res_tokens[1], special_tokens)   # output, pooled, extra
#         print("embed: ", len(embed_output))