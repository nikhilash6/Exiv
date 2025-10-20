import unittest

from exiv.components.text_vision_encoder.te_t5 import T5XXL
from exiv.components.text_vision_encoder.text_tokenizer import UMTT5XXLTokenizer
from exiv.utils.device import MemoryManager

class TextEncoderTest(unittest.TestCase):
    def setUp(self):
        MemoryManager.clear_memory()    
    
    def tearDown(self):
        MemoryManager.clear_memory()
    
    def test_t5xxl_encoder(self):
        # this will test both the encoder and the tokenizer
        prompt_list = [
            "a photo of a white dog and a blue bird",
            "a photo of a (white:2) (dog:1) and a ((blue)) (bird:3)"
        ]
        
        tokenizer = UMTT5XXLTokenizer()
        res_tokens = []
        for p in prompt_list:
            tokens = tokenizer.tokenize_with_weights(p)
            res_tokens.append(tokens)
        
        path = "../../test_utils/assets/models/umt5_xxl_fp8_e4m3fn_scaled.safetensors"
        t5_xxl = T5XXL(model_path=path)
        embeds = t5_xxl.encode_token_weights(res_tokens[0])
        
        print('embed shape: ', embeds.shape)
        
        