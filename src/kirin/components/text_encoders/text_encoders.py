import torch
from torch import Tensor
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5Tokenizer


class TextEncoder:
    def __init__(self, *args, **kwargs):
        pass

    def forward(self, text: list[str]) -> Tensor:
        assert hasattr(self, 'tokenizer') and hasattr(self, 'encoder'), "TextEncoder not initialized!"
        
        # tokenize
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=False,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt", 
        )
        
        # encode
        outputs = self.hf_module(
            input_ids=batch_encoding["input_ids"].to(self.hf_module.device),
            attention_mask=None,
            output_hidden_states=False,
        )


class SD_ClipL(TextEncoder):
    def __init__(self, max_length: int = 77, torch_dtype = torch.bfloat16):
        '''
        vit - vision transformer
        large - large model (428M) [h - 986M, g - 1.9B]
        patch14 - 14x14 patch
        aka "clip-l"
        '''
        model_name = "openai/clip-vit-large-patch14"
        self.tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(model_name, max_length=max_length)
        self.encoder: CLIPTextModel = CLIPTextModel.from_pretrained(model_name, torch_dtype=torch_dtype)

        
class T5XXL(TextEncoder):
    def __init__(self, max_length: int = 512, torch_dtype = torch.bfloat16):
        model_name = "google/t5-v1_1-xxl"
        self.tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained(model_name, max_length=max_length)
        self.encoder: T5EncoderModel = T5EncoderModel.from_pretrained(model_name, torch_dtype=torch_dtype)


class Flux_Schnell_T5(T5XXL):
    def __init__(self, torch_dtype = torch.bfloat16):
        super().__init__(max_length=77, torch_dtype=torch_dtype)