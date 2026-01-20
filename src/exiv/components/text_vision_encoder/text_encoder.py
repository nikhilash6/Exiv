from typing import Optional, Tuple

import torch

from exiv.utils.file import ensure_model_availability
from exiv.utils.file_path import FilePathData, FilePaths
from .te_t5 import T5XXL, UMT5XXL
from .encoder_base import TextEncoder
from .text_tokenizer import SDTokenizer, UMTT5XXLTokenizer
from ..enum import TextEncoderType
from ...model_utils.model_mixin import ModelMixin, get_state_dict
from ...utils.device import OFFLOAD_DEVICE, VRAM_DEVICE
from ...utils.logging import app_logger

# TODO: add respective tokenizers as different models are incorporated
TE_TYPE_CLS_MAP = {
    TextEncoderType.T5_XXL.value: (T5XXL, UMTT5XXLTokenizer),
    TextEncoderType.UMT5_XXL.value: (UMT5XXL, UMTT5XXLTokenizer),
}

def te_type(sd):
    if "text_model.encoder.layers.30.mlp.fc1.weight" in sd:
        return TextEncoderType.CLIP_G.value
    if "text_model.encoder.layers.22.mlp.fc1.weight" in sd:
        return TextEncoderType.CLIP_H.value
    if "text_model.encoder.layers.0.mlp.fc1.weight" in sd:
        return TextEncoderType.CLIP_L.value
    if "encoder.block.23.layer.1.DenseReluDense.wi_1.weight" in sd:
        weight = sd["encoder.block.23.layer.1.DenseReluDense.wi_1.weight"]
        if weight.shape[-1] == 4096:
            if "shared.weight" in sd and sd["shared.weight"].shape[0] > 100000:
                return TextEncoderType.UMT5_XXL.value
            return TextEncoderType.T5_XXL.value
        elif weight.shape[-1] == 2048:
            return TextEncoderType.T5_XL.value
    if 'encoder.block.23.layer.1.DenseReluDense.wi.weight' in sd:
        return TextEncoderType.T5_XXL_OLD.value
    if "encoder.block.0.layer.0.SelfAttention.k.weight" in sd:
        weight = sd['encoder.block.0.layer.0.SelfAttention.k.weight']
        if weight.shape[0] == 384:
            return TextEncoderType.BYT5_SMALL_GLYPH.value
        return TextEncoderType.T5_BASE.value
    if 'model.layers.0.post_feedforward_layernorm.weight' in sd:
        return TextEncoderType.GEMMA_2_2B.value
    if 'model.layers.0.self_attn.k_proj.bias' in sd:
        weight = sd['model.layers.0.self_attn.k_proj.bias']
        if weight.shape[0] == 256:
            return TextEncoderType.QWEN25_3B.value
        if weight.shape[0] == 512:
            return TextEncoderType.QWEN25_7B.value
    if "model.layers.0.post_attention_layernorm.weight" in sd:
        return TextEncoderType.LLAMA3_8.value
    return None

class TextPipeline:
    def __init__(self, text_encoder: TextEncoder, tokenizer: SDTokenizer):
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
    
    def load_model(self):
        self.text_encoder.load_model()
        
    def encode(self, text):
        tokens, special_tokens = self.tokenize(text)
        out = self.text_encoder.encode_token_weights(tokens, special_tokens)
        return out
    
    def tokenize(self, text, return_word_ids=False) -> Tuple:
        # returns (tokens, special_tokens)
        return self.tokenizer.tokenize_with_weights(text, return_word_ids)

def create_text_pipeline(
    filename: Optional[str] = None, 
    model_type: Optional[str] = None, 
    dtype=torch.float16
) -> TextPipeline:
    assert filename is not None or model_type is not None, "atleast one of filename or model_type is required to create the text encoder"
    
    model_path = None   # NOTE: TE models use the default model from their config
    if filename:  # NOTE: filename is given priority
        model_path_data: FilePathData = FilePaths.get_path(filename=filename, file_type="text_encoder")
        model_path: str = ensure_model_availability(model_path_data.path, model_path_data.url)
        state_dict = get_state_dict(model_path, device=OFFLOAD_DEVICE)
        te_type: str = te_type(state_dict)
        del state_dict
    else:
        te_type: str = model_type
    
    if te_type is None or te_type not in TE_TYPE_CLS_MAP:
        raise Exception("Text encoder not supported")
    
    te_cls, tokenizer_cls = TE_TYPE_CLS_MAP[te_type]
    return TextPipeline(te_cls(model_path, dtype=dtype), tokenizer_cls())