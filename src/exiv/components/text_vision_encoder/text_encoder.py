from typing import Tuple
from .te_t5 import T5XXL
from .encoder_base import TextEncoder
from .text_tokenizer import UMTT5XXLTokenizer
from ..enum import TextEncoderType
from ...model_utils.model_mixin import ModelMixin
from ...utils.device import VRAM_DEVICE
from ...utils.logging import app_logger

TE_TYPE_CLS_MAP = {
    TextEncoderType.T5_XXL: T5XXL
}

def te_type(sd):
    if "text_model.encoder.layers.30.mlp.fc1.weight" in sd:
        return TextEncoderType.CLIP_G
    if "text_model.encoder.layers.22.mlp.fc1.weight" in sd:
        return TextEncoderType.CLIP_H
    if "text_model.encoder.layers.0.mlp.fc1.weight" in sd:
        return TextEncoderType.CLIP_L
    if "encoder.block.23.layer.1.DenseReluDense.wi_1.weight" in sd:
        weight = sd["encoder.block.23.layer.1.DenseReluDense.wi_1.weight"]
        if weight.shape[-1] == 4096:
            return TextEncoderType.T5_XXL
        elif weight.shape[-1] == 2048:
            return TextEncoderType.T5_XL
    if 'encoder.block.23.layer.1.DenseReluDense.wi.weight' in sd:
        return TextEncoderType.T5_XXL_OLD
    if "encoder.block.0.layer.0.SelfAttention.k.weight" in sd:
        weight = sd['encoder.block.0.layer.0.SelfAttention.k.weight']
        if weight.shape[0] == 384:
            return TextEncoderType.BYT5_SMALL_GLYPH
        return TextEncoderType.T5_BASE
    if 'model.layers.0.post_feedforward_layernorm.weight' in sd:
        return TextEncoderType.GEMMA_2_2B
    if 'model.layers.0.self_attn.k_proj.bias' in sd:
        weight = sd['model.layers.0.self_attn.k_proj.bias']
        if weight.shape[0] == 256:
            return TextEncoderType.QWEN25_3B
        if weight.shape[0] == 512:
            return TextEncoderType.QWEN25_7B
    if "model.layers.0.post_attention_layernorm.weight" in sd:
        return TextEncoderType.LLAMA3_8
    return None

# auto selects the appropriate class for the given
# model weights
def create_text_encoder(model_path) -> TextEncoder:
    state_dict = ModelMixin.get_state_dict(model_path, VRAM_DEVICE)
    te_type: TextEncoderType = te_type(state_dict)
    if te_type is None or te_type not in TE_TYPE_CLS_MAP:
        raise Exception("Text encoder not supported")
    
    te_cls = TE_TYPE_CLS_MAP[te_type]
    del state_dict
    return te_cls(model_path)

class ModelEncoder:
    # model specific text encoder base
    # these include the respective tokenizer as well
    pass

class WanEncoder(ModelEncoder):
    def __init__(self, t5_xxl: TextEncoder):
        self.t5_xxl = t5_xxl
        self.tokenizer = UMTT5XXLTokenizer()
    
    def load_model(self, t5_xxl_download_url = None):
        self.t5_xxl.load_model(download_url=t5_xxl_download_url)
        assert self.t5_xxl.te_type == TextEncoderType.T5_XXL.value, f"expected T5_XXL but found {self.t5_xxl.te_type}"
        
    def encode(self, text):
        tokens, special_tokens = self.tokenize(text)
        return self.t5_xxl.encode_token_weights(tokens, special_tokens)
    
    def tokenize(self, text, return_word_ids=False) -> Tuple:
        # returns (tokens, special_tokens)
        return self.tokenizer.tokenize_with_weights(text, return_word_ids)