import functools
from typing import List, Optional

from .t5 import T5, T5XXL
from .text_tokenizer import UMTT5XXLTokenizer, WanT5Tokenizer
from ..enum import TextEncoderType
from ...utils.file import ensure_model_available
from ...model_utils.model_mixin import ModelMixin
from ...utils.device import DEFAULT_DEVICE
from ...utils.logging import app_logger

TE_TYPE_CLS_MAP = {
    TextEncoderType.T5_XXL: T5XXL
}

# NOTE: not inheriting from ModelMixin, model is loaded all at once (for now)
# primary reason rn is that we infer the TE type after loading it
class TextEncoder:
    def __init__(self, path: str):
        self.path = path
        self.te_model = None
        
    def load_model(self):
        path = ensure_model_available(self.path)
        if not path:
            app_logger.warning(f"text encoder {path} not found!")
            return None

        self.te_model = self.load_text_encoder(path)
    
    def load_text_encoder(self, model_path: str):
        state_dict = ModelMixin.get_state_dict(model_path, DEFAULT_DEVICE)
        if self.te_type is None or self.te_type not in TE_TYPE_CLS_MAP:
            raise Exception("Text encoder not supported")
        
        te_cls = TE_TYPE_CLS_MAP[self.te_type]
        return te_cls.load_state_dict(state_dict)
        
    @functools.lru_cache(max_size=2)
    @property
    def te_type(self, sd):
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

class ModelEncoder:
    # model specific text encoder base
    # these include the respective tokenizer as well
    pass

class WanEncoder(ModelEncoder):
    def __init__(self, t5_xxl: TextEncoder):
        self.t5_xxl = t5_xxl
        self.tokenizer = UMTT5XXLTokenizer()
    
    def load_model(self):
        self.t5_xxl.load_model()
        assert self.t5_xxl.te_type == TextEncoderType.T5_XXL, f"expected T5_XXL but found {self.t5_xxl.te_type}"
        
    def encode(self, text):
        tokens = self.tokenize(text)
        return None     # TODO: complete this 
    
    def tokenize(self, text, return_word_ids=False):
        return self.tokenizer.tokenize_with_weights(text, return_word_ids)