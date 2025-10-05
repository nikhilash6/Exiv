import warnings
from typing import List, Optional

from exiv.components.enum import TextEncoderType

from .text_tokenizer import WanT5Tokenizer
from ...utils.file import ensure_model_available

# NOTE: not inheriting from ModelMixin, model is loaded all at once
class TextEncoder:
    def __init__(self, path: str):
        self.path = path
        self.te_model = None
        
    def load_model(self):
        path = ensure_model_available(self.path)
        if not path:
            warnings.warn(f"text encoder {path} not found!")
            return None

        self.te_model = load_text_encoder_sd(path)
    
    @property
    def te_type(self):
        if self.te_model is None:
            raise Exception("can't determine TE type without loading the model")
        
        sd = self.te_model
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

# these include the respective tokenizer as well
class ModelTextEncoder:
    pass

class WanTextEncoder(ModelTextEncoder):
    def __init__(self, t5_xxl: TextEncoder):
        self.t5_xxl = t5_xxl
        self.tokenizer = WanT5Tokenizer()
    
    def load_encoder_dict(self):
        self.t5_xxl.load_model()
        assert self.t5_xxl.te_type == TextEncoderType.T5_XXL, f"expected T5_XXL but found {self.t5_xxl.te_type}"
        