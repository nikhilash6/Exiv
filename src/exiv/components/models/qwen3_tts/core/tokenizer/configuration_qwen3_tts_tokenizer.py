# coding=utf-8
# Copyright 2026 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Qwen3TTSTokenizer model configuration"""

from ......utils.logging import app_logger

class BaseConfig:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def to_dict(self):
        return self.__dict__.copy()


class MimiConfig(BaseConfig):
    def __init__(
        self,
        vocab_size=1024,
        hidden_size=512,
        num_hidden_layers=8,
        num_attention_heads=8,
        num_key_value_heads=8,
        intermediate_size=2048,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=None,
        eos_token_id=None,
        tie_word_embeddings=True,
        rope_theta=10000.0,
        rope_scaling=None,
        q_norm=True,
        k_norm=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.use_cache = use_cache
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.tie_word_embeddings = tie_word_embeddings
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.q_norm = q_norm
        self.k_norm = k_norm


class Qwen3TTSTokenizerDecoderConfig(BaseConfig):
    r"""
    This is the configuration class to store the configuration of a [`Qwen3TTSTokenizerDecoderConfig`].
    """

    def __init__(
        self,
        codebook_size=2048,
        hidden_size=1024,
        latent_dim=1024,
        max_position_embeddings=8000,
        rope_theta=10000,
        num_attention_heads=16,
        num_key_value_heads=16,
        attention_bias=False,
        sliding_window=72,
        intermediate_size=3072,
        hidden_act="silu",
        layer_scale_initial_scale=0.01,
        rms_norm_eps=1e-5,
        num_hidden_layers=8,
        num_quantizers=16,
        upsample_rates=(8, 5, 4, 3),
        upsampling_ratios=(2, 2),
        decoder_dim=1536,
        attention_dropout=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.codebook_size = codebook_size
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.attention_bias = attention_bias
        self.sliding_window = sliding_window
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.layer_scale_initial_scale = layer_scale_initial_scale
        self.rms_norm_eps = rms_norm_eps
        self.num_hidden_layers = num_hidden_layers
        self.num_quantizers = num_quantizers
        self.upsample_rates = upsample_rates
        self.upsampling_ratios = upsampling_ratios
        self.decoder_dim = decoder_dim
        self.attention_dropout = attention_dropout

    @property
    def layer_types(self):
        """
        All layer in code2wav should be sliding attention
        """
        return ["sliding_attention"] * self.num_hidden_layers


class Qwen3TTSTokenizerConfig(BaseConfig):
    """
    This is the configuration class to store the configuration of a [`Qwen3TTSTokenizerConfig`].
    """

    model_type = "qwen3_tts_tokenizer_12hz"

    def __init__(
        self,
        encoder_config=None,
        decoder_config=None,
        encoder_valid_num_quantizers=16,
        input_sample_rate=24000,
        output_sample_rate=24000,
        decode_upsample_rate=1920,
        encode_downsample_rate=1920,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if encoder_config is None:
            encoder_config = {}
            app_logger.info("encoder_config is None. Initializing encoder with default values")
        if decoder_config is None:
            decoder_config = {}
            app_logger.info("decoder_config is None. Initializing decoder with default values")

        if isinstance(encoder_config, MimiConfig):
            self.encoder_config = encoder_config
        else:
            self.encoder_config = MimiConfig(**encoder_config)
            
        if isinstance(decoder_config, Qwen3TTSTokenizerDecoderConfig):
            self.decoder_config = decoder_config
        else:
            self.decoder_config = Qwen3TTSTokenizerDecoderConfig(**decoder_config)

        self.encoder_valid_num_quantizers = encoder_valid_num_quantizers
        self.input_sample_rate = input_sample_rate
        self.output_sample_rate = output_sample_rate
        self.decode_upsample_rate = decode_upsample_rate
        self.encode_downsample_rate = encode_downsample_rate

