# Copyright 2026 The Qwen team, Alibaba Group. All rights reserved.
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
"""PyTorch Qwen3TTSTokenizer model."""

import math
from dataclasses import dataclass
from typing import Callable, Optional, Union, List, Any

import numpy as np
import torch
from torch import nn
from torch.nn import Parameter
from torch.nn import functional as F

from .tokenizer_config import (
    Qwen3TTSTokenizerConfig,
    Qwen3TTSTokenizerDecoderConfig,
)
from .utils import (
    ModelOutput,
    BaseModelOutputWithPast,
    repeat_kv,
    rotate_half,
    apply_rotary_pos_emb,
)
from .mimi_model import MimiModel
from ....utils.logging import app_logger
from ....model_utils.autoregressive_model_mixin import ARModelMixin
from ....model_utils.meta_utils import materialize_meta_buffers
from ...attention import eager_attention_forward

"""
   Qwen3TTSTokenizerModel  <-- The Wrapper
   │
   ├── encoder: Qwen3TTSTokenizerEncoder (MimiModel)  <-- Turns Audio to Codes
   │
   └── decoder: Qwen3TTSTokenizerDecoder              <-- Turns Codes to Audio
       │
       ├── quantizer: SplitResidualVectorQuantizer    <-- Turns Codes to Vectors
       │
       ├── pre_transformer: Qwen3TTSTokenizerDecoderTransformerModel <-- Smooths vectors using Attention
       │   ├── layer 1
       │   ├── layer 2...
       │
       └── upsample: ModuleList[CausalConvNet...]     <-- Stretches vectors into Audio Waveforms
"""



@dataclass
class Qwen3TTSTokenizerEncoderOutput(ModelOutput):
    r"""
    audio_codes (`List[torch.LongTensor]`):
        Discret code embeddings computed using `model.encode`, each tensor has shape (codes_length_i, num_quantizers).
    """

    audio_codes: List[torch.LongTensor] = None


@dataclass
class Qwen3TTSTokenizerDecoderOutput(ModelOutput):
    r"""
    audio_values (`List[torch.FloatTensor]`):
        Decoded audio values, obtained using the decoder part of Qwen3TTSTokenizerV1.
        Each tensor has shape (segment_length_i).
    """

    audio_values: List[torch.FloatTensor] = None


class Qwen3TTSTokenizerCausalConvNet(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dilation=1,
        stride=1,
        groups=1,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
        )
        self.stride = stride
        self.kernel_size = (kernel_size - 1) * dilation + 1
        self.dilation = dilation
        self.padding = self.kernel_size - self.stride

    def _get_extra_padding_for_conv1d(self, hidden_state: torch.Tensor) -> int:
        length = hidden_state.shape[-1]
        n_frames = (length - self.kernel_size + self.padding) / self.stride + 1
        ideal_length = (math.ceil(n_frames) - 1) * self.stride + (self.kernel_size - self.padding)
        return ideal_length - length

    def forward(self, hidden_state):
        extra_padding = self._get_extra_padding_for_conv1d(hidden_state)
        hidden_state = F.pad(hidden_state, (self.padding, extra_padding), mode="constant", value=0)
        return self.conv(hidden_state).contiguous()


class Qwen3TTSTokenizerCausalTransConvNet(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        self.conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=stride)

        pad = kernel_size - stride
        self.left_pad = 0
        self.right_pad = int(pad)

    def forward(self, hidden_state):
        hidden_state = self.conv(hidden_state)
        if self.right_pad > 0:
            hidden_state = hidden_state[..., : hidden_state.shape[-1] - self.right_pad]
        return hidden_state.contiguous()


class Qwen3TTSTokenizerConvNeXtBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dwconv = Qwen3TTSTokenizerCausalConvNet(
            dim,
            dim,
            kernel_size=7,
            groups=dim,
            dilation=1,
        )
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(1e-6 * torch.ones(dim))

    def forward(self, hidden_states):
        input = hidden_states

        hidden_states = self.dwconv(hidden_states)
        hidden_states = hidden_states.permute(0, 2, 1)
        hidden_states = self.norm(hidden_states)
        hidden_states = self.pwconv1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.pwconv2(hidden_states)

        hidden_states = self.gamma * hidden_states

        hidden_states = hidden_states.permute(0, 2, 1)

        hidden_states = input + hidden_states

        return hidden_states


def _compute_rope_inv_freq(config, device=None):
    base = getattr(config, "rope_theta", 10000.0)
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.int64, device=device).float() / head_dim))
    return inv_freq, 1.0

class Qwen3TTSTokenizerDecoderRotatoryEmbedding(nn.Module):
    def __init__(self, config: Qwen3TTSTokenizerDecoderConfig, device=None):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_attention_heads)
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.config = config
        
        # Simple ROPE implementation with scaling support
        self.attention_scaling = 1.0 # Default scaling
        inv_freq, _ = _compute_rope_inv_freq(config, device=device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x, position_ids):
        # position_ids: [batch, seq_len]
        # inv_freq: [head_dim // 2]
        
        # Lazy initialization: if inv_freq is zeros, recompute it
        if self.inv_freq.abs().max() == 0:
            device = self.inv_freq.device
            self.inv_freq = _compute_rope_inv_freq(self.config, device=device)[0].to(device)
        
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        # Force Float32 for RoPE calculations to prevent precision drift
        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Qwen3TTSTokenizerDecoderAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Qwen3TTSTokenizerDecoderConfig, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.q_norm = nn.Identity()
        self.k_norm = nn.Identity()
        self.sliding_window = config.sliding_window

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Any] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # TODO: complete the caching using a custom hook system (there should be a way to distinguish AR vs non-AR hooks)
        # Simplified caching: if past_key_values is present, it's expected to be managed externally or by a DynamicCache replacement
        if past_key_values is not None:
             # This part requires a local DynamicCache or just not using cache for now if not needed
             # For Qwen3-TTS Tokenizer decoder, it's usually used in chunked_decode which doesn't rely on HF Cache object
             pass

        attn_output = eager_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            self.config.num_attention_heads,
            attention_mask,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output


class Qwen3TTSTokenizerDecoderMlp(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        
        # Local SILU activation
        self.act_fn = F.silu if config.hidden_act == "silu" else F.gelu

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Qwen3TTSTokenizerDecoderRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6) -> None:
        """
        Qwen3TTSTokenizerDecoderRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class Qwen3TTSTokenizerDecoderLayerScale(nn.Module):
    def __init__(self, config):
        super().__init__()
        channels = config.hidden_size
        initial_scale = config.layer_scale_initial_scale
        self.scale = nn.Parameter(torch.full((channels,), initial_scale, requires_grad=True))

    def forward(self, x: torch.Tensor):
        return self.scale * x


class Qwen3TTSTokenizerDecoderTransformerLayer(nn.Module):
    def __init__(self, config: Qwen3TTSTokenizerDecoderConfig, layer_idx):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3TTSTokenizerDecoderAttention(config, layer_idx)
        self.mlp = Qwen3TTSTokenizerDecoderMlp(config)
        self.input_layernorm = Qwen3TTSTokenizerDecoderRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3TTSTokenizerDecoderRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn_layer_scale = Qwen3TTSTokenizerDecoderLayerScale(config)
        self.mlp_layer_scale = Qwen3TTSTokenizerDecoderLayerScale(config)
        self.attention_type = "sliding_attention"

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Any] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> tuple[torch.FloatTensor, Optional[tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = residual + self.self_attn_layer_scale(hidden_states)

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + self.mlp_layer_scale(hidden_states)

        return hidden_states


class Qwen3TTSTokenizerDecoderTransformerModel(nn.Module):
    def __init__(self, config: Qwen3TTSTokenizerDecoderConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [Qwen3TTSTokenizerDecoderTransformerLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3TTSTokenizerDecoderRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3TTSTokenizerDecoderRotatoryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.has_sliding_layers = "sliding_attention" in self.config.layer_types
        self.window_size = config.sliding_window

        self.input_proj = nn.Linear(config.latent_dim, config.hidden_size)
        self.output_proj = nn.Linear(config.hidden_size, config.latent_dim)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        cache_position=None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        if inputs_embeds is None:
            raise ValueError("inputs_embeds must be specified")
        
        hidden_states = self.input_proj(inputs_embeds)

        if cache_position is None:
            past_seen_tokens = 0 # No caching support for now
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + hidden_states.shape[1], device=hidden_states.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # Simple causal mask creation
        if attention_mask is None:
            seq_len = hidden_states.shape[1]
            attention_mask = torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=hidden_states.device, dtype=hidden_states.dtype), diagonal=1)
            attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        hidden_states = self.output_proj(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=None,
        )


class SnakeBeta(nn.Module):
    def __init__(self, in_features, alpha=1.0):
        super().__init__()
        self.in_features = in_features
        self.alpha = Parameter(torch.zeros(in_features) * alpha)
        self.beta = Parameter(torch.zeros(in_features) * alpha)
        self.no_div_by_zero = 0.000000001

    def forward(self, hidden_states):
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)
        beta = self.beta.unsqueeze(0).unsqueeze(-1)
        alpha = torch.exp(alpha)
        beta = torch.exp(beta)
        hidden_states = hidden_states + (1.0 / (beta + self.no_div_by_zero)) * torch.pow(
            torch.sin(hidden_states * alpha), 2
        )
        return hidden_states


class Qwen3TTSTokenizerDecoderDecoderResidualUnit(nn.Module):
    def __init__(self, dim: int = 16, dilation: int = 1):
        super().__init__()
        self.act1 = SnakeBeta(dim)
        self.conv1 = Qwen3TTSTokenizerCausalConvNet(dim, dim, kernel_size=7, dilation=dilation)
        self.act2 = SnakeBeta(dim)
        self.conv2 = Qwen3TTSTokenizerCausalConvNet(dim, dim, kernel_size=1)

    def forward(self, hidden_state):
        residual = hidden_state
        hidden_state = self.act1(hidden_state)
        hidden_state = self.conv1(hidden_state)
        hidden_state = self.act2(hidden_state)
        hidden_state = self.conv2(hidden_state)
        return hidden_state + residual


class Qwen3TTSTokenizerDecoderDecoderBlock(nn.Module):
    def __init__(self, config: Qwen3TTSTokenizerDecoderConfig, layer_idx):
        super().__init__()
        in_dim = config.decoder_dim // 2**layer_idx
        out_dim = config.decoder_dim // 2 ** (layer_idx + 1)
        upsample_rate = config.upsample_rates[layer_idx]

        block = [
            SnakeBeta(in_dim),
            Qwen3TTSTokenizerCausalTransConvNet(in_dim, out_dim, 2 * upsample_rate, upsample_rate),
        ]

        for dilation in (1, 3, 9):
            block.append(Qwen3TTSTokenizerDecoderDecoderResidualUnit(out_dim, dilation))

        self.block = nn.ModuleList(block)

    def forward(self, hidden):
        for block in self.block:
            hidden = block(hidden)
        return hidden


from .vq import SplitResidualVectorQuantizer

class Qwen3TTSTokenizerDecoder(nn.Module):
    def __init__(self, config: Qwen3TTSTokenizerDecoderConfig):
        super().__init__()
        self.config = config
        self.total_upsample = np.prod(config.upsample_rates + config.upsampling_ratios)
        self.pre_transformer = Qwen3TTSTokenizerDecoderTransformerModel(config)
        
        self.quantizer = SplitResidualVectorQuantizer(
            dimension=config.codebook_dim // 2,
            n_q=config.num_quantizers,
            n_q_semantic=1,
            bins=config.codebook_size,
            input_dimension=config.codebook_dim,
            output_dimension=config.codebook_dim,
        )

        self.pre_conv = Qwen3TTSTokenizerCausalConvNet(config.codebook_dim, config.latent_dim, kernel_size=3)

        upsample = []
        for factor in config.upsampling_ratios:
            upsample.append(
                nn.ModuleList(
                    [
                        Qwen3TTSTokenizerCausalTransConvNet(config.latent_dim, config.latent_dim, factor, factor),
                        Qwen3TTSTokenizerConvNeXtBlock(config.latent_dim),
                    ]
                )
            )
        self.upsample = nn.ModuleList(upsample)

        decoder = [Qwen3TTSTokenizerCausalConvNet(config.latent_dim, config.decoder_dim, 7)]
        for i in range(len(config.upsample_rates)):
            decoder.append(Qwen3TTSTokenizerDecoderDecoderBlock(config, i))
        output_dim = config.decoder_dim // 2 ** len(config.upsample_rates)
        decoder += [SnakeBeta(output_dim), Qwen3TTSTokenizerCausalConvNet(output_dim, 1, 7)]
        self.decoder = nn.ModuleList(decoder)

    def forward(self, codes):
        hidden = self.quantizer.decode(codes)
        hidden = self.pre_conv(hidden).transpose(1, 2)
        hidden = self.pre_transformer(inputs_embeds=hidden).last_hidden_state
        hidden = hidden.permute(0, 2, 1)
        for blocks in self.upsample:
            for block in blocks:
                hidden = block(hidden)
        wav = hidden
        for block in self.decoder:
            wav = block(wav)
        return wav.clamp(min=-1, max=1)

    def chunked_decode(self, codes, chunk_size=300, left_context_size=25):
        wavs = []
        start_index = 0
        while start_index < codes.shape[-1]:
            end_index = min(start_index + chunk_size, codes.shape[-1])
            context_size = left_context_size if start_index - left_context_size > 0 else start_index
            codes_chunk = codes[..., start_index - context_size : end_index]
            wav_chunk = self(codes_chunk)
            wavs.append(wav_chunk[..., context_size * self.total_upsample :])
            start_index = end_index
        return torch.cat(wavs, dim=-1)

# NOTE: qwen uses mimi for encoder but its own custom
# logic (esp snakebeta) for decoding
class Qwen3TTSTokenizerEncoder(MimiModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        
        # only use Mimi for encoding, drop the decoder parts
        self.upsample = None
        self.decoder_transformer = None
        self.decoder = None


class Qwen3TTSTokenizerModel(ARModelMixin):
    def __init__(self, config: Qwen3TTSTokenizerConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.dtype = kwargs.get("dtype", torch.float32)
        self.encoder_valid_num_quantizers = config.encoder_valid_num_quantizers
        self.input_sample_rate = config.input_sample_rate
        self.output_sample_rate = config.output_sample_rate
        self.decode_upsample_rate = config.decode_upsample_rate
        self.encode_downsample_rate = config.encode_downsample_rate
        
        self.encoder = Qwen3TTSTokenizerEncoder(config.encoder_config)
            
        self.decoder = Qwen3TTSTokenizerDecoder(config.decoder_config)
    
    def get_model_type(self):
        return self.config.model_type
    
    def get_input_sample_rate(self):
        return self.input_sample_rate
    
    def get_output_sample_rate(self):
        return self.output_sample_rate
    
    def get_encode_downsample_rate(self):
        return self.encode_downsample_rate
    
    def get_decode_upsample_rate(self):
        return self.decode_upsample_rate
    
    def encode(self, input_values: torch.Tensor, padding_mask: Optional[torch.Tensor] = None, return_dict: bool = True) -> Qwen3TTSTokenizerEncoderOutput:
        with torch.no_grad():
            encoded_frames = self.encoder.encode(input_values=input_values.unsqueeze(1))
        audio_codes = encoded_frames.audio_codes[:, :self.encoder_valid_num_quantizers]
        audio_codes = [code[..., :-(-mask.sum() // self.encode_downsample_rate)].transpose(0, 1) for code, mask in zip(audio_codes, padding_mask)]
        return Qwen3TTSTokenizerEncoderOutput(audio_codes)

    def decode(self, audio_codes: torch.Tensor, return_dict: bool = True) -> Qwen3TTSTokenizerDecoderOutput:
        audio_lengths = (audio_codes[..., 0] > -1).sum(1) * self.decode_upsample_rate
        audio_codes = torch.clamp(audio_codes, min=0)
        with torch.no_grad():
            audio_values = self.decoder.chunked_decode(audio_codes.transpose(1, 2)).squeeze(1)
        audio_values = [a[:l] for a, l in zip(audio_values, audio_lengths)]
        return Qwen3TTSTokenizerDecoderOutput(audio_values)

