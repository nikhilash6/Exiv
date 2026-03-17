# Copyright 2024 Kyutai, and the HuggingFace Inc. team. All rights reserved.
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
"""
Mimi Model Architecture

Data Flow - Encode (Audio -> Codes):
   Raw Audio [B, 1, T]
        ↓
   encoder (MimiEncoder - SEANet CNN) [B, hidden_size, T//320]
        ↓
   encoder_transformer (Transformer) [B, T//320, hidden_size]
        ↓
   downsample (Conv1d) [B, hidden_size, T//640]  <-- Matches frame_rate (12.5Hz)
        ↓
   quantizer (RVQ) [B, num_quantizers, T//640]  <-- Discrete codes

Data Flow - Decode (Codes -> Audio):
   Codes [B, num_quantizers, T//640]
        ↓
   quantizer.decode (RVQ) [B, hidden_size, T//640]
        ↓
   upsample (ConvTranspose1d) [B, hidden_size, T//320]
        ↓
   decoder_transformer (Transformer) [B, T//320, hidden_size]
        ↓
   decoder (MimiDecoder - SEANet CNN) [B, 1, T]  <-- Reconstructed Audio

Note: T = audio samples at 24kHz. Frame rate is 12.5Hz, so downsample by 2x from encodec_frame_rate (25Hz).
      SEANet encoder does 320x downsampling (8*5*4*2), transformer works on latent, 
      then extra 2x downsample to reach target frame rate.
"""
import torch
import torch.nn as nn
from torch import Tensor

import math
from dataclasses import dataclass
from typing import Optional, Any, Union
from transformers.cache_utils import Cache, DynamicCache

from ....utils.logging import app_logger
from ...attention import create_attention_mask, repeat_kv
from ....model_utils.meta_utils import materialize_meta_buffers
from ...models.qwen3_tts.core.common_modules import _compute_rope_inv_freq
from .utils import rotate_half, apply_rotary_pos_emb


@dataclass
class MimiOutput:
    audio_codes: Optional[torch.LongTensor] = None
    audio_values: Optional[torch.FloatTensor] = None
    encoder_past_key_values: Optional[Union[Cache, list[torch.FloatTensor]]] = None
    decoder_past_key_values: Optional[Union[Cache, list[torch.FloatTensor]]] = None


@dataclass
class MimiEncoderOutput:
    audio_codes: Optional[torch.LongTensor] = None
    encoder_past_key_values: Optional[Union[Cache, list[torch.FloatTensor]]] = None
    padding_cache: Optional["MimiConv1dPaddingCache"] = None


@dataclass
class MimiDecoderOutput:
    audio_values: Optional[torch.FloatTensor] = None
    decoder_past_key_values: Optional[Union[Cache, list[torch.FloatTensor]]] = None


class MimiConv1dPaddingCache:
    """Padding cache for MimiConv1d causal convolutions to support streaming."""

    def __init__(
        self,
        num_layers: int,
        per_layer_padding: list[int],
        per_layer_padding_mode: list[str],
        per_layer_in_channels: list[int],
    ):
        from_args_num_layers = {len(per_layer_padding), len(per_layer_padding_mode), len(per_layer_in_channels)}

        if len(from_args_num_layers) != 1 or from_args_num_layers.pop() != num_layers:
            raise ValueError(
                f"Expected `num_layers` ({num_layers}) values in padding args"
            )
        
        self.per_layer_padding = per_layer_padding
        self.per_layer_padding_mode = per_layer_padding_mode
        self.per_layer_in_channels = per_layer_in_channels
        self.per_layer_is_init = [True] * num_layers
        self.padding_cache = [None] * num_layers

    def update(self, hidden_states: torch.Tensor, layer_idx: int):
        batch_size, dtype, device = hidden_states.shape[0], hidden_states.dtype, hidden_states.device
        padding = self.per_layer_padding[layer_idx]
        padding_mode = self.per_layer_padding_mode[layer_idx]
        in_channels = self.per_layer_in_channels[layer_idx]

        if self.padding_cache[layer_idx] is None:
            if padding_mode == "constant":
                current_cache = torch.zeros(batch_size, in_channels, padding, device=device, dtype=dtype)
            elif padding_mode == "replicate":
                current_cache = torch.ones(batch_size, in_channels, padding, device=device, dtype=dtype) * hidden_states[..., :1]
        else:
            current_cache = self.padding_cache[layer_idx]

        if padding > 0:
            padding_states = hidden_states[:, :, -padding:]
        else:
            padding_states = torch.empty(batch_size, in_channels, padding, dtype=dtype, device=device)
        
        self.padding_cache[layer_idx] = padding_states
        return current_cache


class MimiConv1d(nn.Module):
    """Conv1d with asymmetric or causal padding."""

    def __init__(
        self,
        config,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        pad_mode: Optional[str] = None,
        bias: bool = True,
        layer_idx: Optional[int] = None,
    ):
        super().__init__()
        self.causal = config.use_causal_conv
        self.pad_mode = config.pad_mode if pad_mode is None else pad_mode
        self.layer_idx = layer_idx
        self.in_channels = in_channels

        if stride > 1 and dilation > 1:
            app_logger.warning(
                f"MimiConv1d with stride > 1 and dilation > 1 (k={kernel_size}, s={stride}, d={dilation})"
            )

        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride, dilation=dilation, groups=groups, bias=bias
        )

        kernel_size = self.conv.kernel_size[0]
        stride = self.conv.stride[0]
        dilation = self.conv.dilation[0]

        kernel_size = (kernel_size - 1) * dilation + 1

        self.stride = stride
        self.kernel_size = kernel_size
        self.padding_total = kernel_size - stride
        self.padding_right = self.padding_total // 2
        self.padding_left = self.padding_total - self.padding_right

    def apply_weight_norm(self):
        weight_norm = nn.utils.weight_norm
        if hasattr(nn.utils.parametrizations, "weight_norm"):
            weight_norm = nn.utils.parametrizations.weight_norm
        weight_norm(self.conv)

    def remove_weight_norm(self):
        nn.utils.remove_weight_norm(self.conv)

    def _get_extra_padding_for_conv1d(self, hidden_states: torch.Tensor) -> torch.Tensor:
        length = hidden_states.shape[-1]
        n_frames = (length - self.kernel_size + self.padding_total) / self.stride + 1
        n_frames = math.ceil(n_frames) - 1
        ideal_length = n_frames * self.stride + self.kernel_size - self.padding_total
        return ideal_length - length

    @staticmethod
    def _pad1d(hidden_states: torch.Tensor, paddings: tuple[int, int], mode: str = "zero", value: float = 0.0):
        length = hidden_states.shape[-1]
        padding_left, padding_right = paddings
        if mode != "reflect":
            return nn.functional.pad(hidden_states, paddings, mode, value)

        max_pad = max(padding_left, padding_right)
        extra_pad = 0
        if length <= max_pad:
            extra_pad = max_pad - length + 1
            hidden_states = nn.functional.pad(hidden_states, (0, extra_pad))
        padded = nn.functional.pad(hidden_states, paddings, mode, value)
        end = padded.shape[-1] - extra_pad
        return padded[..., :end]

    def _get_output_length(self, input_length: torch.LongTensor) -> torch.LongTensor:
        n_frames = (input_length - self.kernel_size + self.padding_total) / self.stride + 1
        n_frames = torch.ceil(n_frames).to(torch.int64) - 1
        ideal_length = n_frames * self.stride + self.kernel_size - self.padding_total
        extra_padding = ideal_length - input_length

        if self.causal:
            padding_left = self.padding_total
            padding_right = extra_padding
        else:
            padding_left = self.padding_left
            padding_right = self.padding_right + extra_padding

        input_length = input_length + padding_left + padding_right
        output_length = (input_length + 2 * self.conv.padding[0] - self.conv.dilation[0] * (self.conv.kernel_size[0] - 1) - 1) // self.conv.stride[0] + 1
        return output_length

    def forward(self, hidden_states, padding_cache=None):
        extra_padding = self._get_extra_padding_for_conv1d(hidden_states)

        if not self.causal and padding_cache is not None:
            raise ValueError("`padding_cache` is not supported for non-causal convolutions.")

        if self.causal and padding_cache is not None:
            layer_padding_cache = padding_cache.update(hidden_states, self.layer_idx)
            hidden_states = torch.cat([layer_padding_cache, hidden_states], dim=2)
        elif self.causal:
            hidden_states = self._pad1d(hidden_states, (self.padding_total, extra_padding), mode=self.pad_mode)
        else:
            hidden_states = self._pad1d(hidden_states, (self.padding_left, self.padding_right + extra_padding), mode=self.pad_mode)

        return self.conv(hidden_states)


class MimiConvTranspose1d(nn.Module):
    """ConvTranspose1d with asymmetric or causal padding."""

    def __init__(
        self,
        config,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
        bias=True,
    ):
        super().__init__()
        self.causal = config.use_causal_conv
        self.trim_right_ratio = config.trim_right_ratio
        self.conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, groups=groups, bias=bias)

        if not (self.causal or self.trim_right_ratio == 1.0):
            raise ValueError("`trim_right_ratio` != 1.0 only makes sense for causal convolutions")

        kernel_size = self.conv.kernel_size[0]
        stride = self.conv.stride[0]
        padding_total = kernel_size - stride

        if self.causal:
            self.padding_right = math.ceil(padding_total * self.trim_right_ratio)
        else:
            self.padding_right = padding_total // 2

        self.padding_left = padding_total - self.padding_right

    def apply_weight_norm(self):
        weight_norm = nn.utils.weight_norm
        if hasattr(nn.utils.parametrizations, "weight_norm"):
            weight_norm = nn.utils.parametrizations.weight_norm
        weight_norm(self.conv)

    def remove_weight_norm(self):
        nn.utils.remove_weight_norm(self.conv)

    def forward(self, hidden_states):
        hidden_states = self.conv(hidden_states)
        end = hidden_states.shape[-1] - self.padding_right
        return hidden_states[..., self.padding_left : end]


class MimiResnetBlock(nn.Module):
    """Residual block from SEANet model as used by Mimi."""

    def __init__(self, config: object, dim: int, dilations: list[int]):
        super().__init__()
        kernel_sizes = (config.residual_kernel_size, 1)
        if len(kernel_sizes) != len(dilations):
            raise ValueError("Number of kernel sizes should match number of dilations")

        hidden = dim // config.compress
        block = []
        for i, (kernel_size, dilation) in enumerate(zip(kernel_sizes, dilations)):
            in_chs = dim if i == 0 else hidden
            out_chs = dim if i == len(kernel_sizes) - 1 else hidden
            block += [nn.ELU()]
            block += [MimiConv1d(config, in_chs, out_chs, kernel_size, dilation=dilation)]
        self.block = nn.ModuleList(block)

        if config.use_conv_shortcut:
            self.shortcut = MimiConv1d(config, dim, dim, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, hidden_states, padding_cache=None):
        residual = hidden_states

        for layer in self.block:
            if isinstance(layer, MimiConv1d):
                hidden_states = layer(hidden_states, padding_cache=padding_cache)
            else:
                hidden_states = layer(hidden_states)

        if isinstance(self.shortcut, MimiConv1d):
            residual = self.shortcut(residual, padding_cache=padding_cache)
        else:
            residual = self.shortcut(residual)

        return residual + hidden_states


class MimiEncoder(nn.Module):
    """SEANet encoder as used by Mimi."""

    def __init__(self, config: object):
        super().__init__()
        model = [MimiConv1d(config, config.audio_channels, config.num_filters, config.kernel_size)]
        scaling = 1
        mimiconv1d_layer_names = ["layers.0"]

        for ratio in reversed(config.upsampling_ratios):
            current_scale = scaling * config.num_filters
            for j in range(config.num_residual_layers):
                mimiconv1d_layer_names.extend([f"layers.{len(model)}.block.1", f"layers.{len(model)}.block.3"])
                model += [MimiResnetBlock(config, current_scale, [config.dilation_growth_rate**j, 1])]
            model += [nn.ELU()]
            mimiconv1d_layer_names.append(f"layers.{len(model)}")
            model += [MimiConv1d(config, current_scale, current_scale * 2, kernel_size=ratio * 2, stride=ratio)]
            scaling *= 2

        model += [nn.ELU()]
        mimiconv1d_layer_names.append(f"layers.{len(model)}")
        model += [MimiConv1d(config, scaling * config.num_filters, config.hidden_size, config.last_kernel_size)]

        self.layers = nn.ModuleList(model)
        self._mimiconv1d_layer_names = mimiconv1d_layer_names

        for layer_idx, layername in enumerate(self._mimiconv1d_layer_names):
            conv_layer = self.get_submodule(layername)
            setattr(conv_layer, "layer_idx", layer_idx)

    def forward(self, hidden_states, padding_cache=None):
        for layer in self.layers:
            if isinstance(layer, (MimiConv1d, MimiResnetBlock)):
                hidden_states = layer(hidden_states, padding_cache=padding_cache)
            else:
                hidden_states = layer(hidden_states)
        return hidden_states


class MimiLayerScale(nn.Module):
    """Layer scale from Touvron et al 2021."""

    def __init__(self, config):
        super().__init__()
        channels = config.hidden_size
        initial_scale = config.layer_scale_initial_scale
        self.scale = nn.Parameter(torch.full((channels,), initial_scale, requires_grad=True))

    def forward(self, x: torch.Tensor):
        return self.scale * x


class MimiRotaryEmbedding(nn.Module):
    def __init__(self, config: object, device=None):
        super().__init__()
        if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.config = config

        inv_freq, self.attention_scaling = _compute_rope_inv_freq(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @materialize_meta_buffers(inv_freq=lambda self, device: _compute_rope_inv_freq(self.config, device)[0])
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class MimiMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = nn.GELU() if config.hidden_act == "gelu" else nn.SiLU()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class MimiAttention(nn.Module):
    """Multi-headed attention."""

    def __init__(self, config: object, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.scaling = 1 / math.sqrt(config.head_dim)
        self.sliding_window = getattr(config, "sliding_window", None)

        if self.hidden_size % self.num_heads != 0:
            raise ValueError(f"hidden_size must be divisible by num_heads")

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        self.rotary_emb = MimiRotaryEmbedding(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Simplified cache handling - just pass through for now
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            # Note: actual cache update would go here if implementing full cache support

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights if output_attentions else None


class MimiTransformerLayer(nn.Module):
    def __init__(self, config: object, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = MimiAttention(config=config, layer_idx=layer_idx)
        self.mlp = MimiMLP(config)
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps)
        self.self_attn_layer_scale = MimiLayerScale(config)
        self.mlp_layer_scale = MimiLayerScale(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
        )
        hidden_states = residual + self.self_attn_layer_scale(hidden_states)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + self.mlp_layer_scale(hidden_states)

        return (hidden_states,) if not output_attentions else (hidden_states, self_attn_weights)


class MimiTransformerModel(nn.Module):
    """Transformer model for Mimi."""

    def __init__(self, config: object):
        super().__init__()
        self.layers = nn.ModuleList(
            [MimiTransformerLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.gradient_checkpointing = False
        self.config = config

    def forward(
        self,
        hidden_states: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else False
        output_hidden_states = output_hidden_states if output_hidden_states is not None else False
        use_cache = use_cache if use_cache is not None else False
        return_dict = return_dict if return_dict is not None else True

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        # Create causal mask using our own implementation
        if cache_position is None:
            past_seen_tokens = 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + hidden_states.shape[1], device=hidden_states.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        seq_len = hidden_states.shape[1]
        device = hidden_states.device
        causal_mask = create_attention_mask(seq_len, seq_len, device, sliding_window=getattr(self.config, 'sliding_window', None))

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if return_dict:
            return {"last_hidden_state": hidden_states, "past_key_values": past_key_values, 
                    "hidden_states": all_hidden_states, "attentions": all_self_attns}
        return (hidden_states, past_key_values, all_hidden_states, all_self_attns)


class MimiDecoder(nn.Module):
    """SEANet decoder as used by Mimi."""

    def __init__(self, config: object):
        super().__init__()
        scaling = int(2 ** len(config.upsampling_ratios))
        model = [MimiConv1d(config, config.hidden_size, scaling * config.num_filters, config.kernel_size)]

        for ratio in config.upsampling_ratios:
            current_scale = scaling * config.num_filters
            model += [nn.ELU()]
            model += [MimiConvTranspose1d(config, current_scale, current_scale // 2, kernel_size=ratio * 2, stride=ratio)]
            for j in range(config.num_residual_layers):
                model += [MimiResnetBlock(config, current_scale // 2, (config.dilation_growth_rate**j, 1))]
            scaling //= 2

        model += [nn.ELU()]
        model += [MimiConv1d(config, config.num_filters, config.audio_channels, config.last_kernel_size)]
        self.layers = nn.ModuleList(model)

    def forward(self, hidden_states):
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


class MimiEuclideanCodebook(nn.Module):
    """Codebook with Euclidean distance."""

    def __init__(self, config: object, epsilon: float = 1e-5):
        super().__init__()
        embed = torch.zeros(config.codebook_size, config.codebook_dim)
        self.codebook_size = config.codebook_size

        self.register_buffer("initialized", torch.tensor([True], dtype=torch.float32), persistent=True)
        self.register_buffer("cluster_usage", torch.ones(config.codebook_size), persistent=True)
        self.register_buffer("embed_sum", embed, persistent=True)
        self._embed = None
        self.epsilon = epsilon

    @property
    def embed(self) -> torch.Tensor:
        if self._embed is None:
            self._embed = self.embed_sum / self.cluster_usage.clamp(min=self.epsilon)[:, None]
        return self._embed

    @materialize_meta_buffers(
        initialized=lambda self, device: torch.tensor([True], dtype=torch.float32, device=device),
        cluster_usage=lambda self, device: torch.ones(self.codebook_size, device=device),
        embed_sum=lambda self, device: torch.zeros(self.codebook_size, getattr(self, "codebook_dim", 256), device=device)
    )
    def quantize(self, hidden_states):
        dists = torch.cdist(hidden_states[None].float(), self.embed[None].float(), p=2)[0]
        embed_ind = dists.argmin(dim=-1)
        return embed_ind

    def encode(self, hidden_states):
        shape = hidden_states.shape
        hidden_states = hidden_states.reshape((-1, shape[-1]))
        embed_ind = self.quantize(hidden_states)
        embed_ind = embed_ind.view(*shape[:-1])
        return embed_ind

    def decode(self, embed_ind):
        quantize = nn.functional.embedding(embed_ind, self.embed)
        return quantize


class MimiVectorQuantization(nn.Module):
    """Vector quantization implementation."""

    def __init__(self, config: object):
        super().__init__()
        self.codebook = MimiEuclideanCodebook(config)

    def encode(self, hidden_states):
        hidden_states = hidden_states.permute(0, 2, 1)
        embed_in = self.codebook.encode(hidden_states)
        return embed_in

    def decode(self, embed_ind):
        quantize = self.codebook.decode(embed_ind)
        quantize = quantize.permute(0, 2, 1)
        return quantize


class MimiResidualVectorQuantizer(nn.Module):
    """Residual Vector Quantizer."""

    def __init__(self, config: object, num_quantizers: Optional[int] = None):
        super().__init__()
        self.codebook_size = config.codebook_size
        self.frame_rate = config.frame_rate
        self.num_quantizers = num_quantizers if num_quantizers is not None else config.num_quantizers
        self.layers = nn.ModuleList([MimiVectorQuantization(config) for _ in range(self.num_quantizers)])

        self.input_proj = None
        self.output_proj = None
        if config.vector_quantization_hidden_dimension != config.hidden_size:
            self.input_proj = torch.nn.Conv1d(
                config.hidden_size, config.vector_quantization_hidden_dimension, 1, bias=False
            )
            self.output_proj = torch.nn.Conv1d(
                config.vector_quantization_hidden_dimension, config.hidden_size, 1, bias=False
            )

    def encode(self, embeddings: torch.Tensor, num_quantizers: Optional[int] = None) -> torch.Tensor:
        if self.input_proj is not None:
            embeddings = self.input_proj(embeddings)

        num_quantizers = num_quantizers if num_quantizers is not None else self.num_quantizers

        residual = embeddings
        all_indices = []
        for layer in self.layers[:num_quantizers]:
            indices = layer.encode(residual)
            quantized = layer.decode(indices)
            residual = residual - quantized
            all_indices.append(indices)
        out_indices = torch.stack(all_indices)
        return out_indices

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        quantized_out = torch.tensor(0.0, device=codes.device)
        codes = codes.transpose(0, 1)
        for i, indices in enumerate(codes):
            layer = self.layers[i]
            quantized = layer.decode(indices)
            quantized_out = quantized_out + quantized

        if self.output_proj is not None:
            quantized_out = self.output_proj(quantized_out)
        return quantized_out


class MimiSplitResidualVectorQuantizer(nn.Module):
    """Split Residual Vector Quantizer."""

    def __init__(self, config: object):
        super().__init__()
        self.codebook_size = config.codebook_size
        self.frame_rate = config.frame_rate
        self.max_num_quantizers = config.num_quantizers

        self.num_semantic_quantizers = config.num_semantic_quantizers
        self.num_acoustic_quantizers = config.num_quantizers - config.num_semantic_quantizers

        self.semantic_residual_vector_quantizer = MimiResidualVectorQuantizer(config, self.num_semantic_quantizers)
        self.acoustic_residual_vector_quantizer = MimiResidualVectorQuantizer(config, self.num_acoustic_quantizers)

    def encode(self, embeddings: torch.Tensor, num_quantizers: Optional[float] = None) -> torch.Tensor:
        num_quantizers = self.max_num_quantizers if num_quantizers is None else num_quantizers

        if num_quantizers > self.max_num_quantizers:
            raise ValueError(f"num_quantizers {num_quantizers} > max {self.max_num_quantizers}")

        if num_quantizers < self.num_semantic_quantizers:
            raise ValueError(f"num_quantizers {num_quantizers} < semantic quantizers {self.num_semantic_quantizers}")

        codes = self.semantic_residual_vector_quantizer.encode(embeddings)

        if num_quantizers > self.num_semantic_quantizers:
            acoustic_codes = self.acoustic_residual_vector_quantizer.encode(
                embeddings, num_quantizers=num_quantizers - self.num_semantic_quantizers
            )
            codes = torch.cat([codes, acoustic_codes], dim=0)

        return codes

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        quantized_out = self.semantic_residual_vector_quantizer.decode(codes[:, : self.num_semantic_quantizers])

        if codes.shape[1] > self.num_semantic_quantizers:
            quantized_out += self.acoustic_residual_vector_quantizer.decode(codes[:, self.num_semantic_quantizers :])
        return quantized_out


class MimiModel(nn.Module):
    """Mimi model for audio codec."""

    def __init__(self, config: object):
        super().__init__()
        self.config = config

        self.encoder = MimiEncoder(config)
        self.encoder_transformer = MimiTransformerModel(config)

        self.downsample = None
        self.upsample = None
        if config.frame_rate != config.encodec_frame_rate:
            self.downsample = MimiConv1d(
                config,
                config.hidden_size,
                config.hidden_size,
                kernel_size=2 * int(config.encodec_frame_rate / config.frame_rate),
                stride=2,
                bias=False,
                pad_mode="replicate",
                layer_idx=len(self.encoder._mimiconv1d_layer_names),
            )

            self.upsample = MimiConvTranspose1d(
                config,
                config.hidden_size,
                config.hidden_size,
                kernel_size=2 * int(config.encodec_frame_rate / config.frame_rate),
                stride=2,
                bias=False,
                groups=config.upsample_groups,
            )

        self.decoder_transformer = MimiTransformerModel(config)
        self.decoder = MimiDecoder(config)

        self.quantizer = MimiSplitResidualVectorQuantizer(config)

        self.bits_per_codebook = int(math.log2(self.config.codebook_size))
        if 2**self.bits_per_codebook != self.config.codebook_size:
            raise ValueError("codebook_size must be a power of 2")

    def get_encoder(self):
        return self.encoder

    def _encode_frame(
        self,
        input_values: torch.Tensor,
        num_quantizers: int,
        padding_mask: int,
        past_key_values: Optional = None,
        padding_cache: Optional[MimiConv1dPaddingCache] = None,
        return_dict: Optional[bool] = None,
    ):
        embeddings = self.encoder(input_values, padding_cache=padding_cache)

        encoder_outputs = self.encoder_transformer(
            embeddings.transpose(1, 2), past_key_values=past_key_values, return_dict=return_dict
        )
        if return_dict:
            past_key_values = encoder_outputs.get("past_key_values")
            hidden_states = encoder_outputs["last_hidden_state"]
        else:
            past_key_values = encoder_outputs[1] if len(encoder_outputs) > 1 else None
            hidden_states = encoder_outputs[0]
        embeddings = hidden_states.transpose(1, 2)
        embeddings = self.downsample(embeddings, padding_cache=padding_cache)

        codes = self.quantizer.encode(embeddings, num_quantizers)
        codes = codes.transpose(0, 1)
        return codes, past_key_values, padding_cache

    def get_encoded_length(self, input_length: torch.LongTensor) -> torch.LongTensor:
        output_length = input_length

        for layer_name in self.encoder._mimiconv1d_layer_names:
            output_length = self.encoder.get_submodule(layer_name)._get_output_length(output_length)

        output_length = self.downsample._get_output_length(output_length)
        return output_length

    def get_audio_codes_mask(self, padding_mask: torch.Tensor, padding_side: str = "right"):
        encoded_lengths = self.get_encoded_length(padding_mask.sum(dim=-1))

        audio_codes_mask = torch.arange(encoded_lengths.max(), device=encoded_lengths.device).expand(
            len(encoded_lengths), -1
        )
        audio_codes_mask = audio_codes_mask < encoded_lengths.unsqueeze(1)
        audio_codes_mask = audio_codes_mask.to(padding_mask.device)

        if padding_side == "right":
            return audio_codes_mask
        return audio_codes_mask.flip(dims=[-1])

    def encode(
        self,
        input_values: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        num_quantizers: Optional[float] = None,
        encoder_past_key_values: Optional = None,
        padding_cache: Optional[MimiConv1dPaddingCache] = None,
        use_streaming: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        use_streaming = use_streaming if use_streaming is not None else getattr(self.config, 'use_streaming', False)

        num_quantizers = self.config.num_quantizers if num_quantizers is None else num_quantizers

        if num_quantizers > self.config.num_quantizers:
            raise ValueError(f"num_quantizers {num_quantizers} > max {self.config.num_quantizers}")

        _, channels, input_length = input_values.shape

        if channels < 1 or channels > 2:
            raise ValueError(f"Number of audio channels must be 1 or 2, got {channels}")

        if padding_mask is None:
            padding_mask = torch.ones_like(input_values).bool()

        if use_streaming and padding_cache is None:
            per_layer_padding, per_layer_padding_mode, per_layer_in_channels = [], [], []
            for layer_name in self.encoder._mimiconv1d_layer_names:
                per_layer_padding.append(self.encoder.get_submodule(layer_name).padding_total)
                per_layer_padding_mode.append(self.encoder.get_submodule(layer_name).pad_mode)
                per_layer_in_channels.append(self.encoder.get_submodule(layer_name).in_channels)

            per_layer_padding.append(self.downsample.padding_total)
            per_layer_padding_mode.append(self.downsample.pad_mode)
            per_layer_in_channels.append(self.downsample.in_channels)

            padding_cache = MimiConv1dPaddingCache(
                num_layers=len(self.encoder._mimiconv1d_layer_names) + 1,
                per_layer_padding=per_layer_padding,
                per_layer_padding_mode=per_layer_padding_mode,
                per_layer_in_channels=per_layer_in_channels,
            )

        encoded_frames, encoder_past_key_values, padding_cache = self._encode_frame(
            input_values,
            num_quantizers,
            padding_mask.bool(),
            past_key_values=encoder_past_key_values,
            padding_cache=padding_cache,
            return_dict=return_dict,
        )

        if not return_dict:
            return (encoded_frames, encoder_past_key_values, padding_cache)

        return MimiEncoderOutput(encoded_frames, encoder_past_key_values, padding_cache)

    def _decode_frame(
        self,
        codes: torch.Tensor,
        past_key_values: Optional = None,
        return_dict: Optional[bool] = None,
    ):
        embeddings = self.quantizer.decode(codes)

        embeddings = self.upsample(embeddings)
        decoder_outputs = self.decoder_transformer(
            embeddings.transpose(1, 2), past_key_values=past_key_values, return_dict=return_dict
        )
        if return_dict:
            past_key_values = decoder_outputs.get("past_key_values")
            hidden_states = decoder_outputs["last_hidden_state"]
        else:
            past_key_values = decoder_outputs[1] if len(decoder_outputs) > 1 else None
            hidden_states = decoder_outputs[0]
        embeddings = hidden_states.transpose(1, 2)
        outputs = self.decoder(embeddings)
        return outputs, past_key_values

    def decode(
        self,
        audio_codes: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        decoder_past_key_values: Optional = None,
        return_dict: Optional[bool] = None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        audio_values, decoder_past_key_values = self._decode_frame(
            audio_codes, past_key_values=decoder_past_key_values, return_dict=return_dict
        )

        if padding_mask is not None and padding_mask.shape[-1] < audio_values.shape[-1]:
            audio_values = audio_values[..., : padding_mask.shape[-1]]

        if not return_dict:
            return (audio_values, decoder_past_key_values)
        return MimiDecoderOutput(audio_values, decoder_past_key_values)

    def forward(
        self,
        input_values: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        num_quantizers: Optional[int] = None,
        audio_codes: Optional[torch.Tensor] = None,
        encoder_past_key_values: Optional = None,
        decoder_past_key_values: Optional = None,
        return_dict: Optional[bool] = None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        if padding_mask is None:
            padding_mask = torch.ones_like(input_values).bool()

        if audio_codes is None:
            encoder_outputs = self.encode(
                input_values, padding_mask, num_quantizers, encoder_past_key_values, return_dict=return_dict
            )
            if return_dict:
                audio_codes = encoder_outputs.audio_codes
                encoder_past_key_values = encoder_outputs.encoder_past_key_values
            else:
                audio_codes = encoder_outputs[0]
                encoder_past_key_values = encoder_outputs[1] if len(encoder_outputs) > 1 else None

        decoder_outputs = self.decode(audio_codes, padding_mask, decoder_past_key_values, return_dict=return_dict)
        if return_dict:
            audio_values = decoder_outputs.audio_values
            decoder_past_key_values = decoder_outputs.decoder_past_key_values
        else:
            audio_values = decoder_outputs[0]
            decoder_past_key_values = decoder_outputs[1] if len(decoder_outputs) > 1 else None

        if not return_dict:
            return (audio_codes, audio_values, encoder_past_key_values, decoder_past_key_values)

        return MimiOutput(
            audio_codes=audio_codes,
            audio_values=audio_values,
            encoder_past_key_values=encoder_past_key_values,
            decoder_past_key_values=decoder_past_key_values,
        )


__all__ = ["MimiModel"]
