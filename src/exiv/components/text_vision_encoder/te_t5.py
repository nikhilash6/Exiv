import torch
from torch import nn

import math
import copy
from typing import Optional

from ..activations import ACT2FN
from .encoder_base import TextEncoder, T5Config, T5XXLConfig, UMT5XXLConfig
from ..enum import TextEncoderType
from ..attention import optimized_attention
from ...utils.logging import app_logger

# code adapted from Huggingface Transformers

# NOTE: we are passing the position bias around in the layers so we 
# don't have recompute it everytime in individual layers
# NOTE: deleting all the KV cache stuff for now, since we are only using the
# the encoder part of the model
class T5LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Construct a layernorm module in the T5 style. No bias and no subtraction of mean.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        # T5 uses a layer_norm which only scales and doesn't shift, which is also known as Root Mean
        # Square Layer Normalization https://huggingface.co/papers/1910.07467 thus variance is calculated
        # w/o mean and there is no bias. Additionally we want to make sure that the accumulation for
        # half-precision inputs is done in fp32

        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states


class T5DenseActDense(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        self.wi = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.act = ACT2FN[config.dense_act_fn]

    def forward(self, hidden_states):
        hidden_states = self.wi(hidden_states)
        hidden_states = self.act(hidden_states)
        if (
            isinstance(self.wo.weight, torch.Tensor)
            and hidden_states.dtype != self.wo.weight.dtype
            and self.wo.weight.dtype != torch.int8
        ):
            hidden_states = hidden_states.to(self.wo.weight.dtype)
        return self.wo(hidden_states)


class T5DenseGatedActDense(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        self.wi_0 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wi_1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.act = ACT2FN[config.dense_act_fn]

    def forward(self, hidden_states):
        hidden_gelu = self.act(self.wi_0(hidden_states))
        hidden_linear = self.wi_1(hidden_states)
        hidden_states = hidden_gelu * hidden_linear

        # To make 8bit quantization work for google/flan-t5-xxl, self.wo is kept in float32.
        # See https://github.com/huggingface/transformers/issues/20287
        # we also make sure the weights are not in `int8` in case users will force `_keep_in_fp32_modules` to be `None``
        if (
            isinstance(self.wo.weight, torch.Tensor)
            and hidden_states.dtype != self.wo.weight.dtype
            and self.wo.weight.dtype != torch.int8
        ):
            hidden_states = hidden_states.to(self.wo.weight.dtype)

        return self.wo(hidden_states)


class T5LayerFF(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        if config.is_gated_act:
            self.DenseReluDense = T5DenseGatedActDense(config)
        else:
            self.DenseReluDense = T5DenseActDense(config)

        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)

    def forward(self, hidden_states):
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.DenseReluDense(forwarded_states)
        hidden_states += forwarded_states
        return hidden_states


class T5Attention(nn.Module):
    def __init__(
        self,
        config: T5Config,
        has_relative_attention_bias=False,
    ):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.has_relative_attention_bias = has_relative_attention_bias
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.relative_attention_max_distance = config.relative_attention_max_distance
        
        # TODO: some stuff seems redundant like here d_kv = d_model / num_heads
        self.d_model = config.d_model
        self.key_value_proj_dim = config.d_kv
        self.n_heads = config.num_heads
        self.dropout = config.dropout_rate      # dropput not in use
        self.inner_dim = self.n_heads * self.key_value_proj_dim

        # Mesh TensorFlow initialization to avoid scaling before softmax
        self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)

        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets, self.n_heads)
        self.gradient_checkpointing = False


    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on

        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer

        Returns:
            a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
        """
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        # now relative_position is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_position_if_large = torch.min(
            relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
        )

        relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets

    def compute_bias(self, query_length, key_length, device=None, cache_position=None):
        """Compute binned relative position bias"""
        if device is None:
            device = self.relative_attention_bias.weight.device
        if cache_position is None:
            context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
        else:
            context_position = cache_position[:, None].to(device)
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        relative_position = memory_position - context_position  # shape (query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        return values

    def forward(
        self,
        hidden_states,
        mask=None,
        key_value_states=None,
        position_bias=None,
    ):
        """
        Self-attention (if key_value_states is None) or cross-attention (if key_value_states is provided).
        """
        # hidden_states: (batch_size, seq_length, dim)
        batch_size, seq_length = hidden_states.shape[:2]

        query_states = self.q(hidden_states)
        # query_states = query_states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

        is_cross_attention = key_value_states is not None
        source_states = key_value_states if is_cross_attention else hidden_states   # i know this looks llm generated but its not

        key_states = self.k(source_states)
        value_states = self.v(source_states)
        # self.key_value_proj_dim is basically dim_head that we normally use
        # (batch_size, seq_length, hidden_dim) → (batch_size, seq_length, n_heads, key_value_proj_dim)
        # key_states = key_states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
        # value_states = value_states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

        # compute or use position bias
        if position_bias is None:
            key_length = key_states.shape[-2]
            if not self.has_relative_attention_bias:
                position_bias = torch.zeros(
                    (1, self.n_heads, seq_length, key_length),
                    device=query_states.device,
                    dtype=query_states.dtype,
                )
            else:
                position_bias = self.compute_bias(seq_length, key_length, device=query_states.device)

            # add mask to the position bias
            if mask is not None:
                mask = position_bias + mask
            else:
                mask = position_bias

        # ultimately, scores = scores + mask + bias
        # in this code we are using the mask AS the bias
        attn_output = optimized_attention(
            query_states,
            key_states,
            value_states,
            self.n_heads,
            mask
        )

        attn_output = self.o(attn_output)
        return attn_output, position_bias

class T5LayerSelfAttention(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        self.SelfAttention = T5Attention(config, has_relative_attention_bias=has_relative_attention_bias)
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attn_output, position_bias = self.SelfAttention(
            normed_hidden_states,
            mask=attention_mask,
            position_bias=position_bias,
        )
        hidden_states += attn_output
        return hidden_states, position_bias

# we don't need cross attn, but keeping the code just in case
class T5LayerCrossAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # relative distance has no meaning since we are calculating attn b/w separate sequences
        self.EncDecAttention = T5Attention(config, has_relative_attention_bias=False)
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)

    def forward(
        self,
        hidden_states,
        key_value_states,
        attention_mask=None,
        position_bias=None,
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        # cross-attention: Q comes from hidden_states, K/V from encoder output
        attn_output, position_bias = self.EncDecAttention(
            normed_hidden_states,
            mask=attention_mask,
            key_value_states=key_value_states,
            position_bias=position_bias,
        )

        hidden_states += attn_output
        return hidden_states, position_bias

class T5Block(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.layer = nn.ModuleList()
        self.layer.append(
            T5LayerSelfAttention(config, has_relative_attention_bias=has_relative_attention_bias)
        )
        
        # always False for now
        if self.is_decoder:
            self.layer.append(T5LayerCrossAttention(config))

        self.layer.append(T5LayerFF(config))

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
    ):
        # self attention
        hidden_states, position_bias = self.layer[0](
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
        )

        # feed forward
        hidden_states = self.layer[-1](hidden_states)
        return hidden_states, position_bias

# encoder / decoder stack
class T5Stack(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.is_decoder = config.is_decoder     # always false

        self.block = nn.ModuleList(
            [T5Block(config, has_relative_attention_bias=bool(i == 0)) for i in range(config.num_layers)]
        )
        self.final_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)

    # removing the decoder and head masking stuff (i believe that is for experimental purposes)
    # also removing embedding lookup inside this
    def forward(
        self, 
        x,                                      # input embeddings (bs, seq_len, embed_dim)
        attention_mask=None,                    # padding mask (bs, seq_len)
        intermediate_out_layer_idex=None,       # intermediate layer idx whose output is needed (if not final)
        final_layer_norm_intermediate=True,
        **kwargs
    ):
        mask = None
        if attention_mask is not None:
            # (bs, seq_len) -> (bs, 1, seq_len, seq_len)
            mask = 1.0 - attention_mask.to(x.dtype).reshape((attention_mask.shape[0], 1, -1, attention_mask.shape[-1])).expand(attention_mask.shape[0], 1, attention_mask.shape[-1], attention_mask.shape[-1])
            mask = mask.masked_fill(mask.to(torch.bool), -torch.finfo(x.dtype).max)

        intermediate = None
        past_bias = None
        
        if intermediate_out_layer_idex is not None:
            # wrapping neg idx from the end, pythonic way
            if intermediate_out_layer_idex < 0:
                intermediate_out_layer_idex = len(self.block) + intermediate_out_layer_idex

        for i, l in enumerate(self.block):
            x, past_bias = l(x, mask, past_bias)
            if i == intermediate_out_layer_idex:
                intermediate = x.clone()
        
        x = self.final_layer_norm(x)
        
        if intermediate is not None and final_layer_norm_intermediate:
            intermediate = self.final_layer_norm(intermediate)
        
        return x, intermediate


class T5(TextEncoder):
    def __init__(self, model_path: str, config = T5Config(), te_type = TextEncoderType.T5_BASE.value, **kwargs):
        kwargs["enable_attention_mask"] = True
        kwargs["zero_out_masked"] = True
        kwargs["layer"] = kwargs.get("layer", "last")
        kwargs["layer_idx"] = kwargs.get("layer_idx", -2)
        super().__init__(model_path, config, te_type, **kwargs)
        
        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        self.encoder = T5Stack(config)

    def get_input_embeddings(self):
        return self.shared

    # TODO: looks redundant
    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings

    # TODO: need to look into the weight tying as done in the og transformers impl.

    def forward(
        self,
        input_ids,
        attention_mask: Optional[torch.FloatTensor] = None,
        input_embeds: Optional[torch.Tensor] = None,
        **kwargs
    ) :
        assert input_ids is not None or input_embeds is not None, "need either input IDs or the embeds to encode"
        
        if input_ids is None: 
            x = input_embeds
        else: 
            x = self.shared(input_ids)
        
        if self.dtype not in [torch.float32, torch.float16, torch.bfloat16]:
            x = torch.nan_to_num(x) #Fix for fp8 T5 base
        return self.encoder(x, attention_mask=attention_mask, **kwargs)


class T5XXL(T5):
    def __init__(self, model_path, config = T5XXLConfig()):
        super().__init__(model_path, config, TextEncoderType.T5_XXL.value)

  
class UMT5XXL(T5):
    def __init__(self, model_path, config = UMT5XXLConfig()):
        super().__init__(model_path, config, TextEncoderType.T5_XXL.value)
        