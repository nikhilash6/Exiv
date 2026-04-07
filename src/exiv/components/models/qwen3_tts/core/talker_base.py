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

import time
import torch
from torch import nn

from typing import Optional
from transformers.cache_utils import Cache, DynamicCache
from tqdm import tqdm
from .....utils.logging import app_logger

from .common_modules import Qwen3TTSTalkerResizeMLP, Qwen3TTSRMSNorm, Qwen3TTSTalkerConfig, Qwen3TTSTalkerRotaryEmbedding, Qwen3TTSTalkerTextMLP, rotate_half
from .subtalker_base import Qwen3TTSSubTalker
from .config import Qwen3TTSConfig, Qwen3TTSTalkerConfig
from ...common import AROutput
from ....attention import eager_attention_forward, repeat_kv
from ....audio_encoders.qwen3_tts_speaker_encoder import Qwen3TTSSpeakerEncoder, mel_spectrogram
from .....components.attention import create_attention_mask
from .....model_utils.autoregressive_model_mixin import ARModelMixin
from .....model_utils.autoregressive_generation_utils import (
    ARSampler,
    LogitsProcessor,
    StoppingCriteria,
    apply_logits_processors,
    build_generation_components,
    check_stopping_criteria,
)


def apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section, mrope_interleaved=False, unsqueeze_dim=1):
    # mrope_section keeps this dynamic, like text can be allocated different channel slices in different scenarios
    # interleaving seems to be for memory coalescing 
    if mrope_interleaved:
        def apply_interleaved_rope(x, modality_num):
            x_t = x[0].clone()
            index_ranges = []
            for i, n in enumerate(mrope_section[1:], 1):
                beg_idx = i
                end_idx = n * modality_num
                index_ranges.append((beg_idx, end_idx))
            for beg_idx, end_idx in index_ranges:
                x_t[..., beg_idx:end_idx:modality_num] = x[beg_idx, ..., beg_idx:end_idx:modality_num]
            return x_t

        dim = cos.shape[-1]
        modality_num = len(mrope_section)
        cos = torch.cat([apply_interleaved_rope(cos[..., : dim // 2], modality_num)] * 2, dim=-1).unsqueeze(
            unsqueeze_dim
        )
        sin = torch.cat([apply_interleaved_rope(sin[..., : dim // 2], modality_num)] * 2, dim=-1).unsqueeze(
            unsqueeze_dim
        )
    else:
        mrope_section = mrope_section * 2
        cos = torch.cat([m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
            unsqueeze_dim
        )
        sin = torch.cat([m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
            unsqueeze_dim
        )

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class Qwen3TTSTalkerAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
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
        self.q_norm = Qwen3TTSRMSNorm(
            self.head_dim, eps=config.rms_norm_eps
        )  # unlike olmo, only on the head dim!
        self.k_norm = Qwen3TTSRMSNorm(
            self.head_dim, eps=config.rms_norm_eps
        )  # thus post q_norm does not need reshape
        self.sliding_window = getattr(config, "sliding_window", None)
        self.rope_scaling = config.rope_scaling

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states, key_states, cos, sin, self.rope_scaling["mrope_section"], self.rope_scaling["interleaved"]
        )

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)
            

        attn_output = eager_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask=attention_mask,
            heads=self.config.num_attention_heads,
            sliding_window=self.sliding_window,  # diff with Llama
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output

class Qwen3TTSTalkerDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3TTSTalkerAttention(config, layer_idx)

        self.mlp = Qwen3TTSTalkerTextMLP(config, intermediate_size=config.intermediate_size)

        self.input_layernorm = Qwen3TTSRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3TTSRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,            # no use
        past_key_values: Optional[tuple[torch.Tensor]] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs,
    ) -> tuple[torch.FloatTensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence.
            position_embeddings (`tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """

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
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        outputs = (hidden_states,)
        return outputs

# Talker base model: transformer backbone for main speech generation
class Qwen3TTSTalkerModel(ARModelMixin):

    def __init__(self, config, dtype=torch.float32, **kwargs):
        super().__init__(dtype=dtype, **kwargs)
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.layers = nn.ModuleList(
            [Qwen3TTSTalkerDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3TTSRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3TTSTalkerRotaryEmbedding(config)
        self.codec_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.text_embedding = nn.Embedding(config.text_vocab_size, config.text_hidden_size)

    def get_input_embeddings(self):
        return self.codec_embedding
    
    def get_text_embeddings(self):
        return self.text_embedding

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[list[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs,
    ) -> AROutput:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )


        # the hard coded `3` is for temporal, height and width.
        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
        

        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            text_position_ids = position_ids[0]
        
        # Determine sequence lengths for mask creation
        query_len = inputs_embeds.shape[1]
        kv_offset = past_key_values.get_seq_length() if past_key_values is not None else 0
        kv_len = kv_offset + query_len
        device = inputs_embeds.device
        
        # TODO: look into this, added during dev
        # For generation with KV cache, we need a mask that allows the query to see all cached positions
        # The query position is offset by kv_offset (the cache length before this query)
        if query_len == 1 and kv_offset > 0:
            # Single token generation with cache - allow seeing all cached positions
            # Create mask of shape [1, 1, query_len, kv_len]
            # All cached positions (0 to kv_offset-1) should be visible (0)
            # The query itself should also be visible
            mask = torch.zeros(1, 1, query_len, kv_len, device=device, dtype=inputs_embeds.dtype)
            causal_mask = mask
        else:
            causal_mask = create_attention_mask(
                query_len, 
                kv_len, 
                device,
                sliding_window=self.config.sliding_window,
                dtype=inputs_embeds.dtype
            )
        
        

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=text_position_ids,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **flash_attn_kwargs,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return AROutput(
            logits=None,
            extra={
                "last_hidden_state":hidden_states,
                "past_key_values":past_key_values,
                "hidden_states":all_hidden_states,
                "attentions":all_self_attns,
            }
        )

# generates first VQ codebook (code 0), calls subtalker for residual codes (1-15)
class Qwen3TTSTalker(ARModelMixin):
    def __init__(self, config: Qwen3TTSTalkerConfig, dtype=torch.float32, **kwargs):
        super().__init__(dtype=dtype, **kwargs)
        self.config = config
        self.model = Qwen3TTSTalkerModel(config, dtype=dtype, **kwargs)
        self.vocab_size = config.vocab_size
        self.text_projection = Qwen3TTSTalkerResizeMLP(
            config.text_hidden_size, config.text_hidden_size, config.hidden_size, config.hidden_act, bias=True
        )

        self.codec_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.code_predictor = Qwen3TTSSubTalker(
            config=config.code_predictor_config,
            talker_config=config,
            dtype=dtype,
            **kwargs
        )
        self.rope_deltas = None

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def get_text_embeddings(self):
        return self.model.get_text_embeddings()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[list[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> AROutput:
        """
        Single forward pass through the talker model.
        
        This is a pure forward pass - no generation loop, no code_predictor calls.
        For generation, use generate() instead.
        
        Args:
            inputs_embeds: Input embeddings [batch, seq_len, hidden_size]
            attention_mask: Attention mask
            past_key_values: Cached KV states
            use_cache: Whether to return updated KV cache
            
        Returns:
            AROutput with logits, past_key_values, and last_hidden_state
        """
        if (input_ids is None) and (inputs_embeds is None):
            raise ValueError("You must specify either input_ids or inputs_embeds")
        
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
        
        # Handle position_ids/rope for the first call
        if attention_mask is not None:
            if (
                cache_position is None
                or (cache_position is not None and cache_position[0] == 0)
                or self.rope_deltas is None
            ):
                delta0 = (1 - attention_mask).sum(dim=-1).unsqueeze(1)
                position_ids, rope_deltas = self.get_rope_index(attention_mask)
                rope_deltas = rope_deltas - delta0
                self.rope_deltas = rope_deltas
            else:
                batch_size, seq_length = inputs_embeds.shape[0], inputs_embeds.shape[1]
                delta = cache_position[0] + self.rope_deltas if cache_position is not None else 0
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        outputs = self.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.extra.get("last_hidden_state")
        logits = self.codec_head(hidden_states)

        return AROutput(
            logits=logits,
            past_key_values=outputs.extra.get("past_key_values"),
            hidden_states=outputs.extra.get("hidden_states"),
            extra={
                "attentions": outputs.extra.get("attentions"),
                "last_hidden_state": hidden_states,
            }
        )

    def generate(
        self,
        inputs_embeds: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor] = None,
        trailing_text_hidden: Optional[torch.Tensor] = None,
        tts_pad_embed: Optional[torch.Tensor] = None,
        max_new_tokens: int = 2048,
        min_new_tokens: int = 2,
        do_sample: bool = True,
        temperature: float = 0.9,
        top_k: int = 50,
        top_p: float = 1.0,
        eos_token_id: Optional[int] = None,
        repetition_penalty: float = 1.05,
        subtalker_dosample: bool = True,
        subtalker_top_k: int = 50,
        subtalker_top_p: float = 1.0,
        subtalker_temperature: float = 0.9,
        # optional
        logits_processors: list[LogitsProcessor] | None = None,
        stopping_criteria: list[StoppingCriteria] | None = None,
        sampler: Optional[ARSampler] = None,
        subtalker_logits_processors: list[LogitsProcessor] | None = None,
        subtalker_sampler: Optional[ARSampler] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate codec tokens autoregressively.
        
        This coordinates the talker (generates first codebook) with code_predictor
        (generates residual codebooks) at each step.
        
        Args:
            inputs_embeds: Initial input embeddings [batch, seq_len, hidden_size]
            attention_mask: Attention mask for initial sequence
            trailing_text_hidden: Text embeddings to add at each step
            tts_pad_embed: Padding embedding to use when trailing_text_hidden is exhausted
            max_new_tokens: Maximum number of codec tokens to generate
            min_new_tokens: Minimum tokens before allowing EOS
            do_sample, temperature, top_k, top_p, repetition_penalty: Sampling parameters
            subtalker_*: Parameters for code_predictor sampling
            eos_token_id: Token ID that stops generation
            logits_processors: Custom logits processors list for talker (overrides temp/top_k/top_p/penalty)
            stopping_criteria: Custom stopping criteria list (overrides max_new_tokens/eos/min_new_tokens)
            sampler: Custom sampler for talker (overrides do_sample)
            subtalker_logits_processors: Custom logits processors list for subtalker
            subtalker_sampler: Custom sampler for subtalker
            
        Returns:
            Generated codec token IDs [batch, num_generated_tokens, num_code_groups]
        """
        batch_size = inputs_embeds.shape[0]
        device = inputs_embeds.device
        
        # Set default EOS token
        if eos_token_id is None:
            eos_token_id = self.config.codec_eos_token_id
        
        # Ensure trailing_text_hidden and tts_pad_embed are provided
        if trailing_text_hidden is None:
            trailing_text_hidden = torch.zeros(batch_size, 1, self.config.hidden_size, device=device, dtype=self.dtype)
        if tts_pad_embed is None:
            tts_pad_embed = torch.zeros(batch_size, 1, self.config.hidden_size, device=device, dtype=self.dtype)
        
        # Track generated codec sequences
        all_codec_ids = []
        prompt_length = 0
        
        # Initialize generation state
        past_key_values = None
        current_embeds = inputs_embeds
        current_seq_len = inputs_embeds.shape[1]
        
        # default generation components
        if logits_processors is None or stopping_criteria is None or sampler is None:
            default_processors, default_criteria, default_sampler = build_generation_components(
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                prompt_length=prompt_length,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                eos_token_id=eos_token_id,
                use_last_token_repetition_penalty=True,  # Qwen3-TTS style: only penalize last token
            )
            logits_processors = logits_processors or default_processors
            stopping_criteria = stopping_criteria or default_criteria
            sampler = sampler or default_sampler
        
        pbar = tqdm(total=max_new_tokens, desc="Generating audio", unit="tok", ncols=80)
        for step in range(max_new_tokens):
            # Calculate cache_position for proper RoPE encoding
            if step == 0:
                cache_position = None  # First step uses full sequence
            else:
                cache_position = torch.tensor([current_seq_len + step - 1], device=device, dtype=torch.long)
            
            # Forward pass through talker (single step)
            outputs = self(
                inputs_embeds=current_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                cache_position=cache_position,
            )
            
            logits = outputs.logits
            past_key_values = outputs.past_key_values
            past_hidden = outputs.extra["last_hidden_state"][:, -1:, :]  # [batch, 1, hidden]
            
            # Get logits for last position
            next_token_logits = logits[:, -1, :].clone()
            
            # Build first codebook history for repetition penalty (before generating new token)
            prev_first_codes = None
            if all_codec_ids:
                prev_first_codes = torch.stack([ids[:, 0] for ids in all_codec_ids], dim=1)
            
            next_token_logits = apply_logits_processors(logits_processors, prev_first_codes, next_token_logits)
            
            # Sample next token for first codebook
            next_token = sampler(next_token_logits)
            
            # Get embedding for the predicted first code
            last_id_hidden = self.get_input_embeddings()(next_token)
            
            if subtalker_logits_processors is None or subtalker_sampler is None:
                st_processors, _, st_sampler = build_generation_components(
                    max_new_tokens=self.config.num_code_groups - 1,
                    do_sample=subtalker_dosample,
                    temperature=subtalker_temperature,
                    top_k=subtalker_top_k,
                    top_p=subtalker_top_p,
                )
                subtalker_logits_processors = subtalker_logits_processors or st_processors
                subtalker_sampler = subtalker_sampler or st_sampler
            
            # Run code predictor to get remaining 15 codes
            generated_ids = self.code_predictor.generate(
                inputs_embeds=torch.cat((past_hidden, last_id_hidden), dim=1),
                max_new_tokens=self.config.num_code_groups - 1,
                logits_processors=subtalker_logits_processors,
                sampler=subtalker_sampler,
            )
            
            # Combine predicted first code + generated residual codes
            codec_ids = torch.cat((next_token, generated_ids), dim=-1)
            all_codec_ids.append(codec_ids)
            
            # Build current sequence for stopping criteria (include newly generated token)
            if prev_first_codes is not None:
                current_sequence = torch.cat([prev_first_codes, next_token], dim=1)
            else:
                current_sequence = next_token
            
            # stopping criteria
            should_stop = check_stopping_criteria(stopping_criteria, current_sequence, step=step)
            pbar.update(1)
            pbar.set_postfix({'tokens': len(all_codec_ids)})
            if should_stop.all():
                break
            
            # Prepare embeddings for next iteration
            codec_hiddens = torch.cat(
                [last_id_hidden]
                + [self.code_predictor.get_input_embeddings()[i](generated_ids[..., i:i+1]) for i in range(self.config.num_code_groups - 1)],
                dim=1,
            )
            next_embeds = codec_hiddens.sum(1, keepdim=True)
            
            # Add trailing text embedding
            if step < trailing_text_hidden.shape[1]:
                next_embeds = next_embeds + trailing_text_hidden[:, step:step+1, :]
            else:
                next_embeds = next_embeds + tts_pad_embed
            
            current_embeds = next_embeds
        
        pbar.close()
        
        # Stack all generated codec IDs
        return torch.stack(all_codec_ids, dim=1)  # [batch, seq_len, num_code_groups]

    def get_rope_index(
        self,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        position_ids = attention_mask.float().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
        max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
        mrope_position_deltas = max_position_ids + 1 - torch.sum(attention_mask, dim=-1, keepdim=True)

        return position_ids, mrope_position_deltas

# Main TTS model: orchestrates text-to-speech generation, manages speaker/language prompts
class Qwen3TTSBase(ARModelMixin):

    def __init__(self, config: Qwen3TTSConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config

        self.talker = Qwen3TTSTalker(self.config.talker_config, **kwargs)

        if config.tts_model_type == "base":
            self.speaker_encoder = Qwen3TTSSpeakerEncoder(self.config.speaker_encoder_config)
        else:
            self.speaker_encoder = None

        self.speech_tokenizer = None
        self.generate_config = None

        self.supported_speakers = self.config.talker_config.spk_id.keys()
        self.supported_languages = ["auto"]
        for language_id in self.config.talker_config.codec_language_id.keys():
            if "dialect" not in language_id:
                self.supported_languages.append(language_id)
        
        self.speaker_encoder_sample_rate = self.config.speaker_encoder_config.sample_rate
        self.tokenizer_type = self.config.tokenizer_type
        self.tts_model_size = self.config.tts_model_size
        self.tts_model_type = self.config.tts_model_type
    
    def load_speech_tokenizer(self, speech_tokenizer):
        self.speech_tokenizer = speech_tokenizer
    
    def load_generate_config(self, generate_config):
        self.generate_config = generate_config
    
    def get_supported_speakers(self):
        return self.supported_speakers
    
    def get_supported_languages(self):
        return self.supported_languages
    
    @property
    def eos_token_id(self) -> int:
        """Return the codec EOS token ID used to stop audio generation."""
        return self.config.talker_config.codec_eos_token_id
    
    def extract_speaker_embedding(self, audio, sr):
        assert sr == 24000, "Only support 24kHz audio"
        mels = mel_spectrogram(
            torch.from_numpy(audio).unsqueeze(0), 
            n_fft=1024, 
            num_mels=128, 
            sampling_rate=24000,
            hop_size=256, 
            win_size=1024, 
            fmin=0, 
            fmax=12000
        ).transpose(1, 2)
        speaker_embedding = self.speaker_encoder(mels.to(self.gpu_device).to(self.dtype))[0]
        return speaker_embedding
    
    def generate_speaker_prompt(
        self,
        voice_clone_prompt: dict
    ):
        """Extract speaker embeddings from voice clone prompt dict."""
        voice_clone_spk_embeds = []
        for index in range(len(voice_clone_prompt['ref_spk_embedding'])):
            ref_spk_embedding = voice_clone_prompt["ref_spk_embedding"][index].to(self.gpu_device).to(self.talker.dtype)            
            voice_clone_spk_embeds.append(ref_spk_embedding)
        
        return voice_clone_spk_embeds

    def generate_icl_prompt(
        self,
        text_id: torch.Tensor,
        ref_id: torch.Tensor,
        ref_code: torch.Tensor,
        tts_pad_embed: torch.Tensor,
        tts_eos_embed: torch.Tensor,
        non_streaming_mode: bool,
    ):
        # text embed (ref id + text id + eos) 1 T1 D
        text_embed = self.talker.text_projection(
            self.talker.get_text_embeddings()(torch.cat([ref_id, text_id], 
                                                            dim=-1)))
        text_embed = torch.cat([text_embed, tts_eos_embed], dim=1)
        # codec embed (codec bos + codec) 1 T2 D
        codec_embed = []
        for i in range(self.talker.config.num_code_groups):
            if i == 0:
                codec_embed.append(self.talker.get_input_embeddings()(ref_code[:, :1]))
            else:
                codec_embed.append(self.talker.code_predictor.get_input_embeddings()[i-1](ref_code[:, i:i+1]))
        codec_embed = torch.cat(codec_embed, dim=1).sum(1).unsqueeze(0)
        codec_embed = torch.cat([self.talker.get_input_embeddings()(
                                    torch.tensor(
                                        [[
                                            self.config.talker_config.codec_bos_id,
                                        ]],
                                        device=self.gpu_device,
                                        dtype=text_id.dtype,
                                    )
                                ), codec_embed], dim=1)
        # compute lens
        text_lens = text_embed.shape[1]
        codec_lens = codec_embed.shape[1]
        if non_streaming_mode:
            icl_input_embed = text_embed + self.talker.get_input_embeddings()(
                                                torch.tensor(
                                                    [[
                                                        self.config.talker_config.codec_pad_id,
                                                    ] * text_lens],
                                                    device=self.gpu_device,
                                                    dtype=text_id.dtype,
                                                )
                                            )
            icl_input_embed = torch.cat([icl_input_embed, codec_embed + tts_pad_embed], dim=1)
            return icl_input_embed, tts_pad_embed
        else:
            if text_lens > codec_lens:
                return text_embed[:, :codec_lens] + codec_embed, text_embed[:, codec_lens:]
            else:
                text_embed = torch.cat([text_embed] + [tts_pad_embed] * (codec_lens - text_lens), dim=1)
                return text_embed + codec_embed, tts_pad_embed
    
    def _split_into_chunks(self, text_ids: torch.Tensor, max_chunk_size: int = 800) -> list[torch.Tensor]:
        """
        Split text into chunks for processing very long texts.
        
        WARNING: Chunking breaks voice continuity! Only use when text is too long
        for single-pass generation (would cause OOM).
        
        Args:
            text_ids: Text token IDs [1, seq_len]
            max_chunk_size: Maximum tokens per chunk (default 800 to leave room for audio)
            
        Returns:
            List of text token chunks
        """
        seq_len = text_ids.shape[1]
        
        # Short text - no splitting needed
        if seq_len <= max_chunk_size:
            return [text_ids]
        
        # Split into larger chunks to minimize voice discontinuity
        # Bigger chunks = fewer boundaries = better quality
        num_chunks = max(2, (seq_len + max_chunk_size - 1) // max_chunk_size)
        chunk_size = seq_len // num_chunks
        
        chunks = []
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, seq_len) if i < num_chunks - 1 else seq_len
            chunk = text_ids[:, start_idx:end_idx]
            chunks.append(chunk)
        
        return chunks
    
    def _merge_audio_codes(self, codes_list: list[torch.Tensor]) -> torch.Tensor:
        """Merge multiple audio code sequences by simple concatenation along time dimension."""
        if len(codes_list) == 1:
            return codes_list[0]
        
        # All tensors should be [batch=1, seq_len, num_code_groups]
        # Concatenate along dim=1 (seq_len)
        return torch.cat(codes_list, dim=1)
    
    def _generate_chunked(
        self,
        input_ids: list[torch.Tensor],
        instruct_ids: Optional[list[torch.Tensor]] = None,
        ref_ids: Optional[list[torch.Tensor]] = None,
        voice_clone_prompt: list[dict] = None,
        languages: list[str] = None,
        speakers: list[str] = None,
        non_streaming_mode = False,
        max_new_tokens: int = 4096,
        do_sample: bool = True,
        top_k: int = 50,
        top_p: float = 1.0,
        temperature: float = 0.9,
        subtalker_dosample: bool = True,
        subtalker_top_k: int = 50,
        subtalker_top_p: float = 1.0,
        subtalker_temperature: float = 0.9,
        eos_token_id: Optional[int] = None,
        repetition_penalty: float = 1.05,
        **kwargs,
    ) -> tuple[list[torch.Tensor], None]:
        """
        Generate audio by splitting text into sentences and processing each independently.
        
        This is MUCH faster for long texts because:
        1. Each sentence is short (~20-40 tokens)
        2. No KV cache needed (each sentence starts fresh)
        3. Memory usage stays constant
        4. Can be easily parallelized (future enhancement)
        
        Args:
            input_ids: List of text token IDs
            
        Returns:
            List of merged audio codes for each batch item
        """
        if languages is None:
            languages = ["auto"] * len(input_ids)
        if speakers is None:
            speakers = [None] * len(input_ids)
        if instruct_ids is None:
            instruct_ids = [None] * len(input_ids)
        if ref_ids is None:
            ref_ids = [None] * len(input_ids)
        
        all_merged_codes = []
        chunk_start_time = time.time()
        
        for batch_idx, (text_ids, instruct_id, ref_id, language, speaker) in enumerate(
            zip(input_ids, instruct_ids, ref_ids, languages, speakers)
        ):
            # Split into chunks (sentences)
            chunks = self._split_into_chunks(text_ids)
            app_logger.info(f"[TTS Chunked] Batch item {batch_idx}: {len(chunks)} chunks")
            
            # Handle voice_clone_prompt - pass the dict directly
            # generate_speaker_prompt expects a dict, not a list
            vcp_for_batch = voice_clone_prompt if isinstance(voice_clone_prompt, dict) else None
            
            if len(chunks) == 1:
                # Single chunk, use normal generation
                chunk_codes, _ = self.generate(
                    input_ids=[chunks[0]],
                    instruct_ids=[instruct_id],
                    ref_ids=[ref_id],
                    voice_clone_prompt=vcp_for_batch,
                    languages=[language],
                    speakers=[speaker],
                    non_streaming_mode=non_streaming_mode,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    top_k=top_k,
                    top_p=top_p,
                    temperature=temperature,
                    subtalker_dosample=subtalker_dosample,
                    subtalker_top_k=subtalker_top_k,
                    subtalker_top_p=subtalker_top_p,
                    subtalker_temperature=subtalker_temperature,
                    eos_token_id=eos_token_id,
                    repetition_penalty=repetition_penalty,
                    enable_chunking=False,  # Disable to prevent recursion
                    **kwargs,
                )
                # chunk_codes[0] is [seq_len, num_code_groups], squeeze to remove any extra dims
                all_merged_codes.append(chunk_codes[0].squeeze())
            else:
                # Generate each chunk independently
                chunk_codes_list = []
                total_chunks = len(chunks)
                
                for chunk_idx, chunk in enumerate(chunks):
                    chunk_start = time.time()
                    app_logger.info(f"[TTS Chunked] Generating chunk {chunk_idx+1}/{total_chunks}...")
                    
                    # Estimate max tokens for this chunk
                    chunk_max_tokens = max_new_tokens // total_chunks + 100
                    
                    chunk_code, _ = self.generate(
                        input_ids=[chunk],
                        instruct_ids=[instruct_id],
                        ref_ids=[ref_id],
                        voice_clone_prompt=vcp_for_batch,
                        languages=[language],
                        speakers=[speaker],
                        non_streaming_mode=non_streaming_mode,
                        max_new_tokens=chunk_max_tokens,
                        do_sample=do_sample,
                        top_k=top_k,
                        top_p=top_p,
                        temperature=temperature,
                        subtalker_dosample=subtalker_dosample,
                        subtalker_top_k=subtalker_top_k,
                        subtalker_top_p=subtalker_top_p,
                        subtalker_temperature=subtalker_temperature,
                        eos_token_id=eos_token_id,
                        repetition_penalty=repetition_penalty,
                        enable_chunking=False,  # Disable to prevent recursion
                        **kwargs,
                    )
                    
                    chunk_time = time.time() - chunk_start
                    # chunk_code[0] is [seq_len, num_code_groups]
                    # Add batch dim for merging: [1, seq_len, num_code_groups]
                    chunk_tensor = chunk_code[0].unsqueeze(0)
                    app_logger.info(f"[TTS Chunked] Chunk {chunk_idx+1}/{total_chunks} done: "
                                   f"shape={chunk_tensor.shape}, time={chunk_time:.2f}s")
                    
                    chunk_codes_list.append(chunk_tensor)
                
                # Merge all chunk codes along time dimension
                merged_codes = self._merge_audio_codes(chunk_codes_list)  # [1, total_seq_len, 16]
                # Squeeze batch dimension for decoder compatibility
                all_merged_codes.append(merged_codes.squeeze(0))
        
        total_time = time.time() - chunk_start_time
        app_logger.info(f"[TTS Chunked] Total time: {total_time:.2f}s")
        
        return all_merged_codes, None
    
    def generate(
        self,
        input_ids: Optional[list[torch.Tensor]] = None,
        instruct_ids: Optional[list[torch.Tensor]] = None,
        ref_ids: Optional[list[torch.Tensor]] = None,
        voice_clone_prompt: list[dict] = None,
        languages: list[str] = None,
        speakers: list[str] = None,
        non_streaming_mode = False,
        max_new_tokens: int = 4096,
        do_sample: bool = True,
        top_k: int = 50,
        top_p: float = 1.0,
        temperature: float = 0.9,
        subtalker_dosample: bool = True,
        subtalker_top_k: int = 50,
        subtalker_top_p: float = 1.0,
        subtalker_temperature: float = 0.9,
        eos_token_id: Optional[int] = None,
        repetition_penalty: float = 1.05,
        enable_chunking: bool = True,
        **kwargs,
    ):
        """
        Generate audio from text with optional automatic chunking for long inputs.
        
        Args:
            enable_chunking: Whether to enable automatic text chunking for long inputs.
                            When enabled, splits text into sentence-sized chunks (~40 tokens)
                            and generates each independently (faster for long texts).
        """
        # Check if chunking is needed (only for very long texts that would OOM)
        if enable_chunking and input_ids is not None:
            max_input_len = max(ids.shape[1] for ids in input_ids)
            if max_input_len > 1500:  # Only chunk very long texts - chunking breaks voice continuity!
                app_logger.warning(f"[TTS Generation] Text too long ({max_input_len} tokens), using chunked generation. "
                                  f"WARNING: Voice may vary between chunks!")
                return self._generate_chunked(
                input_ids=input_ids,
                instruct_ids=instruct_ids,
                ref_ids=ref_ids,
                voice_clone_prompt=voice_clone_prompt,
                languages=languages,
                speakers=speakers,
                non_streaming_mode=non_streaming_mode,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                subtalker_dosample=subtalker_dosample,
                subtalker_top_k=subtalker_top_k,
                subtalker_top_p=subtalker_top_p,
                subtalker_temperature=subtalker_temperature,
                eos_token_id=eos_token_id,
                repetition_penalty=repetition_penalty,
                **kwargs,
            )
        
        talker_kwargs = {
            "max_new_tokens": max_new_tokens,
            "min_new_tokens": 2,
            "do_sample": do_sample,
            "top_k": top_k,
            "top_p": top_p,
            "temperature": temperature,
            "subtalker_dosample": subtalker_dosample, 
            "subtalker_top_k": subtalker_top_k,
            "subtalker_top_p": subtalker_top_p,
            "subtalker_temperature": subtalker_temperature,
            "eos_token_id": eos_token_id if eos_token_id is not None else self.eos_token_id,
            "repetition_penalty": repetition_penalty,
            "suppress_tokens": [
                i
                for i in range(self.config.talker_config.vocab_size - 1024, self.config.talker_config.vocab_size)
                if i not in (self.config.talker_config.codec_eos_token_id,)
            ],
            "output_hidden_states": getattr(kwargs, "output_hidden_states", True),
            "return_dict_in_generate": getattr(kwargs, "return_dict_in_generate", True)
        }
        
        talker_input_embeds = [[] for _ in range(len(input_ids))]

        voice_clone_spk_embeds = None
        # voice clone speaker prompt generate
        if voice_clone_prompt is not None:
            voice_clone_spk_embeds = self.generate_speaker_prompt(voice_clone_prompt)
        
        # instruct text prompt generate
        if instruct_ids is not None:
            for index, instruct_id in enumerate(instruct_ids):
                if instruct_id is not None:
                    talker_input_embeds[index].append(self.talker.text_projection(
                                                  self.talker.get_text_embeddings()(instruct_id)))

        # tts text prompt generate
        trailing_text_hiddens = []
        if speakers is None:
            speakers = [None] * len(input_ids)
        for index, (input_id, language, speaker) in enumerate(zip(input_ids, languages, speakers)):
            if voice_clone_spk_embeds is None:
                if speaker == "" or speaker == None: # Instruct create speaker
                    speaker_embed = None
                else:
                    if speaker.lower() not in self.config.talker_config.spk_id:
                        raise NotImplementedError(f"Speaker {speaker} not implemented")
                    else:
                        spk_id = self.config.talker_config.spk_id[speaker.lower()]
                        speaker_embed = self.talker.get_input_embeddings()(
                                            torch.tensor(
                                                spk_id,
                                                device=self.gpu_device,
                                                dtype=input_id.dtype,
                                            )
                                        )
            else:
                if voice_clone_prompt["x_vector_only_mode"][index] or voice_clone_prompt["icl_mode"][index]:
                    speaker_embed = voice_clone_spk_embeds[index]
                else:
                    speaker_embed = None

            assert language is not None

            if language.lower() == "auto":
                language_id = None
            else:
                if language.lower() not in self.config.talker_config.codec_language_id:
                    raise NotImplementedError(f"Language {language} not implemented")
                else:
                    language_id = self.config.talker_config.codec_language_id[language.lower()]
            
            if (language.lower() in ["chinese", "auto"] and \
                   speaker != "" and speaker is not None and \
                     self.config.talker_config.spk_is_dialect[speaker.lower()] != False):
                dialect = self.config.talker_config.spk_is_dialect[speaker.lower()]
                language_id = self.config.talker_config.codec_language_id[dialect]
            
            tts_bos_embed, tts_eos_embed, tts_pad_embed = self.talker.text_projection(
                self.talker.get_text_embeddings()(
                    torch.tensor(
                        [[self.config.tts_bos_token_id, self.config.tts_eos_token_id, self.config.tts_pad_token_id]],
                        device=self.gpu_device,
                        dtype=input_id.dtype,
                    )
                )
            ).chunk(3, dim=1)  # 3 * [1 1 d]
            
            # codec: tag and speaker
            if language_id is None:
                codec_prefill_list = [[
                                        self.config.talker_config.codec_nothink_id,
                                        self.config.talker_config.codec_think_bos_id,
                                        self.config.talker_config.codec_think_eos_id,
                                    ]]
            else:
                codec_prefill_list = [[
                                        self.config.talker_config.codec_think_id,
                                        self.config.talker_config.codec_think_bos_id,
                                        language_id,
                                        self.config.talker_config.codec_think_eos_id,
                                    ]]

            codec_input_emebdding_0 = self.talker.get_input_embeddings()(
                                                    torch.tensor(
                                                        codec_prefill_list,
                                                        device=self.gpu_device,
                                                        dtype=input_id.dtype,
                                                    )
                                                )
            codec_input_emebdding_1 = self.talker.get_input_embeddings()(
                                                    torch.tensor(
                                                        [[
                                                            self.config.talker_config.codec_pad_id,
                                                            self.config.talker_config.codec_bos_id,
                                                        ]],
                                                        device=self.gpu_device,
                                                        dtype=input_id.dtype,
                                                    )
                                                )
            if speaker_embed is None:
                codec_input_emebdding = torch.cat([codec_input_emebdding_0,
                                                   codec_input_emebdding_1], dim=1)
            else:
                codec_input_emebdding = torch.cat([codec_input_emebdding_0,
                                                   speaker_embed.view(1, 1, -1),
                                                   codec_input_emebdding_1], dim=1)

            # '<|im_start|>assistant\n我叫通义千问，是阿里云的开源大模型。<|im_end|>\n<|im_start|>assistant\n'

            # <|im_start|>assistant\n
            _talker_input_embed_role = self.talker.text_projection(
                                        self.talker.get_text_embeddings()(input_id[:, :3])
                                        )

            # tts_pad * 4 + tts_bos
            _talker_input_embed = torch.cat((tts_pad_embed.expand(-1, codec_input_emebdding.shape[1] - 2, -1),
                                            tts_bos_embed,
                                            ), dim=1) + codec_input_emebdding[:, :-1]

            talker_input_embed = torch.cat((_talker_input_embed_role, _talker_input_embed), dim=1)

            if voice_clone_prompt is not None and voice_clone_prompt["ref_code"] is not None and voice_clone_prompt["icl_mode"][index]:
                icl_input_embed, trailing_text_hidden = self.generate_icl_prompt(
                    text_id=input_id[:, 3:-5],
                    ref_id=ref_ids[index][:, 3:-2],
                    ref_code=voice_clone_prompt["ref_code"][index].to(self.gpu_device),
                    tts_pad_embed=tts_pad_embed,
                    tts_eos_embed=tts_eos_embed,
                    non_streaming_mode=non_streaming_mode,
                )
                talker_input_embed = torch.cat([talker_input_embed, icl_input_embed], dim=1)
            else:
                #  tts_text_first_token
                talker_input_embed = torch.cat([talker_input_embed, 
                                                self.talker.text_projection(self.talker.get_text_embeddings()(input_id[:, 3:4])) + codec_input_emebdding[:, -1:]], 
                                                dim=1)
                if non_streaming_mode:
                    talker_input_embed = talker_input_embed[:, :-1] # 去掉原本放进去的text
                    talker_input_embed = torch.cat([talker_input_embed,
                                                    torch.cat((self.talker.text_projection(
                                                        self.talker.get_text_embeddings()(input_id[:, 3:-5])
                                                    ), tts_eos_embed), dim=1) + self.talker.get_input_embeddings()(
                                                        torch.tensor(
                                                            [[
                                                                self.config.talker_config.codec_pad_id,
                                                            ] * (input_id[:, 3:-5].shape[1] + 1)],
                                                            device=self.gpu_device,
                                                            dtype=input_id.dtype,
                                                        )
                                                    ), 
                                                    tts_pad_embed + self.talker.get_input_embeddings()(
                                                        torch.tensor(
                                                            [[
                                                                self.config.talker_config.codec_bos_id,
                                                            ]],
                                                            device=self.gpu_device,
                                                            dtype=input_id.dtype,
                                                        )
                                                    ) 
                                                    ], dim=1)
                    trailing_text_hidden = tts_pad_embed
                else:
                    # 叫通义千问，是阿里云的开源大模型。
                    trailing_text_hidden = torch.cat((self.talker.text_projection(
                                                        self.talker.get_text_embeddings()(input_id[:, 4:-5])
                                                    ), tts_eos_embed), dim=1)
            talker_input_embeds[index].append(talker_input_embed)
            trailing_text_hiddens.append(trailing_text_hidden)
        
        for index, talker_input_embed in enumerate(talker_input_embeds):
            talker_input_embeds[index] = torch.cat([item for item in talker_input_embed if item is not None], dim=1)

        # for batch inferquence
        original_lengths = torch.tensor([t.shape[1] for t in talker_input_embeds])
        # left padding for talker input embeds
        sequences = [t.squeeze(0) for t in talker_input_embeds]
        sequences_reversed = [t.flip(dims=[0]) for t in sequences]
        padded_reversed = torch.nn.utils.rnn.pad_sequence(
            sequences_reversed,
            batch_first=True,
            padding_value=0.0
        )
        talker_input_embeds = padded_reversed.flip(dims=[1])
        # generate mask
        batch_size, max_len = talker_input_embeds.shape[0], talker_input_embeds.shape[1]
        indices = torch.arange(max_len).expand(batch_size, -1)
        num_pads = max_len - original_lengths
        talker_attention_mask = (indices >= num_pads.unsqueeze(1)).long().to(talker_input_embeds.device)
        # padding trailing text hiddens
        pad_embedding_vector = tts_pad_embed.squeeze()
        sequences_to_pad = [t.squeeze(0) for t in trailing_text_hiddens]
        trailing_text_original_lengths = [s.shape[0] for s in sequences_to_pad]
        padded_hiddens = torch.nn.utils.rnn.pad_sequence(
            sequences_to_pad,
            batch_first=True,
            padding_value=0.0
        )
        arange_tensor = torch.arange(max(trailing_text_original_lengths), 
                                     device=padded_hiddens.device).expand(len(trailing_text_original_lengths), -1)
        lengths_tensor = torch.tensor(trailing_text_original_lengths, device=padded_hiddens.device).unsqueeze(1)
        padding_mask = arange_tensor >= lengths_tensor
        padded_hiddens[padding_mask] = pad_embedding_vector
        trailing_text_hiddens = padded_hiddens

        # Generate using talker
        talker_codes = self.talker.generate(
            inputs_embeds=talker_input_embeds,
            attention_mask=talker_attention_mask,
            trailing_text_hidden=trailing_text_hiddens,
            tts_pad_embed=tts_pad_embed,
            **talker_kwargs,
        )  # [batch, seq_len, num_code_groups]
        
        # Find stop tokens and trim
        first_codebook = talker_codes[:, :, 0]
        eos_token_id = talker_kwargs.get('eos_token_id', self.config.talker_config.codec_eos_token_id)
        is_stop_token = (first_codebook == eos_token_id)
        stop_indices = torch.argmax(is_stop_token.int(), dim=1)
        has_stop_token = is_stop_token.any(dim=1)
        min_new_tokens = talker_kwargs.get('min_new_tokens', 2)
        
        talker_codes_list = []
        for i in range(talker_codes.shape[0]):
            if has_stop_token[i] and stop_indices[i] >= min_new_tokens:
                talker_codes_list.append(talker_codes[i, :stop_indices[i]])
            else:
                talker_codes_list.append(talker_codes[i])
        
        return talker_codes_list, None
