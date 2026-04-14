import torch
import torch.nn as nn

from typing import List, Optional
from transformers.cache_utils import Cache, DynamicCache

from .config import Qwen3TTSConfig, Qwen3TTSTalkerCodePredictorConfig, Qwen3TTSTalkerConfig
from .common_modules import Qwen3TTSRMSNorm, Qwen3TTSRotaryEmbedding, Qwen3TTSTalkerTextMLP, apply_rotary_pos_emb
from ...common import AROutput
from ....attention import eager_attention_forward
from .....model_utils.autoregressive_model_mixin import ARModelMixin
from .....model_utils.autoregressive_generation_utils import ARSampler, LogitsProcessor, apply_logits_processors, build_generation_components
from .....components.attention import create_attention_mask


class Qwen3TTSAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Qwen3TTSConfig, layer_idx: int):
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
        self.q_norm = Qwen3TTSRMSNorm(self.head_dim, eps=config.rms_norm_eps)  # unlike olmo, only on the head dim!
        self.k_norm = Qwen3TTSRMSNorm(
            self.head_dim, eps=config.rms_norm_eps
        )  # thus post q_norm does not need reshape
        self.sliding_window = config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else None

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
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

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

class Qwen3TTSDecoderLayer(nn.Module):
    def __init__(self, config: Qwen3TTSConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = Qwen3TTSAttention(config=config, layer_idx=layer_idx)

        self.mlp = Qwen3TTSTalkerTextMLP(config)
        self.input_layernorm = Qwen3TTSRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3TTSRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention_type = config.layer_types[layer_idx]

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs,
    ) -> tuple[torch.FloatTensor]:
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

# Subtalker base model: transformer backbone for residual code prediction (codes 1-15)
class Qwen3TTSTalkerCodePredictorModel(ARModelMixin):

    def __init__(self, config: Qwen3TTSTalkerCodePredictorConfig, talker_hidden_size=None, dtype=torch.float32, **kwargs):
        super().__init__(dtype=dtype, **kwargs)
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.layers = nn.ModuleList(
            [Qwen3TTSDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3TTSRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3TTSRotaryEmbedding(config=config)
        self.has_sliding_layers = "sliding_attention" in self.config.layer_types
        # TODO: look into this, added during dev
        # use text_hidden_size for codec_embedding (matches talker's hidden size)
        # the codec_embedding projects from talker's hidden size to code predictor's input
        embed_dim = getattr(config, "text_hidden_size", config.hidden_size)
        self.codec_embedding = nn.ModuleList(
            [nn.Embedding(config.vocab_size, embed_dim) for _ in range(config.num_code_groups - 1)]
        )

        # Initialize weights and apply final processing
        # removed post_init() as it's not present in ARModelMixin

    def get_input_embeddings(self):
        return self.codec_embedding

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        cache_position=None,
        generation_steps=None,
        **flash_attn_kwargs,
    ) -> AROutput:
        if input_ids is not None:
            raise ValueError("`input_ids` is expected to be `None`")
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if inputs_embeds is None:
            raise ValueError("`inputs_embeds` must be provided since `input_ids` is expected to be None")

        # TODO (joao): remove this exception in v4.56 -- it exists for users that try to pass a legacy cache
        if not isinstance(past_key_values, (type(None), Cache)):
            raise ValueError("The `past_key_values` should be either a `Cache` object or `None`.")

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            query_len = inputs_embeds.shape[1]
            kv_offset = past_key_values.get_seq_length() if past_key_values is not None else 0
            kv_len = kv_offset + query_len
            device = inputs_embeds.device

            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_attention_mask(query_len, kv_len, device, dtype=inputs_embeds.dtype),
            }
            # The sliding window alternating layers are not always activated depending on the config
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_attention_mask(
                    query_len, kv_len, device, sliding_window=self.config.sliding_window, dtype=inputs_embeds.dtype
                )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=position_ids,
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
                "last_hidden_state" : hidden_states,
                "past_key_values" : past_key_values if use_cache else None,
                "hidden_states" : all_hidden_states,
                "attentions" : all_self_attns,
            }
        )

# Subtalker: predicts residual VQ codes (codes 1-15), wraps the base model with output heads
class Qwen3TTSSubTalker(ARModelMixin):
    config_class = Qwen3TTSTalkerCodePredictorConfig
    base_model_prefix = "talker.code_predictor"

    def __init__(self, config: Qwen3TTSTalkerCodePredictorConfig, talker_config: Qwen3TTSTalkerConfig, dtype=torch.float32, **kwargs):
        super().__init__(dtype=dtype, **kwargs)
        self.config = config
        self.model = Qwen3TTSTalkerCodePredictorModel(config, talker_hidden_size=talker_config.hidden_size, dtype=dtype, **kwargs)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.ModuleList(
            [nn.Linear(config.hidden_size, config.vocab_size, bias=False) for _ in range(config.num_code_groups - 1)]
        )

        if config.hidden_size != talker_config.hidden_size:
            self.small_to_mtp_projection = torch.nn.Linear(talker_config.hidden_size, config.hidden_size, bias=True)
        else:
            self.small_to_mtp_projection = torch.nn.Identity()

        # Initialize weights and apply final processing
        # removed post_init() as it's not present in ARModelMixin

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        cache_position=None,
        generation_steps=None,
        **kwargs,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # Prefill stage
        if inputs_embeds is not None and inputs_embeds.shape[1] > 1:
            generation_steps = inputs_embeds.shape[1] - 2  # hidden & layer 0
        # Generation stage
        else:
            # NOTE: generation_steps is passed from GenerationMixin via model_kwargs
            # It represents the current step index (0 to num_code_groups-2)
            inputs_embeds = self.model.get_input_embeddings()[generation_steps - 1](input_ids)
        inputs_embeds = self.small_to_mtp_projection(inputs_embeds)

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
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
        # TODO: lm_head indexing (look into this)
        logits = self.lm_head[generation_steps](hidden_states)

        return AROutput(
            logits=logits,
            past_key_values=outputs.extra.get("past_key_values"),
            extra={
                "last_hidden_state": hidden_states,
                "hidden_states": outputs.extra.get("hidden_states"),
                "attentions": outputs.extra.get("attentions"),
                "generation_steps": generation_steps + 1,
            }
        )

    def generate(
        self,
        inputs_embeds: torch.FloatTensor,
        max_new_tokens: int,
        do_sample: bool = True,
        temperature: float = 0.9,
        top_k: int = 50,
        top_p: float = 1.0,
        logits_processors: Optional[List[LogitsProcessor]] = None,
        sampler: Optional[ARSampler] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Custom generation for code predictor.
        
        Unlike standard AR models, this uses a different lm_head at each step
        (indexed by generation_steps) to predict residual codes.
        
        Args:
            inputs_embeds: Input embeddings [batch, seq_len, hidden_dim]
            max_new_tokens: Number of residual codes to generate
            do_sample: Whether to use sampling
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Top-p filtering
            logits_processors: Custom logits processors list (overrides temp/top_k/top_p)
            sampler: Custom sampler (overrides do_sample)
            
        Returns:
            Generated token IDs [batch, max_new_tokens]
        """
        batch_size = inputs_embeds.shape[0]
        device = inputs_embeds.device
        
        generated_tokens = []
        generated_input_ids = None
        current_embeds = inputs_embeds
        generation_steps = 0
        
        if logits_processors is None or sampler is None:
            default_processors, _, default_sampler = build_generation_components(
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            logits_processors = logits_processors or default_processors
            sampler = sampler or default_sampler
        
        for i in range(max_new_tokens):
            # Forward pass through code_predictor
            # NOTE: use_cache=False for subtalker since it only generates 15 tokens
            # The KV cache overhead outweighs the benefit for such short sequences
            output = self(
                inputs_embeds=current_embeds,
                use_cache=False,  # Disabled for subtalker - not worth it for 15 tokens
                output_hidden_states=False,
                generation_steps=generation_steps,
            )
            
            # Get logits for last position
            last_logits = output.logits[:, -1, :].clone()
            
            # Apply logits processors
            last_logits = apply_logits_processors(logits_processors, generated_input_ids, last_logits)
            
            # Sample
            next_token = sampler(last_logits)
            
            generated_tokens.append(next_token)
            
            if generated_input_ids is None:
                generated_input_ids = next_token
            else:
                generated_input_ids = torch.cat([generated_input_ids, next_token], dim=1)
            
            # Prepare embedding for next step
            next_embed = self.get_input_embeddings()[i](next_token)
            current_embeds = torch.cat([current_embeds, next_embed], dim=1)
            
            # Update generation_steps
            generation_steps = output.extra.get("generation_steps", generation_steps + 1)
        
        return torch.cat(generated_tokens, dim=-1)
