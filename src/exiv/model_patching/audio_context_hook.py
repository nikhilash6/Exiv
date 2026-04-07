import torch
import torch.nn as nn

import math
from typing import Optional, Dict, Any, Callable, Tuple

from .hook_registry import FeatureType, HookLocation, HookType, ModelHook, HookRegistry, register_hook_method
from ..utils.logging import app_logger
from ..utils.text_chunking import chunk_text_by_sentences


# audio tokens per text token (empirical estimate for TTS)
# ~12.5 audio tok/s at ~2.5 words/s = ~5 audio tokens per word
# with BPE ~1.3 tokens per word, so ~4 audio tokens per text token
AUDIO_TOKENS_PER_TEXT_TOKEN = 4.0
MAX_AUDIO_TOKENS_MARGIN = 1.5


class AudioContextHook(ModelHook):
    
    DEFAULT_OVERLAP_LENGTH = 50     # number of audio tokens overlapping
    DEFAULT_MAX_CHUNK_SIZE = 700    # max text tokens per chunk
    
    def __init__(
        self,
        max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
        prefix_length: int = DEFAULT_OVERLAP_LENGTH
    ):
        super().__init__()
        self.hook_type = HookType.AUDIO_KV_MIXING.value
        self.hook_location = HookLocation.AR_GENERATE.value
        self.max_chunk_size = max_chunk_size
        self.prefix_length = prefix_length
        
        # Store prefix audio token IDs (not embeddings) for next chunk
        self._prefix_audio_ids: Optional[torch.LongTensor] = None
        self._all_outputs: list[torch.LongTensor] = []
        
        # Track cumulative position offset for continuity
        self._position_offset: int = 0
        
    def execute(
        self,
        model: 'ARModelMixin',
        original_fn: Callable,      # this is the 'generate' method
        **kwargs
    ) -> torch.LongTensor:
        inputs_embeds = kwargs.get('inputs_embeds', None)
        if inputs_embeds is None:
            app_logger.warning("AudioContextHook: No inputs_embeds found, using single-chunk generation")
            return original_fn(**kwargs)
        
        seq_len = inputs_embeds.shape[1]
        
        if seq_len <= self.max_chunk_size:
            app_logger.debug(f"AudioContextHook: Input length {seq_len} <= {self.max_chunk_size}, no chunking needed")
            return original_fn(**kwargs)
        
        # Get text for sentence-aware chunking if available
        text = kwargs.get('text', None)
        
        # Split into chunks with their estimated token counts
        chunks_with_tokens = self._split_into_chunks(inputs_embeds, text)
        app_logger.info(f"AudioContextHook: Splitting input of {seq_len} tokens into {len(chunks_with_tokens)} chunks")
        
        self._prefix_audio_ids = None
        self._all_outputs = []
        self._position_offset = 0
        
        for i, (chunk_embeds, chunk_text_tokens) in enumerate(chunks_with_tokens):
            is_first = (i == 0)
            is_last = (i == len(chunks_with_tokens) - 1)
            
            # Calculate max_new_tokens for this chunk based on text length
            # This ensures EOS works correctly - model stops when text is fully spoken
            estimated_audio_tokens = int(chunk_text_tokens * AUDIO_TOKENS_PER_TEXT_TOKEN * MAX_AUDIO_TOKENS_MARGIN)
            # Add prefix length since we're prepending audio context
            if not is_first:
                estimated_audio_tokens += self.prefix_length
            
            app_logger.info(f"AudioContextHook: Generating chunk {i+1}/{len(chunks_with_tokens)} "
                           f"({chunk_embeds.shape[1]} text tokens, ~{estimated_audio_tokens} max audio tokens)")
            
            chunk_output = self._generate_chunk(
                model, original_fn, chunk_embeds, kwargs, is_first, is_last, 
                max_new_tokens=estimated_audio_tokens
            )
            
            # Store output (excluding prefix audio which is from previous chunk)
            new_audio = self._extract_new_audio(chunk_output, is_first)
            self._all_outputs.append(new_audio)
            
            if not is_last:
                self._prefix_audio_ids = chunk_output[:, -self.prefix_length:].detach()
                self._position_offset += new_audio.shape[1]
                app_logger.debug(f"AudioContextHook: Extracted {self.prefix_length} prefix tokens for next chunk, position_offset={self._position_offset}")
        
        # Concatenate all outputs
        final_output = torch.cat(self._all_outputs, dim=1)
        app_logger.info(f"AudioContextHook: Generated {final_output.shape[1]} total tokens across {len(chunks_with_tokens)} chunks")
        
        # Convert from [batch, seq_len, num_code_groups] to list of [seq_len, num_code_groups]
        # This matches the expected format of the pipeline's tokenizer_decode
        talker_codes_list = [final_output[i] for i in range(final_output.shape[0])]
        
        return talker_codes_list
    
    def _split_into_chunks(self, input_tensor: torch.Tensor) -> list[torch.Tensor]:
        """Split input into roughly equal chunks."""
        seq_len = input_tensor.shape[1]
        
        # Calculate number of chunks
        num_chunks = max(2, math.ceil(seq_len / self.max_chunk_size))
        chunk_size = seq_len // num_chunks
        
        chunks = []
        for i in range(num_chunks):
            start_idx = i * chunk_size
            if i == num_chunks - 1:
                end_idx = seq_len
            else:
                end_idx = (i + 1) * chunk_size
            
            chunk = input_tensor[:, start_idx:end_idx]
            chunks.append(chunk)
        
        return chunks
    
    def _generate_chunk(
        self,
        model: nn.Module,
        generate_fn: Callable,
        chunk_embeds: torch.Tensor,
        kwargs: Dict,
        is_first: bool,
        is_last: bool
    ) -> torch.LongTensor:
        """Generate audio for a single text chunk with proper MROPE handling."""
        chunk_kwargs = dict(kwargs)
        
        batch_size = chunk_embeds.shape[0]
        chunk_len = chunk_embeds.shape[1]
        device = chunk_embeds.device
        
        # If we have prefix audio from previous chunk, we need to:
        # 1. Get embeddings for prefix audio tokens
        # 2. Concatenate with current chunk embeddings
        # 3. Create proper attention mask and position_ids for MROPE
        if self._prefix_audio_ids is not None:
            prefix_len = self._prefix_audio_ids.shape[1]
            total_len = prefix_len + chunk_len
            
            # Get embeddings for prefix audio
            # Try different possible locations for the embedding layer
            embed_layer = None
            if hasattr(model, 'embed_tokens'):
                embed_layer = model.embed_tokens
            elif hasattr(model, 'get_input_embeddings'):
                embed_layer = model.get_input_embeddings()
            elif hasattr(model, 'model'):
                if hasattr(model.model, 'embed_tokens'):
                    embed_layer = model.model.embed_tokens
                elif hasattr(model.model, 'get_input_embeddings'):
                    embed_layer = model.model.get_input_embeddings()
            
            if embed_layer is None:
                raise RuntimeError("Could not find embedding layer in model")
            
            with torch.no_grad():
                # Audio tokens may have shape [batch, seq_len, num_code_groups]
                # We only use the first codebook for the prefix continuation
                if self._prefix_audio_ids.dim() == 3:
                    # [batch, seq_len, num_code_groups] -> use first codebook
                    prefix_ids_2d = self._prefix_audio_ids[:, :, 0].to(device)
                else:
                    # [batch, seq_len] 
                    prefix_ids_2d = self._prefix_audio_ids.to(device)
                
                prefix_embeds = embed_layer(prefix_ids_2d)
            
            # Concatenate prefix + current chunk
            combined_embeds = torch.cat([prefix_embeds, chunk_embeds], dim=1)
            chunk_kwargs['inputs_embeds'] = combined_embeds
            
            # Create attention mask: all 1s since all tokens are valid
            chunk_kwargs['attention_mask'] = torch.ones(
                (batch_size, total_len),
                dtype=torch.int64,
                device=device
            )
            
            # For MROPE, position_ids should be [3, batch, seq_len]
            # We need continuous positions across chunks
            # Create position_ids that continue from previous chunk
            base_positions = torch.arange(
                self._position_offset, 
                self._position_offset + total_len,
                device=device
            )
            # Expand to [3, batch, seq_len] for MROPE
            position_ids = base_positions.unsqueeze(0).unsqueeze(0).expand(3, batch_size, -1)
            chunk_kwargs['position_ids'] = position_ids
            
            # Clear cache_position to use our position_ids
            if 'cache_position' in chunk_kwargs:
                chunk_kwargs['cache_position'] = None
            
            app_logger.debug(f"Chunk with prefix: prefix_len={prefix_len}, chunk_len={chunk_len}, total_len={total_len}, position_offset={self._position_offset}")
        else:
            chunk_kwargs['inputs_embeds'] = chunk_embeds
            chunk_kwargs['attention_mask'] = torch.ones(
                (batch_size, chunk_len),
                dtype=torch.int64,
                device=device
            )
            
            # let model compute position_ids for first chunk
            if 'position_ids' in chunk_kwargs:
                del chunk_kwargs['position_ids']
            if 'cache_position' in chunk_kwargs:
                chunk_kwargs['cache_position'] = None
        
        # Reset rope_deltas on model to ensure fresh computation
        if hasattr(model, 'rope_deltas'):
            model.rope_deltas = None
        
        # Clear any past_key_values to start fresh for each chunk
        if 'past_key_values' in chunk_kwargs:
            chunk_kwargs['past_key_values'] = None
        
        # Call the model's _generate method
        output = generate_fn(**chunk_kwargs)
        
        return output
    
    def _extract_new_audio(
        self,
        chunk_output: torch.LongTensor,
        is_first: bool
    ) -> torch.LongTensor:
        """Extract newly generated audio, excluding prefix from previous chunk."""
        if is_first or self._prefix_audio_ids is None: return chunk_output
        else:
            prefix_len = self._prefix_audio_ids.shape[1]
            return chunk_output[:, prefix_len:]


def remove_audio_context(model: 'ARModelMixin'):
    HookRegistry.remove_hook_from_module(model, HookType.AUDIO_KV_MIXING.value, method_name='generate')
    

@register_hook_method(FeatureType.AUDIO_CONTEXT_BLENDING.value)
def enable_audio_context(
    model: 'ARModelMixin',
    max_chunk_size: int = AudioContextHook.DEFAULT_MAX_CHUNK_SIZE,
    prefix_length: int = AudioContextHook.DEFAULT_OVERLAP_LENGTH
) -> AudioContextHook:
    hook = AudioContextHook(max_chunk_size=max_chunk_size, prefix_length=prefix_length)
    # Apply hook to 'generate' method to intercept full generation (not individual forward steps)
    HookRegistry.apply_hook_to_module(model, hook, method_name='generate')
    return hook
