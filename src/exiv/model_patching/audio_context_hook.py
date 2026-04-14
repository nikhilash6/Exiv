import torch
import torch.nn as nn
from functools import partial
import math

from typing import Optional, Dict, Any, Callable, List

from .hook_registry import FeatureType, HookLocation, HookType, ModelHook, HookRegistry, register_hook_method
from ..utils.logging import app_logger


class AudioContextHook(ModelHook):
    
    DEFAULT_MAX_CHUNK_SIZE = 100
    
    def __init__(self, max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE):
        super().__init__()
        self.hook_type = HookType.AUDIO_KV_MIXING.value
        self.hook_location = HookLocation.AR_GENERATE.value
        self.max_chunk_size = max_chunk_size
        
    def _uniform_chunk_input_ids(self, input_ids: torch.Tensor, max_chunk_size: int) -> List[torch.Tensor]:
        """
        uniformly split input_ids tensor into chunks along the sequence dimension.
        
        args:
            input_ids: input token ids tensor [batch, seq_len]
            max_chunk_size: maximum size per chunk
            
        returns:
            list of input_ids tensors
        """
        batch_size, seq_len = input_ids.shape
        
        if seq_len <= max_chunk_size:
            return [input_ids]
        
        # calculate number of chunks needed
        num_chunks = math.ceil(seq_len / max_chunk_size)
        # try to make chunks roughly equal size
        base_chunk_size = seq_len // num_chunks
        remainder = seq_len % num_chunks
        
        chunks = []
        start_idx = 0
        
        for i in range(num_chunks):
            # distribute remainder across first chunks
            chunk_size = base_chunk_size + (1 if i < remainder else 0)
            end_idx = start_idx + chunk_size
            chunks.append(input_ids[:, start_idx:end_idx])
            start_idx = end_idx
            
        return chunks
        
    def execute(self, model: 'ARModelMixin', original_fn: Callable, **kwargs) -> torch.LongTensor:
        """
        handle audio generation with optional chunking support.
        
        behavior based on chunking_enabled and input_ids:
        - chunking_enabled=true + list of input_ids -> process each chunk, concatenate
        - chunking_enabled=true + single input_ids -> auto-chunk with warning, process, concatenate
        - chunking_enabled=false + list of input_ids -> merge input_ids, process as single
        - chunking_enabled=false + single input_ids -> pass through (default behavior)
        """
        input_ids = kwargs.get('input_ids')
        chunking_enabled = getattr(model, '_chunking_enabled', True)
        
        # case 1: chunking_enabled = true
        if chunking_enabled:
            # case 1a: already chunked (list of tensors)
            if isinstance(input_ids, list):
                if len(input_ids) == 1:
                    # single chunk, wrap in list as _generate expects list[torch.Tensor]
                    kwargs['input_ids'] = input_ids
                    return original_fn(**kwargs)
                
                app_logger.info(f"AudioContextHook: Processing {len(input_ids)} independent chunks")
                all_outputs = []
                
                for i, chunk_ids in enumerate(input_ids):
                    app_logger.info(f"AudioContextHook: Chunk {i+1}/{len(input_ids)} ({chunk_ids.shape[1]} tokens)")
                    
                    chunk_kwargs = dict(kwargs)
                    chunk_kwargs['input_ids'] = chunk_ids
                    
                    # generate independently
                    chunk_output = self._generate_chunk(model, original_fn, chunk_kwargs)
                    all_outputs.append(chunk_output[0][0])  # extract tensor from tuple (and from list)
                
                # concatenate outputs along sequence dimension (dim=0 for talker codes)
                final_output = torch.cat(all_outputs, dim=0)
                app_logger.info(f"AudioContextHook: Final output {final_output.shape[0]} tokens")
                
                # return as tuple (talker_codes_list, _) matching model.generate() format
                return ([final_output], None)
            
            # case 1b: not chunked (single tensor) - auto-chunk with warning
            elif torch.is_tensor(input_ids):
                if input_ids.shape[1] > self.max_chunk_size:
                    app_logger.warning(
                        f"AudioContextHook: Auto-chunking single tensor of size {input_ids.shape[1]} "
                        f"into chunks of max size {self.max_chunk_size}. "
                        f"This will NOT be exactly sentence-wise split and can lead to unexpected outputs. "
                        f"Consider pre-chunking your input using sentence boundaries."
                    )
                    
                    chunks = self._uniform_chunk_input_ids(input_ids, self.max_chunk_size)
                    app_logger.info(f"AudioContextHook: Auto-created {len(chunks)} chunks")
                    
                    all_outputs = []
                    for i, chunk_ids in enumerate(chunks):
                        app_logger.info(f"AudioContextHook: Auto-chunk {i+1}/{len(chunks)} ({chunk_ids.shape[1]} tokens)")
                        
                        chunk_kwargs = dict(kwargs)
                        chunk_kwargs['input_ids'] = chunk_ids
                        
                        chunk_output = self._generate_chunk(model, original_fn, chunk_kwargs)
                        all_outputs.append(chunk_output[0][0])  # extract tensor from tuple (and from list)
                    
                    # concatenate outputs along sequence dimension (dim=0 for talker codes)
                    final_output = torch.cat(all_outputs, dim=0)
                    app_logger.info(f"AudioContextHook: Final output {final_output.shape[0]} tokens")
                    
                    # return as tuple (talker_codes_list, _) matching model.generate() format
                    return ([final_output], None)
                else:
                    # single tensor smaller than max_chunk_size, pass through
                    return original_fn(**kwargs)
            else:
                # no input_ids provided, pass through
                return original_fn(**kwargs)
        
        # case 2: chunking_enabled = false
        else:
            # if received chunked input_ids, merge them first
            if isinstance(input_ids, list):
                if len(input_ids) == 1:
                    # keep as list, _generate expects list[torch.Tensor]
                    kwargs['input_ids'] = input_ids
                else:
                    # merge along sequence dimension (dim=1)
                    merged_ids = torch.cat(input_ids, dim=1)
                    app_logger.info(f"AudioContextHook: Merged {len(input_ids)} chunks into single tensor of size {merged_ids.shape[1]}")
                    kwargs['input_ids'] = merged_ids
                
                # clear position-related state for merged input
                if 'position_ids' in kwargs:
                    del kwargs['position_ids']
                if 'cache_position' in kwargs:
                    kwargs['cache_position'] = None
                if 'past_key_values' in kwargs:
                    kwargs['past_key_values'] = None
                if hasattr(model, 'rope_deltas'):
                    model.rope_deltas = None
                
                return original_fn(**kwargs)
            else:
                # already single tensor or none, pass through
                return original_fn(**kwargs)
    
    def _generate_chunk(self, model: nn.Module, generate_fn: Callable, chunk_kwargs: Dict) -> torch.LongTensor:
        """generate a single chunk with fresh state."""
        # ensure input_ids is 2D [batch, seq_len]
        if chunk_kwargs['input_ids'].dim() == 1:
            chunk_kwargs['input_ids'] = chunk_kwargs['input_ids'].unsqueeze(0)
        
        # wrap in list as model expects iterable of sequences
        chunk_kwargs['input_ids'] = [chunk_kwargs['input_ids']]
        
        batch_size = chunk_kwargs['input_ids'][0].shape[0]
        chunk_len = chunk_kwargs['input_ids'][0].shape[1]
        device = chunk_kwargs['input_ids'][0].device
        
        chunk_kwargs['attention_mask'] = torch.ones((batch_size, chunk_len), dtype=torch.int64, device=device)
        
        # clear position-related state
        if 'position_ids' in chunk_kwargs:
            del chunk_kwargs['position_ids']
        if 'cache_position' in chunk_kwargs:
            chunk_kwargs['cache_position'] = None
        if 'past_key_values' in chunk_kwargs:
            chunk_kwargs['past_key_values'] = None
        if hasattr(model, 'rope_deltas'):
            model.rope_deltas = None
        
        # handle partial functions
        if isinstance(generate_fn, partial):
            new_keywords = dict(generate_fn.keywords) if generate_fn.keywords else {}
            new_keywords.pop('inputs_embeds', None)
            new_keywords.pop('attention_mask', None)
            new_keywords.pop('input_ids', None)
            new_partial = partial(generate_fn.func, *generate_fn.args, **new_keywords)
            return new_partial(**chunk_kwargs)
        
        return generate_fn(**chunk_kwargs)


def remove_audio_context(model: 'ARModelMixin'):
    HookRegistry.remove_hook_from_module(model, HookType.AUDIO_KV_MIXING.value, method_name='generate')
    

@register_hook_method(FeatureType.AUDIO_CONTEXT_BLENDING.value)
def enable_audio_context(model: 'ARModelMixin', max_chunk_size: int = AudioContextHook.DEFAULT_MAX_CHUNK_SIZE) -> AudioContextHook:
    hook = AudioContextHook(max_chunk_size=max_chunk_size)
    registry = HookRegistry.get_hook_registry(model)
    registry.register_hook(hook)
    return hook
