import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from typing import Optional, Tuple

from ..utils.logging import app_logger
from ..utils.device import XFORMERS_AVAILABLE, SDPA_AVAILABLE

def create_attention_mask(
      query_len: int,
      kv_len: int,
      device: torch.device,
      sliding_window: int | None = None,
      dtype: torch.dtype = None,
  ) -> torch.Tensor:
    """
    creates a 4D causal attention mask of shape (1, 1, query_len, kv_len)
    returns 0 for allowed positions, -inf for masked positions
    
    NOTE: When query_len == 1, this is a generation step with KV cache.
    In this case, the query should attend to ALL positions in the KV cache,
    not just the first position. The causal mask logic (kv_idx <= q_idx) 
    would only allow attending to position 0 when query_len=1, which is wrong.
    """
    mask_dtype = dtype if dtype is not None else torch.float32
    
    # When query_len == 1, we're generating one token with KV cache.
    # The query should attend to all cached positions (0 to kv_len-1).
    if query_len == 1:
        if sliding_window is not None and kv_len > sliding_window:
            # Sliding window: only attend to last `sliding_window` positions
            mask = torch.zeros(1, 1, query_len, kv_len, device=device, dtype=mask_dtype)
            mask[:, :, :, :kv_len - sliding_window] = float('-inf')
            return mask
        else:
            # Full attention to all cached positions
            return torch.zeros(1, 1, query_len, kv_len, device=device, dtype=mask_dtype)
    
    # For prefill (query_len > 1), use standard causal masking
    q_idx = torch.arange(query_len, device=device).unsqueeze(1)
    kv_idx = torch.arange(kv_len, device=device).unsqueeze(0)

    # causal: can only attend to previous positions
    mask = kv_idx <= q_idx
    if sliding_window is not None:
        mask = mask & (kv_idx > q_idx - sliding_window)

    return torch.where(mask, torch.tensor(0.0, dtype=mask_dtype, device=device), 
                       torch.tensor(float('-inf'), dtype=mask_dtype, device=device)).unsqueeze(0).unsqueeze(0)

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def standard_attention(q, k, v, attn_mask=None):
    # q, k, v: (batch, heads, seq, dim_head)
    b, h, seq_len, dim_head = q.shape

    # attn score matrix = Q * Kt / sqrt(d)
    # scores: (bs, head, seq_len, seq_len)
    scale = dim_head ** -0.5
    scores = torch.einsum("b h i d, b h j d -> b h i j", q, k) * scale

    # for stuff like causal attn / padding ignore
    if attn_mask is not None:
        # heuristic: If the mask is 2D and square, assume it's a causal mask.
        # casual_mask: (seq_len, seq_len)
        # NOTE: general causal mask can be (seq_len, key_len)
        if attn_mask.ndim == 2 and attn_mask.shape[0] == attn_mask.shape[1]:
            # adding batch and head dim, same mask is applied to every head/batch
            # reshaping (seq_len, seq_len) -> (1, 1, seq_len, seq_len)
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            scores = scores.masked_fill(attn_mask.bool() == True, -torch.inf)
        else:
            # padding_mask: (bs, seq_len), varies across batches and key positions
            # Query ↓   Key →
            #         t1  t2  t3  t4  t5
            #         --------------------
            # t1 |    ✓   ✓   ✓   ✗   ✗
            # t2 |    ✓   ✓   ✓   ✗   ✗
            # t3 |    ✓   ✓   ✓   ✗   ✗
            # t4 |    ✓   ✓   ✓   ✗   ✗
            # t5 |    ✓   ✓   ✓   ✗   ✗
            # reshaping (bs, seq_len) -> (bs, 1, 1, seq_len)
            attn_mask = attn_mask.view(b, 1, 1, -1)
            scores = scores.masked_fill(attn_mask.bool() == False, -torch.inf)

    # softmax
    attn = scores.softmax(dim=-1)
    # out: (bs, h, seq_len, dim_head)
    out = torch.einsum("b h i j, b h j d -> b h i d", attn, v)
    
    return out

def available_attn():
    '''
    Auto select the best available attn available
    
    In the first call 'impl' will be set and it will be reused in all subsequent calls
    '''
    if not hasattr(available_attn, "impl"):
        if SDPA_AVAILABLE:
            app_logger.info(" > PyTorch 2.0 SDPA is available. Using it.")
            available_attn.impl = F.scaled_dot_product_attention
        
        elif XFORMERS_AVAILABLE:
            import xformers.ops
            app_logger.debug(" > xFormers is available. Using it.")
            available_attn.impl = xformers.ops.memory_efficient_attention
        
        else:
            app_logger.warning(" > No accelerated attention backend found. Using standard attention.")
            available_attn.impl = standard_attention
    
    return available_attn.impl

def vae_optimized_attention(q: Tensor, k: Tensor, v: Tensor):
    attn_fn = available_attn()
    return attn_fn(q, k, v)

def optimized_attention(q: Tensor, k: Tensor, v: Tensor, heads: int, mask: Tensor = None):
    attn_fn = available_attn()

    assert all(len(x.shape) == 3 for x in [q, k, v]), "shape mismatch, requires 3D tensors (bs, seq_len, dim)"
    
    b, target_len, dim = q.shape
    assert dim % heads == 0, "dim mismatch in attn, not divisible by heads"
    dim_head = dim // heads

    # q, k, v -> (bs, h, -1, dim_head)
    q, k, v = map(lambda t: t.view(b, -1, heads, dim_head).transpose(1, 2), (q, k, v))

    # SDPA and our standard fallback expect (bs, h, seq_len, dim_head)
    if attn_fn is F.scaled_dot_product_attention or attn_fn is standard_attention:
        # SDPA expects a boolean mask
        if mask is not None:
            if mask.ndim == 4:
                # T5 style bias mask
                pass
            elif mask.ndim == 2:
                # normal padding mask
                mask = mask.view(b, 1, 1, -1).expand(-1, heads, q.size(2), -1).bool()
            else:
                raise ValueError(f"unsupported mask shape: {mask.shape}")
        
        out = attn_fn(q, k, v, attn_mask=mask)
        return out.transpose(1, 2).reshape(b, -1, dim)  # (bs, seq_len, dim)

    # xFormers expects (batch, seq, heads, dim_head)
    else:
        q, k, v = map(lambda t: t.transpose(1, 2), (q, k, v),)

        # xFormers uses an attention bias instead of a boolean mask.
        attn_bias = None
        if mask is not None:
            if mask.ndim == 4:
                # 4D T5-style bias [1, 64, 512, 512]
                attn_bias = mask
            elif mask.ndim == 2:
                # 2D padding mask. Create 4D bias.
                attn_bias = torch.zeros((b, heads, q.size(1), k.size(1)), dtype=q.dtype, device=q.device)
                attn_bias.masked_fill_(mask.view(b, 1, 1, -1).bool() == False, -torch.inf)
            else:
                raise ValueError(f"Unsupported mask shape: {mask.shape}")

        out = attn_fn(q, k, v, attn_bias=attn_bias)
        return out.view(b, -1, dim)

# TODO: streamline later
def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    heads: int,
    attention_mask: Optional[torch.Tensor],
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    
    # query, key_states, value_states are 4D: (batch, heads, seq_len, head_dim)
    # optimized_attention expects 3D: (batch, seq_len, dim)
    # Note: key/value may have different seq_len than query when using KV cache
    b, h, s_q, d = query.shape
    _, _, s_k, _ = key_states.shape  # key sequence length may be longer due to cache
    _, _, s_v, _ = value_states.shape  # value should match key
    
    query_3d = query.transpose(1, 2).reshape(b, s_q, h * d)
    key_3d = key_states.transpose(1, 2).reshape(b, s_k, h * d)
    value_3d = value_states.transpose(1, 2).reshape(b, s_v, h * d)
    
    # TODO: check if -inf are handled properly
    return optimized_attention(query_3d, key_3d, value_3d, heads, attention_mask)
