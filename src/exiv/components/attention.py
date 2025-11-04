import torch
import torch.nn.functional as F

from ..utils.logging import app_logger
from ..utils.device import XFORMERS_AVAILABLE, SDPA_AVAILABLE

def standard_attention(q, k, v, heads, mask=None):
    # q, k, v: (batch, heads, seq, dim_head) , simplified dims for understanding
    b, h, seq_len, dim_head = q.shape

    # attn score matrix = Q * Kt / sqrt(d)
    # scores: (bs, head, seq_len, seq_len)
    scale = dim_head ** -0.5
    scores = torch.einsum("b h i d, b h j d -> b h i j", q, k) * scale

    # for stuff like causal attn / padding ignore
    if mask is not None:
        # heuristic: If the mask is 2D and square, assume it's a causal mask.
        # casual_mask: (seq_len, seq_len)
        # NOTE: general causal mask can be (seq_len, key_len)
        if mask.ndim == 2 and mask.shape[0] == mask.shape[1]:
            # adding batch and head dim, same mask is applied to every head/batch
            # reshaping (seq_len, seq_len) -> (1, 1, seq_len, seq_len)
            mask = mask.unsqueeze(0).unsqueeze(0)
            scores = scores.masked_fill(mask.bool() == True, -torch.inf)
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
            mask = mask.view(b, 1, 1, -1)
            scores = scores.masked_fill(mask.bool() == False, -torch.inf)


    # softmax
    attn = scores.softmax(dim=-1)
    # out: (bs, h, seq_len, dim_head)
    out = torch.einsum("b h i j, b h j d -> b h i d", attn, v)
    
    return out


def optimized_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int, mask: torch.Tensor = None):
    '''
    Auto select the best available attn available
    
    In the first call 'impl' will be set and it will be reused in all subsequent calls
    '''
    if not hasattr(optimized_attention, "impl"):
        if SDPA_AVAILABLE:
            app_logger.info(" > PyTorch 2.0 SDPA is available. Using it.")
            optimized_attention.impl = F.scaled_dot_product_attention
        
        elif XFORMERS_AVAILABLE:
            import xformers.ops
            app_logger.debug(" > xFormers is available. Using it.")
            optimized_attention.impl = xformers.ops.memory_efficient_attention
        
        else:
            app_logger.warning(" > No accelerated attention backend found. Using standard attention.")
            optimized_attention.impl = standard_attention

    assert all(len(x.shape) == 3 for x in [q, k, v]), "shape mismatch, requires 3D tensors (bs, seq_len, dim)"
    
    b, seq_len, dim = q.shape
    assert dim % heads == 0, "dim mismatch in attn, not divisible by heads"
    dim_head = dim // heads

    # q, k, v -> (bs, h, seq_len, dim_head)
    q, k, v = map(lambda t: t.view(b, seq_len, heads, dim_head).transpose(1, 2), (q, k, v))

    # SDPA and our standard fallback expect (bs, h, seq_len, dim_head)
    if optimized_attention.impl is F.scaled_dot_product_attention or optimized_attention.impl is standard_attention:
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
        
        out = optimized_attention.impl(q, k, v, attn_mask=mask)
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

        out = optimized_attention.impl(q, k, v, attn_bias=attn_bias)
        return out.view(b, -1, dim)