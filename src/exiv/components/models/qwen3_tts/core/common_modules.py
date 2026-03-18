import torch
import torch.nn as nn

from .config import Qwen3TTSTalkerConfig, Qwen3TTSConfig
from ....activations import ACT2FN


from .....model_utils.meta_utils import materialize_meta_buffers

def _compute_rope_inv_freq(config, device=None):
    """
    Compute inverse frequencies for RoPE (Rotary Position Embedding).
    This is the default/original RoPE implementation.
    """
    base = getattr(config, "rope_theta", 10000.0)
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    
    # Compute inverse frequencies: 1 / (base^(i/dim)) for i in [0, 2, 4, ..., dim-2]
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.int64, device=device).float() / head_dim))
    
    return inv_freq, 1.0  # (inv_freq, attention_scaling)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

# main talker uses 3D rope while the code predictor uses 1D rope
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class Qwen3TTSTalkerRotaryEmbedding(nn.Module):
    def __init__(self, config: Qwen3TTSTalkerConfig, device=None):
        super().__init__()
        self.config = config
        inv_freq, self.attention_scaling = _compute_rope_inv_freq(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    # TODO: add back this decorator
    # @materialize_meta_buffers(inv_freq=lambda self, device: _compute_rope_inv_freq(self.config, device)[0])
    def forward(self, x, position_ids):
        # TODO: fix this
        # lazy initialization: if inv_freq is zeros, recompute it
        if self.inv_freq.abs().max() == 0:
            device = self.inv_freq.device
            self.inv_freq = _compute_rope_inv_freq(self.config, device)[0].to(device)
        
        # In contrast to other models, Qwen3TTSThinkerText has different position ids for the grids
        # So we expand the inv_freq to shape (3, ...)
        inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
        position_ids_expanded = position_ids[:, :, None, :].float()  # shape (3, bs, 1, positions)

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

# code predictor
class Qwen3TTSRotaryEmbedding(nn.Module):
    def __init__(self, config: Qwen3TTSConfig, device=None):
        super().__init__()
        self.config = config
        inv_freq, self.attention_scaling = _compute_rope_inv_freq(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    # @materialize_meta_buffers(inv_freq=lambda self, device: _compute_rope_inv_freq(self.config, device)[0])
    def forward(self, x, position_ids):
        # TODO: fix this
        # lazy initialization: if inv_freq is zeros, recompute it
        if self.inv_freq.abs().max() == 0:
            device = self.inv_freq.device
            self.inv_freq = _compute_rope_inv_freq(self.config, device)[0].to(device)
        
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

class Qwen3TTSRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Qwen3TTSRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

class Qwen3TTSTalkerResizeMLP(nn.Module):
    def __init__(self, input_size: int, intermediate_size: int, output_size: int, act: str, bias=False):
        super().__init__()
        self.linear_fc1 = nn.Linear(input_size, intermediate_size, bias=bias)
        self.linear_fc2 = nn.Linear(intermediate_size, output_size, bias=bias)
        self.act_fn = ACT2FN[act]

    def forward(self, hidden_state):
        return self.linear_fc2(self.act_fn(self.linear_fc1(hidden_state)))

class Qwen3TTSTalkerTextMLP(nn.Module):
    def __init__(self, config, intermediate_size=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size if intermediate_size is not None else config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj
