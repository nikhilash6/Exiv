# coding=utf-8
import torch
from torch import nn
from typing import Optional, Any, Union, Dict
from dataclasses import dataclass, fields
from ....utils.logging import app_logger

class ModelOutput:
    """
    Base class for all model outputs.
    """
    def __getitem__(self, k):
        if isinstance(k, str):
            inner_dict = self.__dict__
            return inner_dict[k]
        else:
            return self.to_tuple()[k]

    def __setitem__(self, k, v):
        setattr(self, k, v)

    def get(self, k, default=None):
        return getattr(self, k, default)

    def __iter__(self):
        for field in fields(self):
            yield getattr(self, field.name)

    def __len__(self):
        return len(fields(self))

    def to_tuple(self):
        return tuple(getattr(self, field.name) for field in fields(self))


@dataclass
class BaseModelOutputWithPast(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Any] = None
    hidden_states: Optional[Any] = None
    attentions: Optional[Any] = None


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
