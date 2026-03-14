# coding=utf-8
import torch
from torch import nn
from typing import Optional, Any, Union, Dict
from dataclasses import dataclass, fields
from ......utils.logging import app_logger

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


class PreTrainedModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    @classmethod
    def _from_config(cls, config, **kwargs):
        return cls(config, **kwargs)

    def post_init(self):
        # Placeholder for weight initialization
        pass

    @property
    def device(self):
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    @property
    def dtype(self):
        try:
            return next(self.parameters()).dtype
        except StopIteration:
            return torch.float32

class MimiModel(PreTrainedModel):
    """
    A lightweight wrapper for MimiModel that attempts to load from transformers 
    but provides a placeholder for architecture if not found.
    """
    def __init__(self, config):
        super().__init__(config)
        try:
            from transformers import MimiModel as HFMimiModel
            self.model = HFMimiModel(config)
        except ImportError:
            app_logger.warning("transformers.MimiModel not found. Encoder may not work.")
            self.model = None

    def encode(self, *args, **kwargs):
        if self.model is not None:
            return self.model.encode(*args, **kwargs)
        raise NotImplementedError("MimiModel backend is missing.")

    def decode(self, *args, **kwargs):
        if self.model is not None:
            return self.model.decode(*args, **kwargs)
        raise NotImplementedError("MimiModel backend is missing.")

    def forward(self, *args, **kwargs):
        if self.model is not None:
            return self.model.forward(*args, **kwargs)
        raise NotImplementedError("MimiModel backend is missing.")

class MimiFeatureExtractor:
    def __init__(self, sampling_rate=24000, **kwargs):
        self.sampling_rate = sampling_rate
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __call__(self, raw_audio, sampling_rate=None, return_tensors=None, **kwargs):
        if sampling_rate is not None and sampling_rate != self.sampling_rate:
             # In a real scenario we might resample, but here we expect normalization to happen before
             pass
        
        if not isinstance(raw_audio, list):
            raw_audio = [raw_audio]
            
        # Standardize to tensors
        tensors = [torch.from_numpy(a) if isinstance(a, np.ndarray) else a for a in raw_audio]
        
        # Simple padding
        max_len = max(t.shape[-1] for t in tensors)
        padded_tensors = []
        padding_masks = []
        
        for t in tensors:
            pad_len = max_len - t.shape[-1]
            if pad_len > 0:
                padded_tensors.append(torch.nn.functional.pad(t, (0, pad_len)))
                mask = torch.cat([torch.ones(t.shape[-1]), torch.zeros(pad_len)])
                padding_masks.append(mask)
            else:
                padded_tensors.append(t)
                padding_masks.append(torch.ones(t.shape[-1]))
                
        batch_values = torch.stack(padded_tensors).unsqueeze(1) # [B, 1, T]
        batch_masks = torch.stack(padding_masks).unsqueeze(1)  # [B, 1, T]
        
        return {
            "input_values": batch_values,
            "padding_mask": batch_masks
        }

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        # In exiv, we can just return a default instance or load from a json if it exists
        return cls(**kwargs)

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
