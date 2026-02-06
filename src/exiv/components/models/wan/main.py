# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.

import torch
from torch import Tensor
import torch.nn as nn
from einops import rearrange

import math
from typing import List

from .cond_preprocessor import is_text_model, is_img_model, Wan21ModelArchConfig, Wan22ModelArchConfig
from ...enum import Model, ModelType
from ...attention import optimized_attention
from ...positional_embeddings import EmbedND, apply_rope
from ...latent_format import LatentFormat, Wan21VAELatentFormat, Wan22VAELatentFormat
from ....components.samplers.sampler_types import get_model_sampling
from ....model_utils.model_mixin import ModelArchConfig, ModelMixin
from ....utils.tensor import common_upscale, pad_to_patch_size
from ....utils.device import VRAM_DEVICE
from ....utils.logging import app_logger

def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float32)

    # calculation
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x

class WanSelfAttention(nn.Module):

    def __init__(
        self,
        dim,
        num_heads,
        window_size=(-1, -1),
        qk_norm=True,
        eps=1e-6,
        kv_dim=None,
    ):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps
        if kv_dim is None:
            kv_dim = dim

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(kv_dim, dim)
        self.v = nn.Linear(kv_dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = nn.RMSNorm(dim, eps=eps, elementwise_affine=True) if qk_norm else nn.Identity()
        self.norm_k = nn.RMSNorm(dim, eps=eps, elementwise_affine=True) if qk_norm else nn.Identity()

    def forward(self, x, freqs):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        def qkv_fn_q(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            return apply_rope(q, freqs)

        def qkv_fn_k(x):
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            return apply_rope(k, freqs)

        q = qkv_fn_q(x)
        k = qkv_fn_k(x)

        x = optimized_attention(
            q.view(b, s, n * d),
            k.view(b, s, n * d),
            self.v(x).view(b, s, n * d),
            heads=self.num_heads,
        )

        x = self.o(x)
        return x


class WanT2VCrossAttention(WanSelfAttention):

    def forward(self, x, context, **kwargs):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            kwargs: extra args that maybe in use in i2v
        """
        # compute query, key, value
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(context))
        v = self.v(context)

        # compute attention
        x = optimized_attention(q, k, v, heads=self.num_heads)

        x = self.o(x)
        return x


class WanI2VCrossAttention(WanSelfAttention):

    def __init__(
        self,
        dim,
        num_heads,
        window_size=(-1, -1),
        qk_norm=True,
        eps=1e-6
    ):
        super().__init__(dim, num_heads, window_size, qk_norm, eps)

        self.k_img = nn.Linear(dim, dim)
        self.v_img = nn.Linear(dim, dim)
        # self.alpha = nn.Parameter(torch.zeros((1, )))
        self.norm_k_img = nn.RMSNorm(dim, eps=eps, elementwise_affine=True) if qk_norm else nn.Identity()

    def forward(self, x, context, context_img_len):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
        """
        context_img = context[:, :context_img_len]
        context = context[:, context_img_len:]

        # compute query, key, value
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(context))
        v = self.v(context)
        k_img = self.norm_k_img(self.k_img(context_img))
        v_img = self.v_img(context_img)
        img_x = optimized_attention(q, k_img, v_img, heads=self.num_heads)
        # compute attention
        x = optimized_attention(q, k, v, heads=self.num_heads)

        # output
        x = x + img_x
        x = self.o(x)
        return x


WAN_CROSSATTENTION_CLASSES = {
    't2v_cross_attn': WanT2VCrossAttention,
    'i2v_cross_attn': WanI2VCrossAttention,
}


def repeat_e(e, x):
    repeats = 1
    if e.size(1) > 1:
        repeats = x.size(1) // e.size(1)
    if repeats == 1:
        return e
    if repeats * e.size(1) == x.size(1):
        return torch.repeat_interleave(e, repeats, dim=1)
    else:
        return torch.repeat_interleave(e, repeats + 1, dim=1)[:, :x.size(1)]


class WanAttentionBlock(nn.Module):

    def __init__(
        self,
        cross_attn_type,
        dim,
        ffn_dim,
        num_heads,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=False,
        eps=1e-6
    ):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = nn.LayerNorm(dim, eps, elementwise_affine=False)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm, eps)
        self.norm3 = nn.LayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim,
                                                                      num_heads,
                                                                      (-1, -1),
                                                                      qk_norm,
                                                                      eps)
        self.norm2 = nn.LayerNorm(dim, eps, elementwise_affine=False)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), 
            nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim)
        )

        # modulation
        self.modulation = nn.Parameter(torch.empty(1, 6, dim))

    def forward(
        self,
        x,
        e,
        freqs,
        context,                # conditioning data (txt/img)
        context_img_len=257,    # splitting point
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, 6, C]
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        
        """
        - e[0] (shift) & e[1] (scale): Modulate the input before self_attn.
        - e[2] (scale): Modulates the output after self_attn.
        - e[3] (shift) & e[4] (scale): Modulate the input before the ffn (Feed-Forward Network).
        - e[5] (scale): Modulates the output after the ffn
        """

        if e.ndim < 4:
            # following the og impl, [B, 6, C], single timestep for the entire video
            e = (self.modulation.to(dtype=x.dtype, device=x.device) + e).chunk(6, dim=1)
        else:
            # custom comfy impl., [B, NumPatches, 6, C], individual timesteps for different patches
            e = (self.modulation.to(dtype=x.dtype, device=x.device).unsqueeze(0) + e).unbind(2)

        # self-attention
        # modulated x = shift + (normalized_x * scale)
        # shift = repeat_e(e[0], x) -> make sure that e[0] has the right dimensions
        # scale = 1 + repeat_e(e[1], x) -> default scale of 1
        y = self.self_attn(
            torch.addcmul(repeat_e(e[0], x), self.norm1(x), 1 + repeat_e(e[1], x)),
            freqs)

        # modulation after self attn
        x = torch.addcmul(x, y, repeat_e(e[2], x))
        del y

        # cross-attention & ffn
        x = x + self.cross_attn(self.norm3(x), context, context_img_len=context_img_len)
        y = self.ffn(torch.addcmul(repeat_e(e[3], x), self.norm2(x), 1 + repeat_e(e[4], x)))
        x = torch.addcmul(x, y, repeat_e(e[5], x))
        return x


class Head(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = nn.LayerNorm(dim, eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.empty(1, 2, dim))

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, C]
        """
        # assert e.dtype == torch.float32
        if e.ndim < 3:
            e = (self.modulation.to(dtype=x.dtype, device=x.device) + e.unsqueeze(1)).chunk(2, dim=1)
        else:
            e = (self.modulation.to(dtype=x.dtype, device=x.device).unsqueeze(0) + e.unsqueeze(2)).unbind(2)

        x = (self.head(torch.addcmul(repeat_e(e[0], x), self.norm(x), 1 + repeat_e(e[1], x))))
        return x


class MLPProj(torch.nn.Module):

    def __init__(self, in_dim, out_dim, flf_pos_embed_token_number=None):
        super().__init__()

        self.proj = torch.nn.Sequential(
            nn.LayerNorm(in_dim), 
            nn.Linear(in_dim, in_dim),
            torch.nn.GELU(), 
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim)
        )

        # emb_pos differentiates b/w the different frames being passed in
        if flf_pos_embed_token_number is not None:
            self.emb_pos = nn.Parameter(torch.empty((1, flf_pos_embed_token_number, in_dim)))
        else:
            self.emb_pos = None

    def forward(self, image_embeds):
        if self.emb_pos is not None:
            image_embeds = image_embeds[:, :self.emb_pos.shape[1]] + self.emb_pos[:, :image_embeds.shape[1]].to(dtype=image_embeds.dtype, device=image_embeds.device)

        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens

class Wan21Model(ModelMixin):
    def __init__(
        self,
        model_type=Model.WAN22_5B_T2V.value,
        patch_size=(1, 2, 2),
        text_len=512,
        in_dim=16,
        dim=1536,          
        ffn_dim=8960,      
        freq_dim=256,
        text_dim=4096,
        out_dim=16,
        num_heads=12,      
        num_layers=30,     
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
        flf_pos_embed_token_number=None,
        in_dim_ref_conv=None,
        wan_attn_block_class=WanAttentionBlock,
        **kwargs
    ):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            window_size (`tuple`, *optional*, defaults to (-1, -1)):
                Window size for local attention (-1 indicates global attention)
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        assert is_text_model(model_type) or is_img_model(model_type), "Unsupported WAN model, aborting"
        self.model_type = model_type
        self.model_arch_config: ModelArchConfig = Wan21ModelArchConfig(model_type=model_type)

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size, dtype=torch.float32)
        
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), 
            nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim)
        )

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), 
            nn.SiLU(), 
            nn.Linear(dim, dim)
        )
        
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        self.cross_attn_type = 't2v_cross_attn' if is_text_model(self.model_type) else 'i2v_cross_attn'
        self.blocks = nn.ModuleList([
            wan_attn_block_class(self.cross_attn_type, dim, ffn_dim, num_heads,
                                 window_size, qk_norm, cross_attn_norm, eps)
            for _ in range(num_layers)
        ])

        # head
        self.head = Head(dim, out_dim, patch_size, eps)

        d = dim // num_heads
        self.rope_embedder = EmbedND(dim=d, theta=10000.0, axes_dim=[d - 4 * (d // 6), 2 * (d // 6), 2 * (d // 6)])

        if is_img_model(self.model_type):
            self.img_emb = MLPProj(1280, dim, flf_pos_embed_token_number=flf_pos_embed_token_number)
        else:
            self.img_emb = None

        if in_dim_ref_conv is not None:
            self.ref_conv = nn.Conv2d(in_dim_ref_conv, dim, kernel_size=patch_size[1:], stride=patch_size[1:])
        else:
            self.ref_conv = None
            
    def forward_orig(
        self,
        x,
        t,
        context,
        clip_fea=None,
        freqs=None,
        reference_latent=None,
        **kwargs,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (Tensor):
                List of input video tensors with shape [B, C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [B, L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        # embeddings
        with torch.autocast(device_type=VRAM_DEVICE, dtype=torch.float32):
            x = self.patch_embedding(x.float()).to(x.dtype)

        grid_sizes = x.shape[2:]
        x = x.flatten(2).transpose(1, 2)

        # time embeddings
        e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t.flatten()).to(dtype=x[0].dtype))
        e = e.reshape(t.shape[0], -1, e.shape[-1])
        e0 = self.time_projection(e).unflatten(2, (6, self.dim))

        # injecting reference image
        full_ref = None
        if self.ref_conv is not None:
            full_ref = reference_latent
            if full_ref is not None:
                full_ref = self.ref_conv(full_ref).flatten(2).transpose(1, 2)
                x = torch.concat((full_ref, x), dim=1)

        # context
        context = self.text_embedding(context)

        # clip conditioning
        context_img_len = None
        if clip_fea is not None:
            if self.img_emb is not None:
                context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
                context = torch.concat([context_clip, context], dim=1)
            else: app_logger.warning(f"{self.__class__.__name__} doesn't support img embeds")
            context_img_len = clip_fea.shape[-2]

        # running through all the attn blocks
        for i, block in enumerate(self.blocks):
            x = block(x, e=e0, freqs=freqs, context=context, context_img_len=context_img_len)
        
        # head
        x = self.head(x, e)

        if full_ref is not None:
            x = x[:, full_ref.shape[1]:]

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        
        return x

    def rope_encode(self, t, h, w, t_start=0, time_indices_map=None, steps_t=None, steps_h=None, steps_w=None, device=None, dtype=None):
        # (dimension + half_patch_size) // patch_size (effectively rounding)
        patch_size = self.patch_size
        t_len = ((t + (patch_size[0] // 2)) // patch_size[0])
        h_len = ((h + (patch_size[1] // 2)) // patch_size[1])
        w_len = ((w + (patch_size[2] // 2)) // patch_size[2])

        # steps can be used to stretch the PEs
        if steps_t is None: steps_t = t_len
        if steps_h is None: steps_h = h_len
        if steps_w is None: steps_w = w_len

        # FIX: doing higher precision calc for now
        # (this has marginal benefits on multi object scenes)
        dtype = torch.float32
        img_ids = torch.zeros((steps_t, steps_h, steps_w, 3), device=device, dtype=dtype)
        
        t_seq = torch.linspace(t_start, t_start + (t_len - 1), steps=steps_t, device=device, dtype=dtype)
        # TODO / PONDER: is there a better place to put this logic so it is more universal ?
        # NOTE: applying map override (Position -> Time Value)
        # e.g., {0: 0.0} replaces the first frame's time with 0
        if time_indices_map:
            for idx, val in time_indices_map.items():
                if 0 <= idx < len(t_seq):
                    t_seq[idx] = val
                    
        img_ids[:, :, :, 0] = img_ids[:, :, :, 0] + t_seq.reshape(-1, 1, 1)
        img_ids[:, :, :, 1] = img_ids[:, :, :, 1] + torch.linspace(0, h_len - 1, steps=steps_h, device=device, dtype=dtype).reshape(1, -1, 1)
        img_ids[:, :, :, 2] = img_ids[:, :, :, 2] + torch.linspace(0, w_len - 1, steps=steps_w, device=device, dtype=dtype).reshape(1, 1, -1)
        img_ids = img_ids.reshape(1, -1, img_ids.shape[-1])

        freqs = self.rope_embedder(img_ids).movedim(1, 2)
        return freqs

    def forward(self, x, timestep, cross_attn, visual_embedding=None, reference_latent=None, **kwargs):
        bs, c, t, h, w = x.shape
        extra_channels = self.patch_embedding.weight.shape[1] - c
        if reference_latent is not None and extra_channels == reference_latent.shape[1]:
            # Input (16) + Cond (20) -> 36 Channels
            x = torch.cat([x, reference_latent], dim=1)
        else:
            # PONDER: should this be a model specific logic or a general cleaning step
            # filling up extra channels 
            if extra_channels > 0:
                null_conditioning = torch.zeros(
                    (x.shape[0], extra_channels, t, h, w), 
                    device=x.device, 
                    dtype=x.dtype
                )
                x = torch.cat([x, null_conditioning], dim=1)

        x = pad_to_patch_size(x, self.patch_size)
        t_len = t
        if self.ref_conv is not None and reference_latent is not None:
            t_len += 1      # the single latent that has been passed (wan specific logic)

        t_start = kwargs.get("t_start", 0)
        time_indices_map = kwargs.get("time_indices_map", None)
        freqs = self.rope_encode(t_len, h, w, t_start, time_indices_map=time_indices_map, device=x.device, dtype=x.dtype)
        # from torch_tracer import TorchTracer
        # with TorchTracer("./exiv_2.pkl"):
        out = self.forward_orig(x, timestep, cross_attn, clip_fea=visual_embedding, freqs=freqs, reference_latent=reference_latent, **kwargs)[:, :, :t, :h, :w]
        return out      # new variable for debugging purposes

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [L, C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        u = x
        b = u.shape[0]
        u = u[:, :math.prod(grid_sizes)].view(b, *grid_sizes, *self.patch_size, c)
        u = torch.einsum('bfhwpqrc->bcfphqwr', u)
        u = u.reshape(b, c, *[i * j for i, j in zip(grid_sizes, self.patch_size)])
        return u

    # ------------------------------------------------------------------------------------------
    # MODEL OVERRIDES
    # ------------------------------------------------------------------------------------------
    def get_memory_footprint_params(self):
        """
        Returns architectural constants for memory estimation.
        
        attn_factor: 
            ~2.0 - 2.5 (Optimized/Flash Attention): We only store Q,K,V and Output. The huge N×N attention matrix is never fully materialized.
            ~4.0 - 6.0 (Vanilla/Old Attention): Calculates and stores the full N×N attention map (memory intensive for long sequences).
        ffn_factor:
            normally 1.0, extra 0.5 for any overhead involved
        """
        try:
             dtype_size = torch.tensor([], dtype=self.dtype).element_size()
        except:
             dtype_size = 2
             
        return {
            "patch_size": self.patch_size,      # (1, 2, 2)
            "hidden_dim": self.dim,             # 5120
            "ffn_dim": self.ffn_dim,            # 13824
            "attn_factor": 30.0,
            "ffn_factor": 1.5,
            "dtype_size": dtype_size,
        }
        
    def get_model_sampling_obj(self):
        return get_model_sampling(ModelType.FLOW, {"sampling_settings": {"shift": 8}})
    
class Wan22Model(Wan21Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_arch_config: ModelArchConfig = Wan22ModelArchConfig(model_type=self.model_type)
        
    def scale_latent_inpaint(self, sigma, noise, latent_image, **kwargs):
        return latent_image
    
    def get_model_sampling_obj(self):
        shift = 5 if self.model_type == Model.WAN22_14B_TI2V.value else 8
        return get_model_sampling(ModelType.FLOW, {"sampling_settings": {"shift": shift}})

class VaceWanAttentionBlock(WanAttentionBlock):
    def __init__(
            self,
            cross_attn_type,
            dim,
            ffn_dim,
            num_heads,
            window_size=(-1, -1),
            qk_norm=True,
            cross_attn_norm=False,
            eps=1e-6,
            block_id=0
    ):
        super().__init__(cross_attn_type, dim, ffn_dim, num_heads, window_size, qk_norm, cross_attn_norm, eps)
        self.block_id = block_id
        if block_id == 0:
            self.before_proj = nn.Linear(self.dim, self.dim)
        self.after_proj = nn.Linear(self.dim, self.dim)

    def forward(self, c, x, **kwargs):
        prev_skips = []
        if self.block_id > 0:
            c_list = torch.unbind(c)
            c = c_list[-1]                  # current state
            prev_skips = list(c_list[:-1])  # skips till now
            
        if self.block_id == 0:
            c = self.before_proj(c) + x
        c = super().forward(c, **kwargs)
        c_skip = self.after_proj(c)
        return torch.stack(prev_skips + [c_skip, c])

class Wan21VaceModel(Wan22Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_arch_config: ModelArchConfig = Wan21ModelArchConfig(model_type=self.model_type)

        vace_layers, vace_in_dim = kwargs.get("vace_layers"), kwargs.get("vace_dim")
        # sparse vace layers, applying to alternate layers
        self.vace_in_dim = self.in_dim if vace_in_dim is None else vace_in_dim
        # assuming equal spacing, will update this later
        self.vace_layers_mapping = {i: n for n, i in enumerate(range(0, self.num_layers, self.num_layers // vace_layers))}

        # blocks
        self.blocks = nn.ModuleList([
            WanAttentionBlock(self.cross_attn_type, self.dim, self.ffn_dim, self.num_heads, self.window_size, self.qk_norm,
                                  self.cross_attn_norm, self.eps,)
            for i in range(self.num_layers)
        ])

        # vace blocks
        self.vace_blocks = nn.ModuleList([
            VaceWanAttentionBlock(self.cross_attn_type, self.dim, self.ffn_dim, self.num_heads, self.window_size, self.qk_norm,
                                     self.cross_attn_norm, self.eps, block_id=i)
            for i in range(vace_layers)
        ])

        # vace patch embeddings
        self.vace_patch_embedding = nn.Conv3d(
            self.vace_in_dim, self.dim, kernel_size=self.patch_size, stride=self.patch_size
        )
        
    def forward_vace(
        self,
        x,
        vace_context,
        vace_strengths,
        kwargs
    ):
        # embeddings
        orig_shape = list(vace_context.shape)
        vace_context = vace_context.movedim(0, 1).reshape([-1] + orig_shape[2:])
        c = self.vace_patch_embedding(vace_context).to(vace_context.dtype)
        c = c.flatten(2).transpose(1, 2)
        c = list(c.split(orig_shape[0], dim=0))

        new_kwargs = dict(x=x)
        new_kwargs.update(kwargs)
        accumulated_hints = None
        for i, c_state in enumerate(c):
            strength = vace_strengths[i] if i < len(vace_strengths) else vace_strengths[0]
            # stack(skip0, c_state) -> stack(skip0, skip1, c_state) -> ...
            for block in self.vace_blocks:
                c_state = block(c_state, **new_kwargs)
            unstacked = torch.unbind(c_state)   # (skip0, skip1, ..., skipN, c_state)
            skips = unstacked[:-1]              # (skip0, skip1, ..., skipN)
            if accumulated_hints is None: accumulated_hints = [s * strength for s in skips]
            else:
                for idx, s in enumerate(skips):
                    accumulated_hints[idx] += s * strength

        return accumulated_hints
    
    def forward_orig(
        self,
        x,
        t,
        cross_attn,
        vace_context=None,
        vace_strength=[1.0],
        freqs=None,
        **kwargs
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            cross_attn (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """

        assert vace_context is not None, "vace_context is required for this pass"
        # embeddings
        with torch.autocast(device_type=VRAM_DEVICE, dtype=torch.float32):
            x = self.patch_embedding(x.float()).to(x.dtype)

        grid_sizes = x.shape[2:]
        x = x.flatten(2).transpose(1, 2)

        # time embeddings
        e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t).to(dtype=x[0].dtype))
        e0 = self.time_projection(e).unflatten(1, (6, self.dim))

        # cross_attn
        cross_attn = self.text_embedding(cross_attn)

        # clip conditioning
        context_img_len = None
        # if clip_fea is not None:
        #     if self.img_emb is not None:
        #         context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
        #         context = torch.concat([context_clip, context], dim=1)
        #     else: app_logger.warning(f"{self.__class__.__name__} doesn't support img embeds")
        #     context_img_len = clip_fea.shape[-2]

        # arguments
        kwargs = dict(
            e=e0,
            freqs=freqs,
            context=cross_attn,
            context_img_len=context_img_len)

        hints = self.forward_vace(x, vace_context, vace_strength, kwargs)
        for block_idx, block in enumerate(self.blocks):
            x = block(x, **kwargs)
            if block_idx in self.vace_layers_mapping:
                hint_idx = self.vace_layers_mapping[block_idx]
                x = x + hints[hint_idx]

        # head
        x = self.head(x, e)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return x