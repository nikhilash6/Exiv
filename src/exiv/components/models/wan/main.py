# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.

import torch
import torch.nn as nn
from einops import rearrange

import math
import uuid

from ...enum import Model
from ...attention import optimized_attention
from ...positional_embeddings import EmbedND, apply_rope
from ...conditionals import CONDCrossAttn, CONDNoiseShape, CONDRegular
from ...latent_format import LatentFormat, Wan21VAELatentFormat
from ....model_utils.helper_methods import get_state_dict
from ....model_utils.model_mixin import ModelMixin
from ....model_utils.common_classes import ModelArchConfig
from ....utils.tensor import common_upscale, pad_to_patch_size, repeat_to_batch_size

class WanModelArchConfig(ModelArchConfig):
    latent_channels = 16


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

class Wan21ModelArchConfig(ModelArchConfig):
    latent_format: LatentFormat = Wan21VAELatentFormat()

class Wan21Model(ModelMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    14B config (will remove this junk after dynamic config loading) -- 
    model_type=Model.WANT2V.value,
        patch_size=(1, 2, 2),
        text_len=512,
        in_dim=16,
        dim=5120,
        ffn_dim=13824,
        freq_dim=256,
        text_dim=4096,
        out_dim=16,
        num_heads=40,
        num_layers=40,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
        flf_pos_embed_token_number=None,
        in_dim_ref_conv=None,
        wan_attn_block_class=WanAttentionBlock,
    """

    def __init__(
        self,
        model_type=Model.WANT2V.value,
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

        assert model_type in [Model.WANT2V.value, Model.WANTI2V.value]
        self.model_type = model_type
        self.model_arch_config: ModelArchConfig = Wan21ModelArchConfig()

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
        cross_attn_type = 't2v_cross_attn' if model_type == Model.WANT2V.value else 'i2v_cross_attn'
        self.blocks = nn.ModuleList([
            wan_attn_block_class(cross_attn_type, dim, ffn_dim, num_heads,
                                 window_size, qk_norm, cross_attn_norm, eps)
            for _ in range(num_layers)
        ])

        # head
        self.head = Head(dim, out_dim, patch_size, eps)

        d = dim // num_heads
        self.rope_embedder = EmbedND(dim=d, theta=10000.0, axes_dim=[d - 4 * (d // 6), 2 * (d // 6), 2 * (d // 6)])

        if model_type == Model.WANTI2V.value:
            self.img_emb = MLPProj(1280, dim, flf_pos_embed_token_number=flf_pos_embed_token_number)
        else:
            self.img_emb = None

        if in_dim_ref_conv is not None:
            self.ref_conv = nn.Conv2d(in_dim_ref_conv, dim, kernel_size=patch_size[1:], stride=patch_size[1:])
        else:
            self.ref_conv = None
            
    def concat_cond(self, **kwargs):
        noise = kwargs.get("noise", None)
        
        extra_channels = self.patch_embedding.weight.shape[1] - noise.shape[1]
        if extra_channels == 0:
            return None

        image = kwargs.get("concat_latent_image", None)
        device = kwargs["device"]

        # CASE 1: No reference image was provided
        if image is None:
            shape_image = list(noise.shape)
            shape_image[1] = extra_channels
            image = torch.zeros(shape_image, dtype=noise.dtype, layout=noise.layout, device=noise.device)
        else:
            # CASE 2: A reference image exists.
            latent_dim = self.latent_format.latent_channels
            
            # Upscale/Resize the reference image
            image = common_upscale(image.to(device), noise.shape[-1], noise.shape[-2], "bilinear", "center")
            
            # Process the latents to match the model's expected distribution
            for i in range(0, image.shape[1], latent_dim):
                image[:, i: i + latent_dim] = self.process_latent_in(image[:, i: i + latent_dim])
            
            # Ensure the batch size matches (e.g., repeating the image for every frame in the batch).
            image = repeat_to_batch_size(image, noise.shape[0])

        # Handle mismatch in channel counts (e.g., if using a specific I2V model vs T2V)
        if extra_channels != image.shape[1] + 4:
            if not self.image_to_video or extra_channels == image.shape[1]:
                return image

        # Truncate the image if it has too many channels for the available slots
        # (e.g. reserving 4 channels for the mask).
        if image.shape[1] > (extra_channels - 4):
            image = image[:, :(extra_channels - 4)]

        # --- Mask Processing ---
        
        mask = kwargs.get("concat_mask", kwargs.get("denoise_mask", None))
        if mask is None:
            mask = torch.zeros_like(noise)[:, :4]
        else:
            if mask.shape[1] != 4:
                mask = torch.mean(mask, dim=1, keepdim=True)
            
            mask = 1.0 - mask
            mask = common_upscale(mask.to(device), noise.shape[-1], noise.shape[-2], "bilinear", "center")
            
            # Pad the mask
            if mask.shape[-3] < noise.shape[-3]:
                mask = torch.nn.functional.pad(mask, (0, 0, 0, 0, 0, noise.shape[-3] - mask.shape[-3]), mode='constant', value=0)
            
            # Expand a 1-channel mask to 4 channels
            if mask.shape[1] == 1:
                mask = mask.repeat(1, 4, 1, 1, 1)
            
            # Ensure batch size alignment.
            mask = repeat_to_batch_size(mask, noise.shape[0])

        # --- Final Assembly ---
        
        # Concatenate the Mask and the Image together
        concat_mask_index = kwargs.get("concat_mask_index", 0)
        if concat_mask_index != 0:
            return torch.cat((image[:, :concat_mask_index], mask, image[:, concat_mask_index:]), dim=1)
        else:
            return torch.cat((mask, image), dim=1)
            
    def format_conds(self, *args, **kwargs):
        out = {}
        cross_attn = kwargs.get("cross_attn", None)
        if cross_attn is not None:
            out['c_crossattn'] = CONDCrossAttn(cross_attn)

        clip_vision_output = kwargs.get("clip_vision_output", None)
        if clip_vision_output is not None:
            out['clip_fea'] = CONDRegular(clip_vision_output["penultimate_hidden_states"])

        time_dim_concat = kwargs.get("time_dim_concat", None)
        if time_dim_concat is not None:
            out['time_dim_concat'] = CONDRegular(self.process_latent_in(time_dim_concat))

        reference_latents = kwargs.get("reference_latents", None)
        if reference_latents is not None:
            out['reference_latent'] = CONDRegular(self.process_latent_in(reference_latents[-1])[:, :, 0])

        concat_cond = self.concat_cond(**kwargs)
        if concat_cond is not None:
            out['c_concat'] = CONDNoiseShape(concat_cond)

        cross_attn_cnet = kwargs.get("cross_attn_controlnet", None)
        if cross_attn_cnet is not None:
            out['crossattn_controlnet'] = CONDCrossAttn(cross_attn_cnet)

        c_concat = kwargs.get("noise_concat", None)
        if c_concat is not None:
            out['c_concat'] = CONDNoiseShape(c_concat)
        
        return out

    def forward_orig(
        self,
        x,
        t,
        context,
        clip_fea=None,
        freqs=None,
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
        x_input_debug = x.detach().clone().float().cpu()
        
        # embeddings
        with torch.autocast(device_type="cuda", dtype=torch.float32):
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
            full_ref = kwargs.get("reference_latent", None)
            if full_ref is not None:
                full_ref = self.ref_conv(full_ref).flatten(2).transpose(1, 2)
                x = torch.concat((full_ref, x), dim=1)

        # context
        context = self.text_embedding(context)

        # clip conditioning
        context_img_len = None
        if False and clip_fea is not None:
            if self.img_emb is not None:
                context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
                context = torch.concat([context_clip, context], dim=1)
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
        
        x_output_debug = x.detach().float().cpu()

        # Metric 1: Did the values change? (Mean Absolute Difference)
        # If this is 0.0, your model is essentially an identity function (doing nothing).
        diff = (x_input_debug - x_output_debug).abs().mean().item()
        
        # Metric 2: Is the output dead?
        # If output std is 0.0, your model has collapsed (outputting constant values).
        out_std = x_output_debug.std().item()

        print(f"--- Forward Pass Check ---")
        print(f"Input Mean/Std:  {x_input_debug.mean():.4f} / {x_input_debug.std():.4f}")
        print(f"Output Mean/Std: {x_output_debug.mean():.4f} / {x_output_debug.std():.4f}")
        print(f"Avg Change (Diff): {diff:.6f}") # Should be significant (e.g., > 0.1)
        
        return x

    def rope_encode(self, t, h, w, t_start=0, steps_t=None, steps_h=None, steps_w=None, device=None, dtype=None):
        patch_size = self.patch_size
        t_len = ((t + (patch_size[0] // 2)) // patch_size[0])
        h_len = ((h + (patch_size[1] // 2)) // patch_size[1])
        w_len = ((w + (patch_size[2] // 2)) // patch_size[2])

        if steps_t is None:
            steps_t = t_len
        if steps_h is None:
            steps_h = h_len
        if steps_w is None:
            steps_w = w_len

        img_ids = torch.zeros((steps_t, steps_h, steps_w, 3), device=device, dtype=dtype)
        img_ids[:, :, :, 0] = img_ids[:, :, :, 0] + torch.linspace(t_start, t_start + (t_len - 1), steps=steps_t, device=device, dtype=dtype).reshape(-1, 1, 1)
        img_ids[:, :, :, 1] = img_ids[:, :, :, 1] + torch.linspace(0, h_len - 1, steps=steps_h, device=device, dtype=dtype).reshape(1, -1, 1)
        img_ids[:, :, :, 2] = img_ids[:, :, :, 2] + torch.linspace(0, w_len - 1, steps=steps_w, device=device, dtype=dtype).reshape(1, 1, -1)
        img_ids = img_ids.reshape(1, -1, img_ids.shape[-1])

        freqs = self.rope_embedder(img_ids).movedim(1, 2)
        return freqs

    def forward(self, x, timestep, context, clip_fea=None, **kwargs):
        bs, c, t, h, w = x.shape
        x = pad_to_patch_size(x, self.patch_size)

        t_len = t
        if self.ref_conv is not None and "reference_latent" in kwargs:
            t_len += 1      # the single latent that has been passed

        freqs = self.rope_encode(t_len, h, w, device=x.device, dtype=x.dtype)
        return self.forward_orig(x, timestep, context, clip_fea=clip_fea, freqs=freqs, **kwargs)[:, :, :t, :h, :w]

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

    def get_mapped_key(self, model_key=None, lora_key=None):
        if hasattr(self, "_cached_key_map"):
            return self._cached_key_map

    def get_memory_footprint_params(self):
        """
        Returns architectural constants for memory estimation.
        """
        try:
             dtype_size = torch.tensor([], dtype=self.dtype).element_size()
        except:
             dtype_size = 2
             
        return {
            "patch_size": self.patch_size,      # (1, 2, 2)
            "hidden_dim": self.dim,             # 5120
            "ffn_dim": self.ffn_dim,            # 13824
            # TODO: optimize this memory usage
            # 1. Query
            # 2. Key  
            # 3. Value
            # 4. RoPE Q
            # 5. RoPE K
            # 6. Attention Output / Buffer
            "attn_factor": 6.0, 
            
            # 1. Up Project
            # 2. Gate/Act  
            "ffn_factor": 2.0,
            
            "dtype_size": dtype_size
        }