# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
import math
from typing import List, Optional, Tuple

from .main import Wan21Model, sinusoidal_embedding_1d, repeat_e, Head
from ...attention import optimized_attention
from ....utils.device import VRAM_DEVICE

# ------------------------------------------------------------------------------------------
# FACE ENCODER & ADAPTER
# ------------------------------------------------------------------------------------------

class CausalConv1d(nn.Module):
    def __init__(self, chan_in, chan_out, kernel_size=3, stride=1, dilation=1, pad_mode="replicate", **kwargs):
        super().__init__()
        self.pad_mode = pad_mode
        padding = (kernel_size - 1, 0)
        self.time_causal_padding = padding
        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size, stride=stride, dilation=dilation, **kwargs)

    def forward(self, x):
        x = F.pad(x, self.time_causal_padding, mode=self.pad_mode)
        return self.conv(x)

class FaceEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.conv1_local = CausalConv1d(in_dim, 1024 * num_heads, 3, stride=1)
        self.norm1 = nn.LayerNorm(1024, elementwise_affine=False, eps=1e-6)
        self.act = nn.SiLU()
        self.conv2 = CausalConv1d(1024, 1024, 3, stride=2)
        self.conv3 = CausalConv1d(1024, 1024, 3, stride=2)
        self.out_proj = nn.Linear(1024, hidden_dim)
        self.norm2 = nn.LayerNorm(1024, elementwise_affine=False, eps=1e-6)
        self.norm3 = nn.LayerNorm(1024, elementwise_affine=False, eps=1e-6)
        self.padding_tokens = nn.Parameter(torch.zeros(1, 1, 1, hidden_dim))

    def forward(self, x):
        x = rearrange(x, "b t c -> b c t")
        b, c, t = x.shape
        x = self.conv1_local(x)
        x = rearrange(x, "b (n c) t -> (b n) t c", n=self.num_heads)
        x = self.norm1(x)
        x = self.act(x)
        x = rearrange(x, "b t c -> b c t")
        x = self.conv2(x)
        x = rearrange(x, "b c t -> b t c")
        x = self.norm2(x)
        x = self.act(x)
        x = rearrange(x, "b t c -> b c t")
        x = self.conv3(x)
        x = rearrange(x, "b c t -> b t c")
        x = self.norm3(x)
        x = self.act(x)
        x = self.out_proj(x)
        x = rearrange(x, "(b n) t c -> b t n c", b=b)
        padding = self.padding_tokens.repeat(b, x.shape[1], 1, 1)
        x = torch.cat([x, padding], dim=-2)
        return x

class FaceBlock(nn.Module):
    def __init__(self, hidden_size: int, heads_num: int, qk_norm: bool = True):
        super().__init__()
        self.hidden_size = hidden_size
        self.heads_num = heads_num
        head_dim = hidden_size // heads_num
        self.linear1_kv = nn.Linear(hidden_size, hidden_size * 2)
        self.linear1_q = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.q_norm = nn.RMSNorm(head_dim, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()
        self.k_norm = nn.RMSNorm(head_dim, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()
        self.pre_norm_feat = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.pre_norm_motion = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

    def forward(self, x: torch.Tensor, motion_vec: torch.Tensor, motion_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T_comp, N, C = motion_vec.shape
        x_motion = self.pre_norm_motion(motion_vec)
        x_feat = self.pre_norm_feat(x)
        kv = self.linear1_kv(x_motion)
        q = self.linear1_q(x_feat)
        k, v = rearrange(kv, "B L N (K H D) -> K B L N H D", K=2, H=self.heads_num)
        q = rearrange(q, "B S (H D) -> B S H D", H=self.heads_num)
        q = self.q_norm(q).to(v)
        k = self.k_norm(k).to(v)
        k = rearrange(k, "B L N H D -> (B L) N H D")
        v = rearrange(v, "B L N H D -> (B L) N H D")
        q = rearrange(q, "B (L S) H D -> (B L) S H D", L=T_comp)
        
        attn = optimized_attention(
            q.flatten(2), 
            k.flatten(2), 
            v.flatten(2), 
            heads=self.heads_num
        )
        
        attn = rearrange(attn, "(B L) S C -> B (L S) C", L=T_comp)
        output = self.linear2(attn)
        if motion_mask is not None:
            output = output * rearrange(motion_mask, "B T H W -> B (T H W)").unsqueeze(-1)
        return output

class FaceAdapter(nn.Module):
    def __init__(self, hidden_dim: int, heads_num: int, num_adapter_layers: int):
        super().__init__()
        self.fuser_blocks = nn.ModuleList([
            FaceBlock(hidden_dim, heads_num) for _ in range(num_adapter_layers)
        ])

def custom_qr(input_tensor):
    original_dtype = input_tensor.dtype
    if original_dtype in [torch.bfloat16, torch.float16]:
        q, r = torch.linalg.qr(input_tensor.to(torch.float32))
        return q.to(original_dtype), r.to(original_dtype)
    return torch.linalg.qr(input_tensor)

def fused_leaky_relu(input, bias, negative_slope=0.2, scale=2 ** 0.5):
    return F.leaky_relu(input + bias, negative_slope) * scale

def upfirdn2d_native(input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1):
    _, minor, in_h, in_w = input.shape
    kernel_h, kernel_w = kernel.shape

    out = input.view(-1, minor, in_h, 1, in_w, 1)
    out = F.pad(out, [0, up_x - 1, 0, 0, 0, up_y - 1, 0, 0])
    out = out.view(-1, minor, in_h * up_y, in_w * up_x)

    out = F.pad(out, [max(pad_x0, 0), max(pad_x1, 0), max(pad_y0, 0), max(pad_y1, 0)])
    out = out[:, :, max(-pad_y0, 0): out.shape[2] - max(-pad_y1, 0),
          max(-pad_x0, 0): out.shape[3] - max(-pad_x1, 0), ]

    out = out.reshape([-1, 1, in_h * up_y + pad_y0 + pad_y1, in_w * up_x + pad_x0 + pad_x1])
    w = torch.flip(kernel, [0, 1]).view(1, 1, kernel_h, kernel_w)
    out = F.conv2d(out, w)
    out = out.reshape(-1, minor, in_h * up_y + pad_y0 + pad_y1 - kernel_h + 1,
                      in_w * up_x + pad_x0 + pad_x1 - kernel_w + 1, )
    return out[:, :, ::down_y, ::down_x]

def upfirdn2d(input, kernel, up=1, down=1, pad=(0, 0)):
    return upfirdn2d_native(input, kernel, up, up, down, down, pad[0], pad[1], pad[0], pad[1])

def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)
    if k.ndim == 1:
        k = k[None, :] * k[:, None]
    k /= k.sum()
    return k

class FusedLeakyReLU(nn.Module):
    def __init__(self, channel, negative_slope=0.2, scale=2 ** 0.5):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1, channel, 1, 1))
        self.negative_slope = negative_slope
        self.scale = scale

    def forward(self, input):
        return fused_leaky_relu(input, self.bias, self.negative_slope, self.scale)

class Blur(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()
        kernel = make_kernel(kernel)
        if upsample_factor > 1:
            kernel = kernel * (upsample_factor ** 2)
        self.register_buffer('kernel', kernel)
        self.pad = pad

    def forward(self, input):
        return upfirdn2d(input, self.kernel, pad=self.pad)

class ScaledLeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.2):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, input):
        return F.leaky_relu(input, negative_slope=self.negative_slope)

class EqualConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_channel, in_channel, kernel_size, kernel_size))
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)
        self.stride = stride
        self.padding = padding
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))
        else:
            self.bias = None

    def forward(self, input):
        return F.conv2d(input, self.weight * self.scale, bias=self.bias, stride=self.stride, padding=self.padding)

class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))
        else:
            self.bias = None
        self.activation = activation
        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul)
        else:
            out = F.linear(input, self.weight * self.scale, bias=self.bias * self.lr_mul)
        return out

class ConvLayer(nn.Sequential):
    def __init__(
            self,
            in_channel,
            out_channel,
            kernel_size,
            downsample=False,
            blur_kernel=[1, 3, 3, 1],
            bias=True,
            activate=True,
    ):
        layers = []

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            layers.append(Blur(blur_kernel, pad=(pad0, pad1)))

            stride = 2
            self.padding = 0

        else:
            stride = 1
            self.padding = kernel_size // 2

        layers.append(EqualConv2d(in_channel, out_channel, kernel_size, padding=self.padding, stride=stride,
                                  bias=bias and not activate))

        if activate:
            if bias:
                layers.append(FusedLeakyReLU(out_channel))
            else:
                layers.append(ScaledLeakyReLU(0.2))

        super().__init__(*layers)

class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, blur_kernel=[1, 3, 3, 1]):
        super().__init__()
        self.conv1 = ConvLayer(in_channel, in_channel, 3)
        self.conv2 = ConvLayer(in_channel, out_channel, 3, downsample=True)
        self.skip = ConvLayer(in_channel, out_channel, 1, downsample=True, activate=False, bias=False)

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)
        skip = self.skip(input)
        return (out + skip) / math.sqrt(2)

class EncoderApp(nn.Module):
    def __init__(self, size, w_dim=512):
        super().__init__()
        channels = {4: 512, 8: 512, 16: 512, 32: 512, 64: 256, 128: 128, 256: 64, 512: 32, 1024: 16}
        self.w_dim = w_dim
        log_size = int(math.log(size, 2))
        self.convs = nn.ModuleList()
        self.convs.append(ConvLayer(3, channels[size], 1))
        in_channel = channels[size]
        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]
            self.convs.append(ResBlock(in_channel, out_channel))
            in_channel = out_channel
        self.convs.append(EqualConv2d(in_channel, self.w_dim, 4, padding=0, bias=False))

    def forward(self, x):
        h = x
        for conv in self.convs: h = conv(h)
        return h.squeeze(-1).squeeze(-1)

class Encoder(nn.Module):
    def __init__(self, size=512, style_dim=512, motion_dim=20):
        super().__init__()
        self.net_app = EncoderApp(size, style_dim)
        fc = [EqualLinear(style_dim, style_dim) for _ in range(4)]
        fc.append(EqualLinear(style_dim, motion_dim))
        self.fc = nn.Sequential(*fc)

    def encode_motion(self, x):
        return self.fc(self.net_app(x))

class Direction(nn.Module):
    def __init__(self, motion_dim=20):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(512, motion_dim))
        self.motion_dim = motion_dim

    def forward(self, input):
        weight = self.weight + 1e-8
        Q, _ = custom_qr(weight)
        input_diag = torch.diag_embed(input)
        out = torch.matmul(input_diag, Q.T)
        return torch.sum(out, dim=1)

class Synthesis(nn.Module):
    def __init__(self, motion_dim=20):
        super().__init__()
        self.direction = Direction(motion_dim)

class Generator(nn.Module):
    def __init__(self, size=512, style_dim=512, motion_dim=20):
        super().__init__()
        self.enc = Encoder(size, style_dim, motion_dim)
        self.dec = Synthesis(motion_dim)

    def get_motion(self, img):
        motion_feat = self.enc.encode_motion(img)
        return self.dec.direction(motion_feat)

# ------------------------------------------------------------------------------------------
# WAN ANIMATE MODEL
# ------------------------------------------------------------------------------------------

class WanAnimateModel(Wan21Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Extra patch embedding for pose latents
        self.pose_patch_embedding = nn.Conv3d(
            16, self.dim, kernel_size=self.patch_size, stride=self.patch_size
        )
        
        # face conditioning components
        self.motion_encoder = Generator(size=512, style_dim=512, motion_dim=20)
        self.face_encoder = FaceEncoder(
            in_dim=kwargs.get("motion_encoder_dim", 512),
            hidden_dim=self.dim,
            num_heads=4
        )
        self.face_adapter = FaceAdapter(
            hidden_dim=self.dim,
            heads_num=self.num_heads,
            num_adapter_layers=self.num_layers // 5
        )

    def forward_orig(
        self,
        x,
        t,
        context,
        clip_fea=None,
        freqs=None,
        reference_latent=None,
        pose_latents=None,
        face_pixel_values=None,
        **kwargs
    ):
        """
        Forward pass for Wan Animate Model.
        
        Args:
            x (Tensor): Main latents [B, 36, T, H, W] (if reference_latent is already concatenated)
                        or [B, 16, T, H, W].
            t (Tensor): Timesteps [B].
            context (Tensor): T5 text embeddings [B, L, C].
            clip_fea (Tensor, optional): CLIP visual features [B, 257, C].
            freqs (Tensor, optional): RoPE frequencies.
            reference_latent (Tensor, optional): 20-channel conditioning latents [B, 20, T, H, W].
                Contains [mask (4), encoded_reference_frames (16)].
            pose_latents (Tensor, optional): 16-channel pose latents [B, 16, T, H, W].
                These are added to the main latents after patch embedding.
            face_pixel_values (Tensor, optional): Raw face pixel values [B, 3, T, 512, 512].
                Used for motion and face identity conditioning via motion_encoder and face_adapter.
        """
        with torch.autocast(device_type=VRAM_DEVICE, dtype=torch.float32):
            x_emb = self.patch_embedding(x.float()).to(x.dtype)

        if pose_latents is not None:
            # pose_latents is assumed to be encoded latents of pose frames [B, 16, T, H, W]
            with torch.autocast(device_type=VRAM_DEVICE, dtype=torch.float32):
                p_emb = self.pose_patch_embedding(pose_latents.float()).to(x.dtype)
            
            # if reference image was prepended, skip the first frame in x_emb for pose addition
            if x_emb.shape[2] > p_emb.shape[2]: x_emb[:, :, 1:] += p_emb
            else: x_emb += p_emb

        grid_sizes = x_emb.shape[2:]
        x_flat = x_emb.flatten(2).transpose(1, 2)

        # face motion conditioning
        motion_vec = None
        if face_pixel_values is not None:
            # face_pixel_values: [B, C, T, H, W]
            B, C, T, H, W = face_pixel_values.shape
            face_flat = rearrange(face_pixel_values, "b c t h w -> (b t) c h w")
            
            # process in batches to save memory
            encode_bs = 8
            motion_list = []
            for i in range(0, face_flat.shape[0], encode_bs):
                motion_list.append(self.motion_encoder.get_motion(face_flat[i : i + encode_bs]))
            
            motion_vec = torch.cat(motion_list)
            motion_vec = rearrange(motion_vec, "(b t) c -> b t c", t=T)
            motion_vec = self.face_encoder(motion_vec) # [B, T, num_heads, head_dim]
            
            # pad for reference frame if necessary
            if x_emb.shape[2] > T:
                pad_face = torch.zeros(B, 1, motion_vec.shape[2], motion_vec.shape[3]).to(motion_vec)
                motion_vec = torch.cat([pad_face, motion_vec], dim=1)

        # time embeddings
        e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t.flatten()).to(dtype=x_flat.dtype))
        e = e.reshape(t.shape[0], -1, e.shape[-1])
        e0 = self.time_projection(e).unflatten(2, (6, self.dim))

        # text / clip embeddings
        context = self.text_embedding(context)
        context_img_len = None
        if clip_fea is not None and self.img_emb is not None:
            context_clip = self.img_emb(clip_fea)
            context = torch.concat([context_clip, context], dim=1)
            context_img_len = clip_fea.shape[-2]

        # transformer blocks with face adapter
        for i, block in enumerate(self.blocks):
            x_flat = block(x_flat, e=e0, freqs=freqs, context=context, context_img_len=context_img_len)
            if motion_vec is not None and i % 5 == 0:   # face adapter every 5 blocks
                adapter_idx = i // 5
                if adapter_idx < len(self.face_adapter.fuser_blocks):
                    x_flat = x_flat + self.face_adapter.fuser_blocks[adapter_idx](x_flat, motion_vec)
        
        x_out = self.head(x_flat, e)
        
        return self.unpatchify(x_out, grid_sizes)
