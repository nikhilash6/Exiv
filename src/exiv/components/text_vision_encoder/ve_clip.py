import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from .encoder_base import VisionEncoder
from ..attention import optimized_attention
from ..activations import get_activation
from ...utils.dtype import cast_like_reference

# for xlm_roberta_vit_l check this - https://huggingface.co/M-CLIP/XLM-Roberta-Large-Vit-L-14

class CLIPAttention(torch.nn.Module):
    def __init__(self, embed_dim, heads, dtype, device):
        super().__init__()

        self.heads = heads
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True, dtype=dtype, device=device)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True, dtype=dtype, device=device)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True, dtype=dtype, device=device)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True, dtype=dtype, device=device)

    def forward(self, x, mask=None):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        out = optimized_attention(q, k, v, self.heads, mask)
        return self.out_proj(out)
    
class CLIPMLP(torch.nn.Module):
    def __init__(self, embed_dim, intermediate_size, activation, dtype, device):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, intermediate_size, bias=True, dtype=dtype, device=device)
        self.activation = get_activation(activation)
        self.fc2 = nn.Linear(intermediate_size, embed_dim, bias=True, dtype=dtype, device=device)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x

class CLIPLayer(torch.nn.Module):
    def __init__(self, embed_dim, heads, intermediate_size, intermediate_activation, dtype, device):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(embed_dim, dtype=dtype, device=device)
        self.self_attn = CLIPAttention(embed_dim, heads, dtype, device)
        self.layer_norm2 = nn.LayerNorm(embed_dim, dtype=dtype, device=device)
        self.mlp = CLIPMLP(embed_dim, intermediate_size, intermediate_activation, dtype, device)

    def forward(self, x, mask=None, optimized_attention=None):
        x += self.self_attn(self.layer_norm1(x), mask, optimized_attention)
        x += self.mlp(self.layer_norm2(x))
        return x

class CLIPEncoder(torch.nn.Module):
    def __init__(self, num_layers, embed_dim, heads, intermediate_size, intermediate_activation, dtype, device):
        super().__init__()
        self.layers = torch.nn.ModuleList([CLIPLayer(embed_dim, heads, intermediate_size, intermediate_activation, dtype, device) for i in range(num_layers)])

    def forward(self, x, mask=None, intermediate_output=None):
        optimized_attention = optimized_attention(x.device, mask=mask is not None, small_input=True)

        all_intermediate = None
        if intermediate_output is not None:
            if intermediate_output == "all":
                all_intermediate = []
                intermediate_output = None
            elif intermediate_output < 0:
                intermediate_output = len(self.layers) + intermediate_output

        intermediate = None
        for i, l in enumerate(self.layers):
            x = l(x, mask, optimized_attention)
            if i == intermediate_output:
                intermediate = x.clone()
            if all_intermediate is not None:
                all_intermediate.append(x.unsqueeze(1).clone())

        if all_intermediate is not None:
            intermediate = torch.cat(all_intermediate, dim=1)

        return x, intermediate
    
class CLIPVisionEmbeddings(torch.nn.Module):
    def __init__(self, embed_dim, num_channels=3, patch_size=14, image_size=224, model_type="", dtype=None, device=None):
        super().__init__()

        num_patches = (image_size // patch_size) ** 2
        if model_type == "siglip_vision_model":
            self.class_embedding = None
            patch_bias = True
        else:
            num_patches = num_patches + 1
            self.class_embedding = torch.nn.Parameter(torch.empty(embed_dim, dtype=dtype, device=device))
            patch_bias = False

        self.patch_embedding = nn.Conv2d(
            in_channels=num_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=patch_bias,
            dtype=dtype,
            device=device
        )

        self.position_embedding = nn.Embedding(num_patches, embed_dim, dtype=dtype, device=device)

    def forward(self, pixel_values):
        embeds = self.patch_embedding(pixel_values).flatten(2).transpose(1, 2)
        if self.class_embedding is not None:
            embeds = torch.cat([cast_like_reference(self.class_embedding, embeds).expand(pixel_values.shape[0], 1, -1), embeds], dim=1)
        return embeds + cast_like_reference(self.position_embedding.weight, embeds)

class CLIPVision(torch.nn.Module):
    def __init__(self, config_dict, dtype, device):
        super().__init__()
        num_layers = config_dict["num_hidden_layers"]
        embed_dim = config_dict["hidden_size"]
        heads = config_dict["num_attention_heads"]
        intermediate_size = config_dict["intermediate_size"]
        intermediate_activation = config_dict["hidden_act"]
        model_type = config_dict["model_type"]

        self.embeddings = CLIPVisionEmbeddings(
            embed_dim, 
            config_dict["num_channels"], 
            config_dict["patch_size"], 
            config_dict["image_size"], 
            model_type=model_type, 
            dtype=dtype, 
            device=device
        )
        if model_type == "siglip_vision_model":
            self.pre_layrnorm = lambda a: a
            self.output_layernorm = True
        else:
            self.pre_layrnorm = nn.LayerNorm(embed_dim)
            self.output_layernorm = False
        
        self.encoder = CLIPEncoder(num_layers, embed_dim, heads, intermediate_size, intermediate_activation, dtype, device)
        self.post_layernorm = nn.LayerNorm(embed_dim)

    def forward(self, pixel_values, attention_mask=None, intermediate_output=None):
        x = self.embeddings(pixel_values)
        x = self.pre_layrnorm(x)
        #TODO: attention_mask?
        x, i = self.encoder(x, mask=None, intermediate_output=intermediate_output)
        if self.output_layernorm:
            x = self.post_layernorm(x)
            pooled_output = x
        else:
            pooled_output = self.post_layernorm(x[:, 0, :])
        return x, i, pooled_output
    
class LlavaProjector(torch.nn.Module):
    def __init__(self, in_dim, out_dim, dtype, device, operations):
        super().__init__()
        self.linear_1 = operations.Linear(in_dim, out_dim, bias=True, device=device, dtype=dtype)
        self.linear_2 = operations.Linear(out_dim, out_dim, bias=True, device=device, dtype=dtype)

    def forward(self, x):
        return self.linear_2(torch.nn.functional.gelu(self.linear_1(x[:, 1:])))

class CLIPViTL(VisionEncoder):
    """
    -   Identified by: Layer count (e.g., "layers.22" for L, "layers.30" for H, "layers.47" for G).
    -   Trained by Original ViT-L by OpenAI. ViT-H and ViT-G by LAION (OpenCLIP project).
    -   Contrastive Language-Image Pre-training (CLIP). Trained on billions
        of (image, text) pairs to learn the relationship between visual concepts and text.
    -   Good At: Excellent for text-to-image similarity and zero-shot image classification.
    -   Architectural Differences:
        * ViT-L (Large): 24 layers.
        * ViT-H (Huge): 32 layers.
        * ViT-G (Giant): 48 layers.
        * "...336.json" variants are trained on higher-res 336x336 images.
        * "...llava.json" is a variant for the LLaVA multimodal chatbot.
    """
    def __init__(self, dtype, device):
        super().__init__(dtype, device)
        
        config_dict = self._get_config()
        
        self.vision_model = CLIPVision(config_dict, dtype, device)
        self.visual_projection = nn.Linear(config_dict["hidden_size"], config_dict["projection_dim"], bias=False)
        self.multi_modal_projector = None
        
    def _get_config(self):
        return {
            "attention_dropout": 0.0,
            "dropout": 0.0,
            "hidden_act": "quick_gelu",
            "hidden_size": 1024,
            "image_size": 224,
            "initializer_factor": 1.0,
            "initializer_range": 0.02,
            "intermediate_size": 4096,
            "layer_norm_eps": 1e-05,
            "model_type": "clip_vision_model",
            "num_attention_heads": 16,
            "num_channels": 3,
            "num_hidden_layers": 24,
            "patch_size": 14,
            "projection_dim": 768,
            "torch_dtype": "float32"
        }


    def forward(self, *args, **kwargs):
        x = self.vision_model(*args, **kwargs)
        out = self.visual_projection(x[2])
        projected = None
        if self.multi_modal_projector is not None:
            projected = self.multi_modal_projector(x[1])

        return (x[0], x[1], out, projected)
    

class CLIPVitLlava(CLIPViTL):
    def __init__(self, dtype, device):
        super().__init__(dtype, device)
        
        config_dict = self._get_config()
        
        self.vision_model = CLIPVision(config_dict, dtype, device)
        self.visual_projection = lambda a: a
        self.multi_modal_projector = LlavaProjector(config_dict["hidden_size"], 4096, dtype, device)

    def _get_config(self):
        return {
            "attention_dropout": 0.0,
            "dropout": 0.0,
            "hidden_act": "quick_gelu",
            "hidden_size": 1024,
            "image_size": 336,
            "initializer_factor": 1.0,
            "initializer_range": 0.02,
            "intermediate_size": 4096,
            "layer_norm_eps": 1e-5,
            "model_type": "clip_vision_model",
            "num_attention_heads": 16,
            "num_channels": 3,
            "num_hidden_layers": 24,
            "patch_size": 14,
            "projection_dim": 768,
            "projector_type": "llava3",
            "torch_dtype": "float32"
        }
