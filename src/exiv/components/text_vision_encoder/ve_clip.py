import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from .ve_config import CLIPViTHConfig, CLIPViTLConfig, CLIPVitLlavaConfig
from .common import VisionEncoderOutput
from .encoder_base import VisionEncoder
from ..attention import optimized_attention
from ..activations import get_activation
from ...utils.dtype import cast_like_reference

# for xlm_roberta_vit_l check this - https://huggingface.co/M-CLIP/XLM-Roberta-Large-Vit-L-14

class CLIPAttention(torch.nn.Module):
    def __init__(self, embed_dim, heads):
        super().__init__()

        self.heads = heads
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

    def forward(self, x, mask=None):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        out = optimized_attention(q, k, v, self.heads, mask)
        return self.out_proj(out)
    
class CLIPMLP(torch.nn.Module):
    def __init__(self, embed_dim, intermediate_size, activation):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, intermediate_size, bias=True)
        self.activation = get_activation(activation)
        self.fc2 = nn.Linear(intermediate_size, embed_dim, bias=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x

class CLIPLayer(torch.nn.Module):
    def __init__(self, embed_dim, heads, intermediate_size, intermediate_activation):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.self_attn = CLIPAttention(embed_dim, heads)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.mlp = CLIPMLP(embed_dim, intermediate_size, intermediate_activation)

    def forward(self, x, mask=None):
        x += self.self_attn(self.layer_norm1(x), mask)
        x += self.mlp(self.layer_norm2(x))
        return x

class CLIPEncoder(torch.nn.Module):
    def __init__(self, num_layers, embed_dim, heads, intermediate_size, intermediate_activation):
        super().__init__()
        self.layers = torch.nn.ModuleList([CLIPLayer(embed_dim, heads, intermediate_size, intermediate_activation) for i in range(num_layers)])

    def forward(self, x, mask=None, intermediate_output=None):
        all_intermediate = None
        if intermediate_output is not None:
            if intermediate_output == "all":
                all_intermediate = []
                intermediate_output = None
            elif intermediate_output < 0:
                intermediate_output = len(self.layers) + intermediate_output

        intermediate = None
        for i, l in enumerate(self.layers):
            x = l(x, mask)
            if i == intermediate_output:
                intermediate = x.clone()
            if all_intermediate is not None:
                all_intermediate.append(x.unsqueeze(1).clone())

        if all_intermediate is not None:
            intermediate = torch.cat(all_intermediate, dim=1)

        return x, intermediate
    
class CLIPVisionEmbeddings(torch.nn.Module):
    def __init__(self, embed_dim, num_channels=3, patch_size=14, image_size=224, model_type=""):
        super().__init__()

        num_patches = (image_size // patch_size) ** 2
        if model_type == "siglip_vision_model":
            self.class_embedding = None
            patch_bias = True
        else:
            num_patches = num_patches + 1
            self.class_embedding = torch.nn.Parameter(torch.empty(embed_dim))
            patch_bias = False

        self.patch_embedding = nn.Conv2d(
            in_channels=num_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=patch_bias,
        )

        self.position_embedding = nn.Embedding(num_patches, embed_dim)

    def forward(self, pixel_values):
        embeds = self.patch_embedding(pixel_values).flatten(2).transpose(1, 2)
        if self.class_embedding is not None:
            embeds = torch.cat([cast_like_reference(self.class_embedding, embeds).expand(pixel_values.shape[0], 1, -1), embeds], dim=1)
        return embeds + cast_like_reference(self.position_embedding.weight, embeds)

class CLIPVision(torch.nn.Module):
    def __init__(self, config_dict):
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
        )
        if model_type == "siglip_vision_model":
            self.pre_layrnorm = lambda a: a
            self.output_layernorm = True
        else:
            self.pre_layrnorm = nn.LayerNorm(embed_dim)
            self.output_layernorm = False
        
        self.encoder = CLIPEncoder(num_layers, embed_dim, heads, intermediate_size, intermediate_activation)
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
    def __init__(self, in_dim, out_dim, operations):
        super().__init__()
        self.linear_1 = operations.Linear(in_dim, out_dim, bias=True)
        self.linear_2 = operations.Linear(out_dim, out_dim, bias=True)

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
    def __init__(self, model_path, dtype=None, device=None):
        super().__init__(model_path=model_path, dtype=dtype, device=device)
        
        self.config = self._get_config()
        
        self.vision_model = CLIPVision(self.config)
        self.visual_projection = nn.Linear(self.config["hidden_size"], self.config["projection_dim"], bias=False)
        self.multi_modal_projector = None
        self.return_all_hidden_states = False
        
    def _get_config(self):
        return CLIPViTLConfig()

    def forward(self, *args, **kwargs):
        # x[0] = last_hidden_state
        # x[1] = intermediate_hidden_states (all layers)
        # x[2] = pooled_output (raw)
        x = self.vision_model(*args, **kwargs)
        
        # calculate projections
        out = self.visual_projection(x[2])
        projected = None
        if self.multi_modal_projector is not None:
            projected = self.multi_modal_projector(x[1])

        return VisionEncoderOutput(
            last_hidden_state=x[0],
            intermediate_hidden_states=x[1],
            image_embedding=out,
            multimodal_projection=projected
        )
    

class CLIPViTH(VisionEncoder):
    def __init__(self, model_path, dtype=None, device=None):
        self.default_model_filename = "CLIP-ViT-H-fp16.safetensors"
        super().__init__(model_path, dtype=dtype, device=device)
        
        self.config = self._get_config()
        
        self.vision_model = CLIPVision(self.config)
        self.visual_projection = nn.Linear(self.config["hidden_size"], self.config["projection_dim"], bias=False)
        self.multi_modal_projector = None
        self.return_all_hidden_states = False
        
    def _get_config(self):
        return CLIPViTHConfig()
        
    def forward(self, *args, **kwargs):
        x = self.vision_model(*args, **kwargs)
        out = self.visual_projection(x[2])
        projected = None
        if self.multi_modal_projector is not None:
            projected = self.multi_modal_projector(x[1])

        return VisionEncoderOutput(
            last_hidden_state=x[0],
            intermediate_hidden_states=x[1],
            image_embedding=out,
            multimodal_projection=projected
        )
    

class CLIPVitLlava(CLIPViTL):
    def __init__(self, model_path, dtype=None, device=None):
        super().__init__(model_path, dtype=dtype, device=device)
        
        self.config = self._get_config()
        
        self.vision_model = CLIPVision(self.config)
        self.visual_projection = lambda a: a
        self.multi_modal_projector = LlavaProjector(self.config["hidden_size"], 4096)
        self.return_all_hidden_states = False

    def _get_config(self):
        return CLIPVitLlavaConfig()
        
    def forward(self, *args, **kwargs):
        x = self.vision_model(*args, **kwargs)
        out = self.visual_projection(x[2])
        projected = None
        if self.multi_modal_projector is not None:
            projected = self.multi_modal_projector(x[1])

        return VisionEncoderOutput(
            last_hidden_state=x[0],
            intermediate_hidden_states=x[1],
            image_embedding=out,
            multimodal_projection=projected
        )
