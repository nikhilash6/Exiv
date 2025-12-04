from pathlib import Path
import torch
from torch import Tensor

import re
import os
import safetensors

from ...utils.device import OFFLOAD_DEVICE, VRAM_DEVICE, ProcDevice
from ...utils.file import ensure_model_availability
from ...utils.logging import app_logger

# code adapted from Forge

def parse_prompt_attention(text):
    """
    Parses a string with attention tokens and returns a list of pairs: text and its associated weight.
    Accepted tokens are:
      (abc) - increases attention to abc by a multiplier of 1.1
      (abc:3.12) - increases attention to abc by a multiplier of 3.12
      [abc] - decreases attention to abc by a multiplier of 1.1
      \( - literal character '('
      \[ - literal character '['
      \) - literal character ')'
      \] - literal character ']'
      \\ - literal character '\'
      anything else - just text
    """
    res = []
    re_attention_gate = re.compile(r"\\([\\()\[\]])|([(])|([)])|(\[)|(\])")
    #                       groups - ----- 1 -----  --2--  --3-- --4-- --5--

    def recursive_parse(subtext, weight):
        matches = list(re_attention_gate.finditer(subtext))
        start_index = 0
        
        # looping through the special characters and the prev start_idx
        for i, match in enumerate(matches):
            text_chunk = subtext[start_index:match.start()]
            if text_chunk:
                res.append([text_chunk, weight])
            
            start_index = match.end()
            escaped, open_paren, close_paren, open_square, close_square = match.groups()

            if escaped: # same as normal text
                res.append([escaped, weight])
            
            elif open_paren:
                # find the matching closing parenthesis
                balance = 1
                end_index = -1
                for j in range(i + 1, len(matches)):
                    if matches[j].group(2): # Another '('
                        balance += 1
                    elif matches[j].group(3): # A ')'
                        balance -= 1
                        if balance == 0:
                            end_index = j
                            break
                
                # if a matching ')' is found, parse the inner content
                if end_index != -1:
                    # check for a weight like (text:1.5)
                    inner_text = subtext[start_index:matches[end_index].start()]
                    new_weight = weight * 1.1
                    
                    weight_match = re.search(r"(.*):([+-]?\d*\.?\d+)\s*$", inner_text)
                    if weight_match:
                        inner_text = weight_match.group(1)
                        new_weight = weight * float(weight_match.group(2))
                    
                    # recursively parse the content inside the parentheses
                    recursive_parse(inner_text, new_weight)
                    start_index = matches[end_index].end()

            elif open_square:
                # similar logic for square brackets, but decrease weight
                balance = 1
                end_index = -1
                for j in range(i + 1, len(matches)):
                    if matches[j].group(4): # Another '['
                        balance += 1
                    elif matches[j].group(5): # A ']'
                        balance -= 1
                        if balance == 0:
                            end_index = j
                            break
                
                if end_index != -1:
                    inner_text = subtext[start_index:matches[end_index].start()]
                    recursive_parse(inner_text, weight / 1.1)
                    start_index = matches[end_index].end()

        # ddd any remaining text after the last special character
        remaining_text = subtext[start_index:]
        if remaining_text:
            res.append([remaining_text, weight])

    recursive_parse(text, 1.0)
    # instead of returning nothing [], return
    # the default valid output
    if not res:
        return [["", 1.0]]
        
    merged = []
    if res:
        merged.append(res[0])
        for i in range(1, len(res)):
            if res[i][1] == merged[-1][1]:
                merged[-1][0] += res[i][0]
            else:
                merged.append(res[i])
                
    return merged


def load_embed(embedding_path: str, embed_key: str = None) -> Tensor | None:
    # loads the textual inversion file
    found_path = ensure_model_availability(model_path=embedding_path)
    embedding_name = os.path.basename(embedding_path)
    
    if not found_path:
        app_logger.warning(f"Embedding '{embedding_name}' not found.")
        return None

    try:
        if found_path.suffix == ".safetensors":
            embed_dict = safetensors.torch.load_file(found_path, device=VRAM_DEVICE) if \
                VRAM_DEVICE != OFFLOAD_DEVICE else safetensors.torch.load_file(found_path)
        else:
            embed_dict = torch.load(found_path, map_location=VRAM_DEVICE, weights_only=True)
    except Exception as e:
        app_logger.error(f"Error loading embedding '{embedding_name}': {e}")
        return None

    if embed_key and embed_key in embed_dict:
        return embed_dict[embed_key]
    elif 'string_to_param' in embed_dict:
        return next(iter(embed_dict['string_to_param'].values()))
    elif embed_dict:
        return next(iter(embed_dict.values()))

    return None

# code directly taken from Huggingface transformers

def convert_sd_to_hf_format(metaclip_state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Convert MetaCLIP state dict to Hugging Face format."""
    
    """
    Converts a CLIP state dictionary from the original OpenAI format to the
    Hugging Face (HF) `transformers` format.

    This function handles both the vision (`visual.*`) and text (`transformer.*`)
    models, which are often stored in the same checkpoint file.

    Key changes include:

    1.  **Prefix Renaming**:
        * `visual.` (vision model) is renamed to `vision_model.`
        * `transformer.` (text model) is renamed to `text_model.encoder.`

    2.  **QKV Splitting**:
        * The original model fuses Query, Key, and Value weights into one
            tensor (e.g., `attn.in_proj_weight`).
        * This function splits that single tensor into three separate
            HF-style tensors: `self_attn.q_proj.weight`,
            `self_attn.k_proj.weight`, and `self_attn.v_proj.weight`.
        * This is done for both weights and biases for both the text and vision models.

    3.  **Layer Renaming**:
        * Renames transformer blocks: `transformer.resblocks.N` -> `encoder.layers.N`
        * Renames layer norms: `ln_1` -> `layer_norm1`, `ln_pre` -> `pre_layrnorm`
        * Renames MLP layers: `mlp.c_fc` -> `mlp.fc1`, `mlp.c_proj` -> `mlp.fc2`

    4.  **Weight Transposing**:
        * The final `visual.proj` and `text_projection` layers are transposed
            to match the weight orientation expected by Hugging Face's
            `torch.nn.Linear` layers.
    """

    hf_state_dict = {}

    for key, value in metaclip_state_dict.items():
        new_key = key

        # Handle specific mappings first before general prefix replacements
        if key == "visual.proj":
            new_key = "visual_projection.weight"
            # Don't transpose! MetaCLIP: x @ proj, HF: Linear(x) = x @ weight.T
            # So we want weight.T = proj, which means weight = proj.T
            # But since we're storing proj as weight, we need proj.T
            value = value.T  # This gives us the correct orientation for Linear layer
        elif key == "text_projection":
            new_key = "text_projection.weight"
            # Same logic as visual projection
            value = value.T
        elif key == "token_embedding.weight":
            new_key = "text_model.embeddings.token_embedding.weight"
        elif key == "positional_embedding":
            new_key = "text_model.embeddings.position_embedding.weight"
        elif key == "ln_final.weight":
            new_key = "text_model.final_layer_norm.weight"
        elif key == "ln_final.bias":
            new_key = "text_model.final_layer_norm.bias"
        # Vision encoder mappings
        elif key.startswith("visual."):
            new_key = key.replace("visual.", "vision_model.")

            # Handle specific vision model components
            if "conv1" in new_key:
                new_key = new_key.replace("conv1", "embeddings.patch_embedding")
            elif "class_embedding" in new_key:
                new_key = new_key.replace("class_embedding", "embeddings.class_embedding")
            elif "positional_embedding" in new_key:
                new_key = new_key.replace("positional_embedding", "embeddings.position_embedding.weight")
            elif "ln_pre" in new_key:
                new_key = new_key.replace("ln_pre", "pre_layrnorm")
            elif "ln_post" in new_key:
                new_key = new_key.replace("ln_post", "post_layernorm")
            elif "transformer.resblocks" in new_key:
                new_key = new_key.replace("transformer.resblocks", "encoder.layers")
                # Handle attention and MLP mappings within transformer blocks
                if "attn.in_proj" in new_key:
                    # Split the in_proj into q, k, v projections
                    if "weight" in new_key:
                        # We'll handle this later in a special case
                        continue
                    elif "bias" in new_key:
                        continue
                elif "attn.out_proj" in new_key:
                    new_key = new_key.replace("attn.out_proj", "self_attn.out_proj")
                elif "ln_1" in new_key:
                    new_key = new_key.replace("ln_1", "layer_norm1")
                elif "ln_2" in new_key:
                    new_key = new_key.replace("ln_2", "layer_norm2")
                elif "mlp.c_fc" in new_key:
                    new_key = new_key.replace("mlp.c_fc", "mlp.fc1")
                elif "mlp.c_proj" in new_key:
                    new_key = new_key.replace("mlp.c_proj", "mlp.fc2")

        # Text encoder mappings
        elif key.startswith("transformer."):
            new_key = key.replace("transformer.", "text_model.encoder.")

            if "resblocks" in new_key:
                new_key = new_key.replace("resblocks", "layers")
                # Similar mappings as vision transformer
                if "attn.in_proj" in new_key:
                    continue  # Handle separately
                elif "attn.out_proj" in new_key:
                    new_key = new_key.replace("attn.out_proj", "self_attn.out_proj")
                elif "ln_1" in new_key:
                    new_key = new_key.replace("ln_1", "layer_norm1")
                elif "ln_2" in new_key:
                    new_key = new_key.replace("ln_2", "layer_norm2")
                elif "mlp.c_fc" in new_key:
                    new_key = new_key.replace("mlp.c_fc", "mlp.fc1")
                elif "mlp.c_proj" in new_key:
                    new_key = new_key.replace("mlp.c_proj", "mlp.fc2")

        hf_state_dict[new_key] = value

    # Handle in_proj weights separately (split into q, k, v)
    for key, value in metaclip_state_dict.items():
        if "attn.in_proj_weight" in key:
            # Split the combined qkv weight into separate q, k, v weights
            dim = value.shape[0] // 3
            q_weight = value[:dim]
            k_weight = value[dim : 2 * dim]
            v_weight = value[2 * dim :]

            base_key = key.replace("attn.in_proj_weight", "")
            if key.startswith("visual."):
                base_key = base_key.replace("visual.transformer.resblocks", "vision_model.encoder.layers")
            else:
                base_key = base_key.replace("transformer.resblocks", "text_model.encoder.layers")

            hf_state_dict[f"{base_key}self_attn.q_proj.weight"] = q_weight
            hf_state_dict[f"{base_key}self_attn.k_proj.weight"] = k_weight
            hf_state_dict[f"{base_key}self_attn.v_proj.weight"] = v_weight

        elif "attn.in_proj_bias" in key:
            # Split the combined qkv bias into separate q, k, v biases
            dim = value.shape[0] // 3
            q_bias = value[:dim]
            k_bias = value[dim : 2 * dim]
            v_bias = value[2 * dim :]

            base_key = key.replace("attn.in_proj_bias", "")
            if key.startswith("visual."):
                base_key = base_key.replace("visual.transformer.resblocks", "vision_model.encoder.layers")
            else:
                base_key = base_key.replace("transformer.resblocks", "text_model.encoder.layers")

            hf_state_dict[f"{base_key}self_attn.q_proj.bias"] = q_bias
            hf_state_dict[f"{base_key}self_attn.k_proj.bias"] = k_bias
            hf_state_dict[f"{base_key}self_attn.v_proj.bias"] = v_bias

    return hf_state_dict
