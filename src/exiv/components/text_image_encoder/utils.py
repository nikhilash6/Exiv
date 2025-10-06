from pathlib import Path
import torch
from torch import Tensor

import re
import os
import safetensors

from ...utils.device import DEFAULT_DEVICE, ProcDevice
from ...utils.file import ensure_model_available
from ...utils.logging import app_logger

# code adapted from ForgeUI

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
    found_path = ensure_model_available(embedding_path)
    embedding_name = os.path.basename(embedding_path)
    
    if not found_path:
        app_logger.warning(f"Embedding '{embedding_name}' not found.")
        return None

    try:
        if found_path.suffix == ".safetensors":
            embed_dict = safetensors.torch.load_file(found_path, device=DEFAULT_DEVICE) if \
                DEFAULT_DEVICE != ProcDevice.CPU.value else safetensors.torch.load_file(found_path)
        else:
            embed_dict = torch.load(found_path, map_location=DEFAULT_DEVICE, weights_only=True)
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