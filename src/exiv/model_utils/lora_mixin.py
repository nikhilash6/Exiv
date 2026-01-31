import torch

import os
import mmap
from typing import Any, List, Optional, Union
import struct, json
from dataclasses import dataclass

from tqdm import tqdm

from ..utils.device import OFFLOAD_DEVICE, VRAM_DEVICE
from ..utils.logging import app_logger


CACHED_MODEL_LORA_KEY_MAP = "_cached_m_l_map"
LORA_WEIGHTS_CACHE_DICT = "lora_weights_cache_dict"

@dataclass
class LoraDefinition:
    path: Optional[str] = None
    base_strength: Union[float, List[float]] = 1.0  # ideally this should be the complete stepwise strength
                                                    # expand_strength_schedule is a safety measure of sorts
    
    @classmethod
    def from_json(cls, data):
        try:
            return cls(**data)
        except Exception as e:
            app_logger.error(f"Failed to load lora definition from json: {data}")
            raise e
    
    @property
    def name(self):
        return os.path.splitext(os.path.basename(self.path))[0]
    
    def get_output(self, input, w_up, w_down, scale, timestep):
        # NOTE: this may not work properly if conv layers in lora have different kernel size
        # TODO: rn this is used as a global method for all kinds of lora types, this will change
        #       as more lora types are added
        if (strength:=self.base_strength[timestep]) == 0: return 0
        
        # (Batch, Tokens, Dim) @ (Dim, Rank) -> (Batch, Tokens, Rank)
        down_weight = w_down.flatten(start_dim=1).T.to(input.dtype)  # Shape: [In, Rank]
        mid_step = input @ down_weight 
        # (Batch, Tokens, Rank) @ (Rank, Out) -> (Batch, Tokens, Out)
        up_weight = w_up.flatten(start_dim=1).T.to(input.dtype)      # Shape: [Rank, Out]
        final_out = mid_step @ up_weight
        
        return final_out * scale * strength
    
    def _uniform_stretch(self, strength_input: List, total_steps):
        # List -> expand, [1, 2, 3] -> [1, 1, 1, 2, 2, 2, 3, 3, 3]
        src_len = len(strength_input)
        new_schedule = []
        inc = src_len / total_steps
        pos = 0.0
        
        for _ in range(total_steps):
            idx = int(pos)
            if idx >= src_len: idx = src_len - 1
            
            new_schedule.append(strength_input[idx])
            pos += inc
        return new_schedule
    
    def _end_stretch(self, strength_input: List, total_steps):
        # [1, 2, 3] -> [1, 2, 3, 3, 3, 3, 3]
        for _ in range(total_steps - len(strength_input)):
            strength_input.append(strength_input[-1])
        return strength_input
    
    def expand_strength_schedule(self, total_steps: int):
        strength_input = self.base_strength
        if isinstance(strength_input, (float, int)):
            # single float -> constant schedule
            self.base_strength = [float(strength_input)] * total_steps
        else:
            strength_input = strength_input[:total_steps]
            self.base_strength = self._end_stretch(strength_input, total_steps)

# TODO: support multi batch / per frame lora (like frame_1 can have lora_1 applied, frame_2 has lora_2 ..)
#       - this can be done by passing lora indices to prepare_loras_for_inference (will require some other modifications as well)
class LoraMixin:
    def __init__(self):
        self.lora_definitions: List[LoraDefinition] = [] 
        self.active_lora_schedule = {}
        self.mmap_cache = {} # {path: {'mm': mmap, 'header': dict, 'offset': int}}
    
    def create_model_lora_key_map(self, state_dict, key_map=None):
        """
        Creates a map of lora_key <-> model_key and vice versa in the same dict.
        Only returns the 'down' key for the LoRA.
        """
        # NOTE: needs to be overidden in the child class, this is a generic impl.
        # set this to a null method to disable lora loading
        if key_map is None:
            sd_keys = state_dict.keys()
            key_map = {}
            
            for k in sd_keys:
                if k.endswith(".weight"):
                    key_lora = k[len("diffusion_model."):-len(".weight")].replace(".", "_")
                    key_map["lora_unet_{}".format(key_lora)] = k
                    key_map["{}".format(k[:-len(".weight")])] = k
                else:
                    key_map["{}".format(k)] = k
                
        setattr(self, CACHED_MODEL_LORA_KEY_MAP, key_map)
    
    def get_mapped_lora_key(self, key = None):
        """
        If model_key is provided then the corresponding lora_key is returned and vice-versa
        """
        if not hasattr(self, CACHED_MODEL_LORA_KEY_MAP): return None    # set by create_model_lora_key_map
        d = getattr(self, CACHED_MODEL_LORA_KEY_MAP)
        return d.get(key, None)

    def add_lora(self, lora_def: LoraDefinition):
        self.lora_definitions.append(lora_def)
    
    def prepare_loras_for_inference(self, total_steps, device=VRAM_DEVICE):
        setattr(self, LORA_WEIGHTS_CACHE_DICT, {})      # reset cache dict
        cache = getattr(self, LORA_WEIGHTS_CACHE_DICT)
        if self.lora_definitions: app_logger.info("Loading LoRAs...")
        for lora in self.lora_definitions:
            lora.expand_strength_schedule(total_steps + 1)
            for name, layer in tqdm(self.named_modules(), desc=f"Loading {lora.name}", leave=False):
                model_key = f"{name}.weight"
                lora_down_key = self.get_mapped_lora_key(model_key)
                if not lora_down_key: continue
                up_key = lora_down_key.replace("down", "up")
                out = self.get_delta_from_mmap(lora, lora_down_key, up_key, device=device)
                # loading everythign irrespective of their current strength
                if out is not None:
                    out += (lora,)
                    # out -> (w_up, w_down, scale)
                    if name not in cache: cache[name] = [out]
                    else: cache[name].append(out)
                    
        setattr(self, LORA_WEIGHTS_CACHE_DICT, cache)
              
    def _ensure_mmap(self, path):
        if path in self.mmap_cache: return self.mmap_cache[path]
        
        with open(path, "rb") as f:
            # Parse Safetensors Header
            header_size = struct.unpack('<Q', f.read(8))[0]
            header_json = f.read(header_size)
            header = json.loads(header_json)
            
            # Map entire file
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            
            self.mmap_cache[path] = {
                'mm': mm,
                'header': header,
                'data_start': 8 + header_size
            }
        return self.mmap_cache[path]

    def _read_from_mmap(self, path, key, device, dtype):
        cache = self._ensure_mmap(path)
        header, mm, base = cache['header'], cache['mm'], cache['data_start']
        
        if key not in header:
            if "diffusion_model." + key in header: key = "diffusion_model." + key
            else: return None
        
        info = header[key]
        start, end = info['data_offsets']
        shape = info['shape']
        
        # map string dtype to torch dtype
        dt_map = {'F16': torch.float16, 'F32': torch.float32, 'BF16': torch.bfloat16, \
            'F64': torch.float64, 'I64': torch.int64, 'I32': torch.int32}
        src_dtype = dt_map.get(info['dtype'], torch.float32)
        
        # zero-copy view from disk to cpu ram
        data_view = memoryview(mm)[base + start : base + end]
        tensor = torch.frombuffer(data_view, dtype=src_dtype).reshape(shape)
        
        # copy to the target device/dtype (this is the only mem consuming step)
        return tensor.to(device=device, dtype=dtype)

    def _read_weight(self, lora_def: LoraDefinition, key: str, device: str, dtype):
        return self._read_from_mmap(lora_def.path, key, device, dtype)
    
    def get_delta_from_mmap(self, lora_def: LoraDefinition, down_key, up_key, device=OFFLOAD_DEVICE) -> Optional[tuple]:        
        dtype=torch.float16     # NOTE: hardcoded for now, need to change
        out = None              # (w_up, w_down, scale)
        # TODO: add more custom keys as they are added
        if "diff" in down_key:
            weight = self._read_weight(lora_def, down_key, device, dtype)
            if weight is None: out = None
            out = (weight, 1.0, 1.0)
        else:
            w_down = self._read_weight(lora_def, down_key, device, dtype)
            w_up = self._read_weight(lora_def, up_key, device, dtype)
            if w_down is None or w_up is None: out = None
            else:
                alpha = self._get_alpha(down_key, lora_def) or w_down.shape[0]
                scale = (alpha / w_down.shape[0])
                out = (w_up, w_down, scale)
        return out

    def _get_alpha(self, down_key, lora_def):
        # calculate alpha key based on down key format (will fix later)
        alpha_key = None
        if ".lora_down.weight" in down_key:
            alpha_key = down_key.replace(".lora_down.weight", ".alpha")
        elif down_key.endswith(".down"):
            alpha_key = down_key.replace(".down", ".alpha")
        if alpha_key is None: return None
        alpha_tensor = self._read_weight(lora_def, alpha_key, "cpu", torch.float32)
        if alpha_tensor is None:
            alpha_key = down_key.split(".lora")[0] + ".alpha"
            alpha_tensor = self._read_weight(lora_def, alpha_key, "cpu", torch.float32)
        return alpha_tensor.item() if alpha_tensor is not None else None

############ DEV methods, not in use #############

def compress_weight(weight, target_rank, device):
    new_lora = {}
    
    for key, dense_tensor in weight.items():
        if "lora" not in key: continue 
        
        # returns small matrices: Up (Out, Rank) and Down (Rank, In)
        w_up, w_down = svd_compress(dense_tensor, target_rank, device)
        
        # 'key' rn -> "lora...down.weight" string
        new_down_key = key
        new_up_key = key.replace("down", "up")
        new_alpha_key = key.split(".")[0] + ".alpha"
        
        # handle Conv2d and other bigger shapes
        orig_shape = dense_tensor.shape
        
        # [Rank, In, K, K...]
        w_down = w_down.reshape(target_rank, *orig_shape[1:])

        # [Out, Rank, 1, 1...]
        spatial_dims = len(orig_shape) - 2
        w_up = w_up.reshape(w_up.shape[0], target_rank, *([1] * spatial_dims))

        new_lora[new_down_key] = w_down
        new_lora[new_up_key] = w_up
        new_lora[new_alpha_key] = torch.tensor(target_rank, dtype=w_up.dtype)

    return new_lora

def svd_compress(dense_weight, target_rank, device):
    original_shape = dense_weight.shape
    
    # flattening trick
    if len(original_shape) > 2:
        dense_weight = dense_weight.flatten(start_dim=1)
    
    # main svd
    u, s, vh = torch.linalg.svd(dense_weight.to(device), full_matrices=False)
    
    # truncate to target rank
    u = u[:, :target_rank]
    s = s[:target_rank]
    vh = vh[:target_rank, :]
    
    # svd = u * sigma * vT => w_up = u * sqrt(sigma) , w_down = sqrt(sigma) * vT
    s_sqrt = torch.diag(torch.sqrt(s))
    lora_up = torch.mm(u, s_sqrt)
    lora_down = torch.mm(s_sqrt, vh)
    
    return lora_up.cpu(), lora_down.cpu()