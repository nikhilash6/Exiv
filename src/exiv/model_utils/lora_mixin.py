import torch

import os
import mmap
from typing import List, Union
import struct, json

from ..utils.logging import app_logger


# TODO: incorporate disable_mmap from the global_config
class LoraMixin:
    def __init__(self):
        self.lora_definitions = [] 
        self.active_lora_schedule = {}
        self.mmap_cache = {} # {path: {'mm': mmap, 'header': dict, 'offset': int}}
        
    def get_key_map(self, model_key = None, lora_key = None):
        """
        If model_key is provided then the corresponding lora_key is returned and vice-versa
        """
        if model_key is not None:
            print("process stuff")
        elif lora_key is not None:
            print("process more stuff")
        else:
            raise Exception("atleast one key param is required")

    def add_lora(self, lora_path: str, base_strength: float = 1.0):
        self.lora_definitions.append({
            "name": os.path.splitext(os.path.basename(lora_path))[0],
            "path": lora_path,
            "base_strength": base_strength,
        })

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
        
        if key not in header: return None
        
        info = header[key]
        start, end = info['data_offsets']
        shape = info['shape']
        
        # map string dtype to torch dtype
        dt_map = {'F16': torch.float16, 'F32': torch.float32, 'BF16': torch.bfloat16}
        src_dtype = dt_map.get(info['dtype'], torch.float32)
        
        # zero-copy view from disk to cpu ram
        data_view = memoryview(mm)[base + start : base + end]
        tensor = torch.frombuffer(data_view, dtype=src_dtype).reshape(shape)
        
        # copy to the target device/dtype (this is the only mem consuming step)
        return tensor.to(device=device, dtype=dtype)

    def get_delta_from_mmap(self, lora_idx, down_key, up_key, current_scale, device="cuda", dtype=torch.float16):
        lora_def = self.lora_definitions[lora_idx]
        path = lora_def["path"]
        
        # TODO: add more custom keys as they are added
        if "diff" in down_key:
            weight = self._read_from_mmap(path, down_key, device, dtype)
            if weight is None: return None
            return weight * current_scale
        
        w_down = self._read_from_mmap(path, down_key, device, dtype)
        w_up = self._read_from_mmap(path, up_key, device, dtype)
        
        if w_down is None or w_up is None: return None

        # calculate alpha
        alpha_key = down_key.replace(".down", ".alpha")
        alpha_tensor = self._read_from_mmap(path, alpha_key, "cpu", torch.float32)
        if alpha_tensor is None:
            alpha_key = down_key.split(".lora")[0] + ".alpha"
            alpha_tensor = self._read_from_mmap(path, alpha_key, "cpu", torch.float32)
        alpha = alpha_tensor.item() if alpha_tensor is not None else w_down.shape[0]
        scale = (alpha / w_down.shape[0]) * current_scale

        # flattening trick: (Out, Rank) @ (Rank, In) -> (Out, In)
        op = torch.mm(w_up.flatten(start_dim=1), w_down.flatten(start_dim=1))
        
        target_shape = [w_up.shape[0]] + list(w_down.shape[1:])
        return op.reshape(target_shape) * scale
    
    def get_combined_delta(self, model_key, timestep, target_device, target_dtype):
        """
        Iterates all active LoRAs, fetches their mmap deltas, and sums them according
        to their current timestep strength
        """
        if timestep < 0: timestep += len(self.active_lora_schedule)
        if timestep >= len(self.active_lora_schedule): return None
        
        total_delta = None
        for lora_idx, strength in self.active_lora_schedule[timestep]:
            lora_down_key = self.get_key_map(model_key)
            lora_up_key = lora_down_key.replace("down", "up")
            
            delta = self.get_delta_from_mmap(
                lora_idx=lora_idx,
                down_key=lora_down_key,
                up_key=lora_up_key,
                current_scale=strength,
                device=target_device,
                dtype=target_dtype
            )
            
            if delta is not None:
                if total_delta is None:
                    total_delta = delta
                else:
                    total_delta.add_(delta) # In-place add for speed
                    
        return total_delta
        
    def _expand_schedule(self, strength_input: Union[float, List[float]], total_steps: int) -> List[float]:
        # single float -> constant schedule
        if isinstance(strength_input, (float, int)):
            return [float(strength_input)] * total_steps
        
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

    def setup_lora_schedule(self, total_steps: int, fuse_const_loras: bool = False):
        """
        Sets up the schedule, basically what lora needs to be applied at what step and with what strength
        
        fuse_const_loras is an experimental feature, it can be useful if we have 100s of loras applied, we
        can use 2-3 mins to fuse them first but if we are only using 2-3 loras then this doesn't make sense
        """
        self.active_lora_schedule = {}
        
        temp_schedules = []
        for lora in self.lora_definitions:
            temp_schedules.append(self._expand_schedule(lora["base_strength"], total_steps))

        # identifying constant vs dynamic strength loras
        constant_indices = []
        dynamic_definitions = []
        dynamic_schedules = []
        
        for i, sched in enumerate(temp_schedules):
            if len(sched) > 0 and (max(sched) == min(sched)) and fuse_const_loras:
                if sched[0] != 0: # Ignore 0-strength LoRAs
                    constant_indices.append(i)
            else:
                dynamic_definitions.append(self.lora_definitions[i])
                dynamic_schedules.append(sched)

        # fusing constant loras
        if len(constant_indices):
            app_logger.info(f"Fusing {len(constant_indices)} constant LoRAs...")
            
            to_fuse = []
            for idx in constant_indices:
                lora = self.lora_definitions[idx]
                strength = temp_schedules[idx][0]
                to_fuse.append((lora, strength))
            
            fused_lora_def = self._merge_constant_loras(to_fuse)
            dynamic_definitions.insert(0, fused_lora_def)
            dynamic_schedules.insert(0, [1.0] * total_steps)
        
        self.lora_definitions = dynamic_definitions
        
        for step in range(total_steps):
            self.active_lora_schedule[step] = [
                (idx, strength[step] if step < len(strength) else strength[-1])
                for idx, strength in enumerate(dynamic_schedules)
            ]
        
        self.current_time_step = -1

    def _merge_constant_loras(self, lora_list):
        # merges constant loras directly into the base model
        merged_weights = {}
        device = self.gpu_device
        dtype = self.dtype
        rank_sum = 0
        
        down_patterns = ["lora.down", "lora_down"]

        with torch.no_grad():
            for lora_def in lora_list:
                weights = lora_def["weights"]
                base_strength = lora_def["base_strength"]
                
                processed_keys = set()
                max_rank = 16
                
                for key in weights.keys():
                    if key in processed_keys: continue
                    
                    # --- Case 1: diff (Simple Addition) ---
                    if ".diff" in key or ".diff_b" in key:
                        w = weights[key].to(device, dtype=dtype)
                        delta = w * base_strength
                        
                        if key not in merged_weights:
                            merged_weights[key] = delta.cpu()
                        else:
                            merged_weights[key].add_(delta.cpu())
                            
                        processed_keys.add(key)
                        del w, delta

                    # --- Case 2: LoRA A/B pairs (Matmul + Scale) ---
                    elif any(p in key for p in down_patterns):
                        down_key = key
                        
                        # finding the Up key
                        up_key = None
                        for p in down_patterns:
                            if p in down_key:
                                up_p = p.replace("down", "up")
                                candidate = down_key.replace(p, up_p)
                                if candidate in weights:
                                    up_key = candidate
                                    break
                        
                        if up_key:
                            # load weights
                            w_down = weights[down_key].to(device, dtype=dtype)
                            w_up = weights[up_key].to(device, dtype=dtype)
                            
                            # calculate dynamic scale (Alpha / Rank)
                            alpha_key = down_key.split(".")[0] + ".alpha" # Heuristic 1 (root)
                            if alpha_key not in weights:
                                alpha_key = down_key.replace(".weight", "") + ".alpha" # Heuristic 2 (layer)

                            rank = w_down.shape[0]
                            max_rank = max(max_rank, rank)
                            alpha = weights[alpha_key].item() if alpha_key in weights else rank
                            scale = alpha / rank
                            
                            eff_strength = base_strength * scale

                            # mat mul (flattening trick from lycoris)
                            # flatten Up to (Out, Rank)
                            # flatten Down to (Rank, In * K * K...)
                            op = torch.mm(
                                w_up.flatten(start_dim=1), 
                                w_down.flatten(start_dim=1)
                            )
                            
                            # reshape back to target dimensions
                            # target shape = [Out] + [In, K, K...]
                            target_shape = [w_up.shape[0]] + list(w_down.shape[1:])
                            delta = op.reshape(target_shape)

                            # apply strength
                            delta = delta * eff_strength
                            
                            # accumulate on cpu
                            if key not in merged_weights:
                                merged_weights[key] = delta.cpu()
                            else:
                                merged_weights[key].add_(delta.cpu())
                            
                            processed_keys.add(down_key)
                            processed_keys.add(up_key)
                            del w_down, w_up, delta

                        else:
                            app_logger.warning(f"Skipping {down_key} during LoRA merging as no complimentary up_key is found")
                            
                rank_sum += max_rank    

            rank_sum = max(128, rank_sum)
            new_lora = compress_and_save(merged_weights, rank_sum, self.device)

        return {
            "name": "fused_constant_loras",
            "weights": new_lora, 
            "base_strength": 1.0,
        }

    def cleanup(self):
        for v in self.mmap_cache.values():
            v['mm'].close()
        self.mmap_cache.clear()

def compress_and_save(merged_weights, target_rank, device):
    new_lora = {}
    
    for key, dense_tensor in merged_weights.items():
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