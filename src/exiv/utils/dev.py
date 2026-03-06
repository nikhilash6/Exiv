# ad-hoc methods used during development
# NOTE: these are strictly for dev use only

import json
import subprocess
import os
import torch
from torch import Tensor

import gc
import sys
import time
import warnings
import logging

from .logging import app_logger
from ..config import global_config

class MemoryMonitor:
    """Holds the checkpoint time for the memory usage printer."""
    last_call_time = time.perf_counter()

def format_bytes(size_bytes):
    if size_bytes == 0:
        return "0B"
    units = ("B", "KB", "MB", "GB", "TB")
    i = 0
    while size_bytes >= 1024 and i < len(units) - 1:
        size_bytes /= 1024
        i += 1
    return f"{size_bytes:.2f}{units[i]}"

def print_memory_usage(tag, n=5, show_all=False):
    # if global_config.logging_level != logging.DEBUG: return
    
    app_logger.info(f"----------------------------- START: {tag} ----------------------------")
    
    # --- Timestamp ---
    current_time = time.perf_counter()
    delta = current_time - getattr(MemoryMonitor, 'last_call_time', current_time)
    print(f"--- Checkpoint (Time Since Last Call: {delta:.2f}s) ---")
    MemoryMonitor.last_call_time = current_time 

    # --- General CPU RAM ---
    if show_all:
        import psutil
        print("\n--- CPU Memory (psutil) ---")
        mem_info = psutil.virtual_memory()
        print(f"Total: {format_bytes(mem_info.total)}")
        print(f"Used:  {format_bytes(mem_info.used)} ({mem_info.percent}%)")
        print(f"Free:  {format_bytes(mem_info.available)}")

    # --- GPU VRAM (PyTorch) ---
    if torch.cuda.is_available():
        print("\n--- GPU Memory (PyTorch) ---")
        torch.cuda.empty_cache()
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        print(f"Allocated: {format_bytes(allocated)}")
        print(f"Reserved:  {format_bytes(reserved)} (Total Capacity)")
    
    # --- Top N Objects ---
    target_str = "CPU & GPU" if show_all else "GPU Only"
    print(f"\n--- Top {n} Largest Objects ({target_str}) ---")
    
    gc.collect()
    all_objects = []
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        for obj in gc.get_objects():
            size_bytes = 0
            desc = ""
            is_gpu_object = False

            try:
                if isinstance(obj, torch.Tensor):
                    if obj.device.type != 'cpu':
                        is_gpu_object = True
                    
                    if obj.device.type == 'meta':
                        size_bytes = 0
                    else:
                        size_bytes = obj.nelement() * obj.element_size()
                    
                    desc = f"Tensor({list(obj.shape)}, {obj.dtype}, {obj.device})"
                
                elif "UntypedStorage" in str(type(obj)):
                    # Check if storage is on CUDA
                    device_str = "cpu"
                    try:
                        device_str = obj.device.type
                    except AttributeError:
                        pass
                    
                    if device_str != 'cpu':
                        is_gpu_object = True

                    size_bytes = obj.size()
                    desc = f"Storage(size={obj.size()}, device={device_str})"
                
                else:
                    # Python objects are CPU bound
                    size_bytes = sys.getsizeof(obj)
                    desc = f"Object({type(obj).__name__})"

                # --- FILTER LOGIC ---
                # If we only want GPU stuff, skip if it's not a GPU object
                if not show_all and not is_gpu_object:
                    continue

                if size_bytes > 0:
                    all_objects.append((size_bytes, desc))
            
            except Exception:
                pass

    all_objects.sort(key=lambda x: x[0], reverse=True)
    
    if not all_objects:
        print("No objects found.")
        return

    for i, (size_bytes, desc) in enumerate(all_objects[:n]):
        print(f"{i+1}. {format_bytes(size_bytes)} - {desc}")
        
    app_logger.info(f"----------------------------- END: {tag} ----------------------------")

def print_zombie(model_type_name):
    # need this pkg to generate the graph image - apt-get install graphviz
    import objgraph
    zombie_models = objgraph.by_type(model_type_name)
    if not zombie_models:
        print("--- SUCCESS: No zombie models found in memory. ---")
    else:
        print(f"--- LEAK FOUND: {len(zombie_models)} zombie '{model_type_name}' instances still in memory.")
        
        print("Generating 'zombie_model_leak.png'...")
        objgraph.show_backrefs(
            [zombie_models[0]], 
            filename='zombie_model_leak.png', 
            max_depth=10
        )
        print("--- LEAK GRAPH SAVED to 'zombie_model_leak.png'. Open this file to see the leak. ---")

def find_who_is_holding_class(target_class_name: str):
    """
    Scans memory for ANY instance of a class named `target_class_name` 
    and reports who is holding it.
    """
    import gc
    import inspect
    from types import ModuleType, FrameType, CellType, FunctionType
    
    print(f"\n🔎 Scanning memory for instances of '{target_class_name}'...")
    
    # 1. Find all instances of this class currently alive in memory
    # We use a generator to avoid creating a new list that holds strong references
    found_objs = [obj for obj in gc.get_objects() if type(obj).__name__ == target_class_name]
    
    if not found_objs:
        print(f"✅ Clean! No instances of '{target_class_name}' found in memory.\n")
        return

    print(f"⚠️  Found {len(found_objs)} instance(s) of '{target_class_name}'! Analyzing references...\n")

    # 2. Analyze who is holding each instance
    for i, obj in enumerate(found_objs):
        obj_id = id(obj)
        print(f"--- [Instance #{i+1}] {target_class_name} @ {hex(obj_id)} ---")
        
        referrers = gc.get_referrers(obj)
        found_refs = 0
        
        for ref in referrers:
            # Ignore the 'found_objs' list itself (Observer Effect)
            if ref is found_objs:
                continue
            # Ignore current stack frame
            if inspect.isframe(ref):
                continue
                
            found_refs += 1
            
            # --- CASE A: Dictionary ---
            if isinstance(ref, dict):
                if "__name__" in ref and "__doc__" in ref and "__package__" in ref:
                    print(f"👉 [GLOBAL SCOPE] Module '{ref.get('__name__')}'")
                elif "self" in ref or "model" in ref:
                     print(f"👉 [LOCALS] Function locals (Keys: {list(ref.keys())[:5]}...)")
                else:
                    # Check if it's an object's __dict__
                    owners = [x for x in gc.get_referrers(ref) if hasattr(x, "__dict__") and x.__dict__ is ref]
                    if owners:
                        print(f"👉 [ATTRIBUTE] Inside object of type '{type(owners[0]).__name__}'")
                    else:
                        print(f"👉 [DICT] Raw dictionary (Keys: {list(ref.keys())[:3]}...)")

            # --- CASE B: Closure Cell (The Zombie Maker) ---
            elif isinstance(ref, CellType):
                print(f"👉 [CLOSURE CELL] Captured inside a function closure.")
                cell_refs = gc.get_referrers(ref)
                funcs = [f for f in cell_refs if inspect.isfunction(f)]
                if funcs:
                    print(f"    ↳ CAPTURED BY FUNCTION: '{funcs[0].__name__}'")

            # --- CASE C: Bound Method ---
            elif inspect.ismethod(ref):
                 print(f"👉 [BOUND METHOD] '{ref.__name__}' (holds instance strongly)")
            
            # --- CASE D: Containers ---
            elif isinstance(ref, (list, tuple)):
                print(f"👉 [CONTAINER] Inside a {type(ref).__name__} of length {len(ref)}")

            # --- CASE E: Custom Objects ---
            elif hasattr(ref, "__class__"):
                print(f"👉 [OBJECT] Property of '{type(ref).__name__}' object")
            
            else:
                print(f"👉 [UNKNOWN] {type(ref)}")

        print(f"--- Found {found_refs} external references ---\n")

def print_tensor_size(t: Tensor):
    num_elements = t.numel()
    element_size = t.element_size()  # Size in bytes (e.g., float32 is 4)

    total_bytes = num_elements * element_size
    total_mb = total_bytes / (1024 * 1024)

    print(f"Tensor VRAM: {total_mb:.2f} MB")
    print(f"Tensor VRAM: {total_bytes:.2f} B")

def print_model_params(model: torch.nn.Module, break_dtype=None):
    # break_dtype - loop breaks if this dtype is encountered
    for name, param in model.named_parameters():
        print(f"Found parameter: {name}")
        print(f"Dtype: {param.dtype}")
        print(f"Device: {param.device}")
        
        if param.dtype == break_dtype:
            break

import torch
from torch import Tensor

def get_covariance_matrix(tensor, channel_dim):
    dims = list(range(tensor.ndim))
    dims.remove(channel_dim)
    dims.append(channel_dim)
    
    t_permuted = tensor.permute(*dims)
    t_flat = t_permuted.reshape(-1, tensor.shape[channel_dim]).float()
    
    t_centered = t_flat - t_flat.mean(dim=0)
    cov_matrix = (t_centered.T @ t_centered) / (t_centered.shape[0] - 1)
    
    return cov_matrix

def calc_norm(cov1, cov2):
    dist = torch.linalg.matrix_norm(cov1 - cov2, ord='fro').item()    # frobenius norm
    status = "Good (Matches)" if dist / cov1.shape[0] < 0.01 else "Bad (Mismatch)"
    print(status)
    return dist

def get_spatial_mse(tensor_a, tensor_b, threshold=0.01):
    import torch
    import torch.nn.functional as F
    mse = F.mse_loss(tensor_a.float(), tensor_b.float()).item()
    status = "Good (Spatial layout matches)" if mse < threshold else "Bad (Spatial layout mismatch)"
    print(f"Spatial MSE: {mse:.4f} -> {status}")
    return mse

def generate_spark_lines(t, bins=20):
    # Flatten and take a meaningful sample (e.g., the first 1000 values)
    data = t.flatten()[:1000].cpu().float().numpy()

    # Normalize to 0-7 for Unicode bars
    d_min, d_max = data.min(), data.max()
    if d_max - d_min < 1e-7:
        print("Tensor is uniform.")
        return

    normalized = ((data - d_min) / (d_max - d_min) * 7).astype(int)
    chars = " ▂▃▄▅▆▇█"
    sparkline = "".join([chars[i] for i in normalized])

    # Print in chunks of 100
    print("Visual Pattern (First 1000 elements):")
    for i in range(0, 1000, 100):
        print(sparkline[i:i+100])

def visualize_latents_pca(tensor):
    from PIL import Image
    import numpy as np
    if tensor.ndim == 3: tensor = tensor.unsqueeze(0)
    elif tensor.ndim == 4: pass 
    elif tensor.ndim == 5: 
        b, c, t, h, w = tensor.shape
        tensor = tensor.permute(0, 2, 1, 3, 4).reshape(-1, c, h, w)
    else: raise ValueError(f"Unsupported shape: {tensor.shape}")

    n, c, h, w = tensor.shape
    tensor = tensor.float().cpu() 
    
    flat_tensor = tensor.permute(0, 2, 3, 1).reshape(-1, c)
    num_samples = min(flat_tensor.size(0), 5000)
    indices = torch.randperm(flat_tensor.size(0))[:num_samples]
    sampled_data = flat_tensor[indices]
    
    mean = sampled_data.mean(dim=0)
    centered = sampled_data - mean
    try:
        U, S, V = torch.svd(centered)
        components = V[:, :3] 
        rgb_flat = (flat_tensor - mean) @ components
        rgb_tensor = rgb_flat.reshape(n, h, w, 3).permute(0, 3, 1, 2)
    except Exception as e:
        print(f"PCA failed: {e}. Falling back to first 3 channels.")
        padding = torch.zeros(n, max(0, 3-c), h, w)
        rgb_tensor = torch.cat([tensor[:, :3, :, :], padding], dim=1)[:, :3, :, :]

    rgb_images = []
    for i in range(n):
        img = rgb_tensor[i]
        flat_img = img.flatten()
        low = torch.quantile(flat_img, 0.01)
        high = torch.quantile(flat_img, 0.99)
        if (high - low) > 1e-5:
            img = (img - low) / (high - low)
        img = img.clamp(0, 1)
        img_np = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        rgb_images.append(Image.fromarray(img_np))

    return rgb_images

# --- 2. Updated Grid Helper Function with Styled Borders ---
def create_styled_image_grid(images, black_gap=3, white_border=1):
    from PIL import Image, ImageDraw
    import math
    """
    Arranges images into a grid separated by black lines with a white center frame.
    
    Args:
        images: List of PIL images.
        black_gap: Thickness of black padding around the white frame.
        white_border: Thickness of white frame immediately surrounding the image.
    """
    num_frames = len(images)
    if num_frames == 0: return None

    n = math.ceil(math.sqrt(num_frames))
    w_frame, h_frame = images[0].size

    # Calculate offsets
    total_pad = black_gap + white_border
    # The full size of one "cell" in the grid including its borders
    stride_w = w_frame + (total_pad * 2)
    stride_h = h_frame + (total_pad * 2)

    # 1. Create Base Canvas: Solid Black
    canvas_w = n * stride_w
    canvas_h = n * stride_h
    grid_img = Image.new('RGB', (canvas_w, canvas_h), color=(0, 0, 0))
    draw = ImageDraw.Draw(grid_img)

    for i, img in enumerate(images):
        row = i // n
        col = i % n

        # Top-left coordinate for this grid cell
        cell_x = col * stride_w
        cell_y = row * stride_h

        # 2. Draw White Rectangle (Layer 2)
        # It starts after the black gap and ends before the next black gap
        white_rect_coords = (
            cell_x + black_gap,
            cell_y + black_gap,
            cell_x + stride_w - black_gap - 1, # -1 required for PIL rectangle math
            cell_y + stride_h - black_gap - 1
        )
        draw.rectangle(white_rect_coords, fill=(255, 255, 255))

        # 3. Paste Image (Layer 3)
        # Centered inside the white rectangle
        img_x = cell_x + total_pad
        img_y = cell_y + total_pad
        grid_img.paste(img, (img_x, img_y))

    return grid_img

import hashlib
def get_tensor_hash(t, visualize_latent=False):
    import numpy as np
    # Fix: cast to float() immediately to avoid bf16 numpy errors
    t = t.detach().float().cpu()
    data = t.contiguous().numpy()
    hash_str = hashlib.sha256(data.tobytes()).hexdigest()[:8]
    
    msg = f"[Probe] | Shape: {tuple(t.shape)} | Mean: {t.mean():.4f} | Std: {t.std():.4f} | Min: {t.min():.4f} | Max: {t.max():.4f}"
    
    # Add Motion Stats if it's a Video Latent (Batch, C, Time, H, W)
    if t.ndim == 5 and t.shape[2] > 1:
        # 1. Frame-to-Frame Delta (Velocity)
        # Compare frame T with T-1
        diffs = t[:, :, 1:] - t[:, :, :-1]
        motion_score = diffs.abs().mean()
        
        # 2. Temporal Standard Deviation
        # Calculate std deviation across the Time dimension (dim 2)
        temp_std = torch.std(t, dim=2).mean()
        
        msg += f" | Motion: {motion_score:.4f} | T_Std: {temp_std:.4f}"
        
    msg += f" | Hash: {hash_str}"
    print(msg)
    
    if visualize_latent:
        try:
            frames_list = visualize_latents_pca(t)
            grid_image = create_styled_image_grid(frames_list, black_gap=3, white_border=1)
            output_filename = "preview_grid_styled.png"
            grid_image.save(output_filename)
            print(f"Saved styled grid to {output_filename}")
        except Exception as e:
            print(f"unable to visualize {str(e)}")
            
def get_video_metadata(filepath):
    cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", filepath]
    result = subprocess.run(cmd, capture_output=True, text=True)
    data = json.loads(result.stdout)
    return data['format'].get('tags', {})

class ProfileContext:
    def __init__(self, name="wan_profile", base_log_dir="./profile_data"):
        self.name = name
        self.base_log_dir = base_log_dir
        self.log_dir = os.path.join(base_log_dir, name)
        os.makedirs(self.log_dir, exist_ok=True)
        self.profiler = None

    def __enter__(self):
        from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler
        
        print(f"DEBUG: Profiler will save to: {os.path.abspath(self.log_dir)}")
        self.profiler = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            # schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
            on_trace_ready=tensorboard_trace_handler(self.log_dir),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            acc_events=True
        )
        self.profiler.__enter__()
        return self.profiler

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.profiler:
            self.profiler.__exit__(exc_type, exc_val, exc_tb)