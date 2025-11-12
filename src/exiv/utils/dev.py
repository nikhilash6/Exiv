# ad-hoc methods used during development
# TODO: delete these before making the final release

import torch
from torch import Tensor

import psutil
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

def print_memory_usage(tag, n=5):
    if global_config.logging_level != logging.DEBUG: return
    
    app_logger.info(f"----------------------------- START: {tag} ----------------------------")
    
    # --- Timestamp ---
    current_time = time.perf_counter()
    delta = current_time - MemoryMonitor.last_call_time
    print(f"--- Checkpoint (Time Since Last Call: {delta:.2f}s) ---")
    MemoryMonitor.last_call_time = current_time # Reset checkpoint

    # --- General CPU RAM ---
    print("\n--- CPU Memory (psutil) ---")
    mem_info = psutil.virtual_memory()
    print(f"Total: {format_bytes(mem_info.total)} (Note: May be host's total)")
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
    else:
        print("\n--- GPU Memory (PyTorch) ---")
        print("CUDA not available.")

    # --- Top N Objects (Tensors & others) ---
    print(f"\n--- Top {n} Largest Objects (via gc) ---")
    print("Note: Sizes for non-tensors are shallow (sys.getsizeof).")
    
    gc.collect()  # Run garbage collection
    
    all_objects = []
    
    # Suppress warnings that can pollute the output
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        for obj in gc.get_objects():
            size_bytes = 0
            desc = ""
            
            try:
                if isinstance(obj, torch.Tensor):
                    # --- FIX: meta tensors have 0 size ---
                    if obj.device.type == 'meta':
                        size_bytes = 0
                    else:
                        size_bytes = obj.nelement() * obj.element_size()
                    
                    desc = (
                        f"Tensor(shape={list(obj.shape)}, "
                        f"dtype={obj.dtype}, device={obj.device})"
                    )
                
                # Specifically find the raw storage blocks
                elif "UntypedStorage" in str(type(obj)):
                    size_bytes = obj.size()
                    device_str = "cpu"
                    try:
                        device_str = obj.device.type
                    except AttributeError:
                        pass # Stays 'cpu' if no device attr
                    desc = f"torch.UntypedStorage(size={obj.size()}, device={device_str})"
                
                else:
                    # Shallow size for all other objects
                    size_bytes = sys.getsizeof(obj)
                    desc = f"Object(type={type(obj).__name__})"
                
                if size_bytes > 0:
                    all_objects.append((size_bytes, desc))
            
            except Exception:
                # Ignore objects that can't be inspected
                pass

    # Sort by size (descending) and print top n
    all_objects.sort(key=lambda x: x[0], reverse=True)
    
    if not all_objects:
        print("No objects found by gc.")
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