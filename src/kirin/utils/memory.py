import torch
import psutil


class MemoryManager:
    def available_memory(self, device="cpu"):
        device = torch.device(device)

        if device.type in ["cuda", "hip"]:
            stats = torch.cuda.memory_stats(device)
            mem_active = stats['active_bytes.all.current']
            mem_reserved = stats['reserved_bytes.all.current']
            mem_free_from_driver, _ = torch.cuda.mem_get_info(device)
            mem_free_torch = mem_reserved - mem_active
            return mem_free_from_driver + mem_free_torch
            
        elif device.type == "mps":
            total_mem = psutil.virtual_memory().total
            driver_alloc = torch.mps.current_allocated_memory()
            return max(total_mem - driver_alloc, 0)

        else:   # device.type == "cpu"
            return psutil.virtual_memory().available

    def total_memory(self, device="cpu"):
        device = torch.device(device)

        if device.type in ["cuda", "hip"]:
            return torch.cuda.get_device_properties(device).total_memory
            
        elif device.type == "mps":
            return psutil.virtual_memory().total

        else:   # device.type == "cpu"
            return psutil.virtual_memory().total

        
mem_manager = MemoryManager()