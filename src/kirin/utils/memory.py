import torch
import psutil

class MemoryManager:
    def available_memory(self, device="cpu"):
        free, _ = self._fetch_mem_amount(device)
        return free

    def total_memory(self, device="cpu"):
        _, total = self._fetch_mem_amount(device)
        return total

    def _fetch_mem_amount(self, device):
        device = torch.device(device)

        if device.type == "cuda" or device.type == "hip":
            stats = torch.cuda.memory_stats(device)
            total_mem = torch.cuda.get_device_properties(device).total_memory
            
            # "real free memory", discarding reserved memory
            mem_active = stats['active_bytes.all.current']
            mem_reserved = stats['reserved_bytes.all.current']
            mem_free_from_driver, _ = torch.cuda.mem_get_info(device)
            mem_free_torch = mem_reserved - mem_active
            
            free_mem = mem_free_from_driver + mem_free_torch
            
        elif device.type == "mps":
            total_mem = psutil.virtual_memory().total
            driver_alloc = torch.mps.current_allocated_memory()
            # "free" = system RAM available - current allocated
            free_mem = max(total_mem - driver_alloc, 0)

        else:   # device.type == "cpu"
            total_mem = psutil.virtual_memory().total
            free_mem = psutil.virtual_memory().available

        self.available_memory, self.total_memory = free_mem, total_mem
        
mem_manager = MemoryManager()