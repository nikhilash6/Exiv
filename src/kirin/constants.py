import torch
import logging

# -------- Package Config
LOGGING_LEVEL = logging.DEBUG  # opts - INFO, WARNING, ERROR, CRITICAL
LOW_VRAM_MODE = False
DISABLE_MMAP = False    # some OS have poor / buggy implementation of mmap
ALWAYS_SAFE_LOAD = True

# -------- System Info
CUDA_AVAILABLE = torch.cuda.is_available()
CUDA_COMPUTE_CAPABILITY = (None, None) if not CUDA_AVAILABLE else torch.cuda.get_device_capability()

# -------- Default prompts
DEFAULT_T2I_PROMPT = "a surreal scenery"