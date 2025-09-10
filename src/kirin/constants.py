import os
import torch
import logging

# TODO: fix these consts and bind to the cli / args
def _get_flag(name: str, default: str = "0") -> bool:
    val = os.getenv(name, default)
    return val.lower() in ("1", "true", "yes") if isinstance(val, str) else bool(val)

# -------- Package Config
LOGGING_LEVEL_FLAG = os.getenv("log_level", 0)      # 0: CRITICAL, 1: ERROR, 2: WARNING, 3: INFO, 4: DEBUG
LOGGING_LEVEL = logging.CRITICAL
if LOGGING_LEVEL_FLAG in [1, "1"]: LOGGING_LEVEL = logging.ERROR
elif LOGGING_LEVEL_FLAG in [2, "2"]: LOGGING_LEVEL = logging.WARNING
elif LOGGING_LEVEL_FLAG in [3, "3"]: LOGGING_LEVEL = logging.INFO
elif LOGGING_LEVEL_FLAG in [4, "4"]: LOGGING_LEVEL = logging.DEBUG

LOW_VRAM_MODE = _get_flag("low_vram", 0)        # O : FALSE, 1 : TRUE
DISABLE_MMAP = _get_flag("disable_mmap", 0)     # some OS have poor / buggy implementation of mmap
ALWAYS_SAFE_LOAD = _get_flag("safe_load", 1) 

# -------- Util consts
BYTES_IN_MB = 1024 * 1024

# -------- Default prompts
DEFAULT_T2I_PROMPT = "a surreal scenery"

