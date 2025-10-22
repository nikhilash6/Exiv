import os
import logging
from typing import Any

class AppConfig:
    def __init__(self):
        # loading defaults from the env
        self.logging_level = self._get_logging_level(os.getenv("log_level", 0))
        
        self.low_vram = self._get_bool_val(os.getenv("low_vram", "0"))
        self.disable_mmap = self._get_bool_val(os.getenv("disable_mmap", "0"))
        self.always_safe_load = self._get_bool_val(os.getenv("safe_load", "1"))
        
        self.use_multi_stream = self._get_bool_val(os.getenv("use_multi_stream", "1"))
        
        self.stop_generation = False

    def _get_bool_val(self, val: Any) -> bool:
        return val.lower() in ("1", "true", "yes") if isinstance(val, str) else bool(val)

    def _get_logging_level(self, level_flag) -> int:
        if level_flag in [1, "1"]: return logging.ERROR
        if level_flag in [2, "2"]: return logging.WARNING
        if level_flag in [3, "3"]: return logging.INFO
        if level_flag in [4, "4"]: return logging.DEBUG
        return logging.CRITICAL

    def update_config(self, metadata: dict):
        if "low_vram" in metadata:
            self.low_vram = self._get_bool_val( metadata["low_vram"])

        if "log_level" in metadata:
            self.logging_level = self._get_logging_level(metadata["log_level"])
        
        if "disable_mmap" in metadata:
             self.disable_mmap =  self._get_bool_val(str(metadata["disable_mmap"]).lower())

global_config = AppConfig()

# -------- Util consts
BYTES_IN_MB = 1024 * 1024

# -------- Default prompts
DEFAULT_T2I_PROMPT = "a surreal scenery"
