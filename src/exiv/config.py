import os
import logging
from typing import Any

from .utils.enum import ExtendedEnum
from .utils.file import FilePaths

os.makedirs(FilePaths.OUTPUT_DIRECTORY, exist_ok=True)

class LOADING_MODE(ExtendedEnum):
    NO_OOM = "no_oom"
    LOW_VRAM = "low_vram"
    NORMAL_LOAD = "normal_load"
    
class AppConfig:
    def __init__(self):
        # loading defaults from the env
        self.logging_level = self._get_logging_level(os.getenv("log_level", 3))

        # by default going with low_vram, if all three are provided 
        # then they will be prioritized in this order -> no_oom -> low_vram -> normal
        self.no_oom = self._get_bool_val(os.getenv("no_oom", "0"))
        self.low_vram = self._get_bool_val(os.getenv("low_vram", "1"))
        self.normal_load = self._get_bool_val(os.getenv("normal_load", "0"))
        
        self.disable_mmap = self._get_bool_val(os.getenv("disable_mmap", "0"))
        self.always_safe_load = self._get_bool_val(os.getenv("safe_load", "1"))
        self.auto_download = self._get_bool_val(os.getenv("auto_download", "0"))
        
        self.use_multi_stream = self._get_bool_val(os.getenv("use_multi_stream", "1"))
        
        self.stop_generation = False

    def _get_bool_val(self, val: Any) -> bool:
        return val.lower() in ("1", "true", "yes") if isinstance(val, str) else bool(val)

    def _get_logging_level(self, level_flag) -> int:
        if level_flag in [1, "1"]: return logging.ERROR         # error + warning
        if level_flag in [2, "2"]: return logging.WARNING       
        if level_flag in [3, "3"]: return logging.INFO          # info + warning + error + critical
        if level_flag in [4, "4"]: return logging.DEBUG         # all logs
        return logging.CRITICAL

    def update_config(self, metadata: dict):
        for flag in LOADING_MODE.value_list():
            if flag in metadata:
                setattr(self, flag, self._get_bool_val(metadata[flag]))
            else:
                setattr(self, flag, False)

        if "log_level" in metadata:
            self.logging_level = self._get_logging_level(metadata["log_level"])
        
        if "disable_mmap" in metadata:
            self.disable_mmap =  self._get_bool_val(str(metadata["disable_mmap"]).lower())
             
        if "auto_download" in metadata:
            self.auto_download = self._get_bool_val(metadata["auto_download"])
             
    @property
    def loading_mode(self):
        if self.no_oom: return LOADING_MODE.NO_OOM.value
        elif self.low_vram: return LOADING_MODE.LOW_VRAM.value
        else: return LOADING_MODE.NORMAL_LOAD.value

global_config = AppConfig()

# -------- Util consts
BYTES_IN_MB = 1024 * 1024

# -------- Default prompts
DEFAULT_T2I_PROMPT = "a surreal scenery"
