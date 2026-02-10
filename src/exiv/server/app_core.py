import argparse
import os
from pydantic import BaseModel
from typing import Any, Optional, List

from ..utils.enum import ExtendedEnum
from exiv.utils.inputs import Input, SliderInput, FileInput
# Lazy import for ExtensionRegistry to avoid potential circular issues at module level
# (though strictly speaking app_core -> components -> extensions is fine)

class TaskContext:
    def __init__(self, callback: Any = None, step_size: float = 0.05):
        self._callback = callback
        self._step_size = step_size
        
        # dynamic progress management
        self._global_cursor: float = 0.0
        self._current_chunk_size: float = 0.0
        self._current_stage: Optional[str] = None
    
    def start_anchor(self, name: str, steps: int = 1):
        """
        Start a new progress anchor.
        'steps' defines how many units of 'step_size' this anchor consumes.
        """
        # move cursor past the previous chunk
        self._global_cursor += self._current_chunk_size
        self._global_cursor = min(self._global_cursor, 1.0)
        
        # set up new chunk
        self._current_chunk_size = steps * self._step_size
        self._current_stage = name
        
        # report 0% for the new stage
        self.progress(0.0, f"Starting {name}", stage=name)

    def progress(self, percent: float, status: str, stage: Optional[str] = None):
        """
        Report progress. 
        'percent' is local progress (0.0 - 1.0) within the current anchor.
        """
        target_stage = stage if stage else self._current_stage
        
        # if no anchor set yet, just pass through raw logic or default behavior
        if self._current_chunk_size <= 0:
             # simple pass-through if no anchor active (or steps=0)
             if self._callback:
                 payload = {"status": status}
                 if target_stage: payload["stage"] = target_stage
                 self._callback(percent, payload)
             return

        # global progress
        global_progress = self._global_cursor + (percent * self._current_chunk_size)
        global_progress = min(max(global_progress, 0.0), 1.0)
        
        payload = {"status": status}
        if target_stage: payload["stage"] = target_stage
        
        if self._callback:
            self._callback(global_progress, payload)

class AppOutputType(ExtendedEnum):
    STRING = 'str'
    NUMBER = 'number'
    IMAGE  = 'image'
    AUDIO  = 'audio'
    VIDEO  = 'video'
    THREE_D = '3D'

class Output(BaseModel):
    id: int
    type: AppOutputType

class App(BaseModel):
    name: str
    description: str = ""
    inputs: dict[str, Input]
    outputs: List[Output]     
    handler: Any
    
    def run_standalone(self):
        try:
            from ..components.extension_registry import ExtensionRegistry
            ExtensionRegistry.get_instance().initialize()
        except Exception as e:
            print(f"[Warning] Failed to load extensions in standalone mode: {e}")

        def mock_progress(p, msg):
            print(f"[{p:.0%}] {msg}")

        parser = argparse.ArgumentParser(description=self.name)
        
        for name, inp in self.inputs.items():
            dtype = float if inp.type == "slider" else str
            parser.add_argument(f"--{name}", default=inp.default, type=dtype, help=inp.label)

        args = vars(parser.parse_args())
        args["report_progress"] = mock_progress

        result = self.handler(**args)
        print("\nOutput:", result)