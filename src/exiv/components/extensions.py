from typing import Any, Dict, List, Optional, Union, Callable
from exiv.utils.inputs import Input

class Extension:
    """
    The Single Manifest Class for all Extensions
    """
    ID: str = "base_extension"
    DISPLAY_NAME: str = "Base Extension"
    VERSION: str = "0.0.1"

    def __init__(self):
        pass

    @classmethod
    def get_inputs(cls) -> Dict[str, Input]:
        """
        Defines the Inputs using the standard Input class
        Used by Frontend to generate the form
        """
        return {}

    def register(self) -> List[Any]:
        """
        Returns a list of capabilities (patches or processors) this extension provides
        
        The Registry will inspect each item:
        - If it's an Extension/Processor object (has inputs/slot): Registered as a Tool/UI item
        - If it's a Callable: Executed immediately (System Patch/Hook)
        """
        return []

    def process(self, **kwargs) -> Any:
        """
        The functional logic if this extension acts as a Processor
        Arguments match the keys in get_inputs()
        """
        raise NotImplementedError("This extension does not support execution mode.")
