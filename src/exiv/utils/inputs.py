from typing import Any, List, Optional
from pydantic import BaseModel, ConfigDict
from .enum import ExtendedEnum

class InputType(ExtendedEnum):
    VIDEO = "video"
    IMAGE = "image"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    SELECT = "select"
    JSON = "json"
    TEXT = "text"
    FILE = "file"
    MODEL = "model"

class Input(BaseModel):
    # interface class for the inputs
    label: str
    type: str 
    default: Any = None
    
    hint: Optional[str] = None
    min: Optional[float] = None
    max: Optional[float] = None
    step: Optional[float] = None
    options: Optional[List[str]] = None
    accept: Optional[str] = None # For files
    categories: Optional[List[str]] = None # For models
    
    # - this allows ANY other argument (min, max, options, accept, icon, etc.)
    # - it will automatically be serialized to the JSON
    model_config = ConfigDict(extra='allow') 
    
    def validate_value(self, value: Any):
        """
        Performs basic checks based on whatever attributes 
        the user defined (min, max, options)
        """
        # this is kind of ad-hoc for now, will change later
        if self.type == "number" and not isinstance(value, (int, float)):
             try: value = float(value)
             except: raise ValueError(f"'{self.label}' must be a number.")
        
        if isinstance(value, (int, float)):
            if self.min is not None and float(value) < self.min:
                raise ValueError(f"'{self.label}' cannot be less than {self.min}")
            
            if self.max is not None and float(value) > self.max:
                raise ValueError(f"'{self.label}' cannot be greater than {self.max}")

        if self.options and value is not None and value not in self.options:
            raise ValueError(f"'{value}' is not a valid option for {self.label}")

        return value

# wrappers for convenience
def SliderInput(label, min, max, default=0):
    return Input(label=label, type="slider", min=min, max=max, default=default)

def FileInput(label, accept="*"):
    return Input(label=label, type="file", accept=accept)

def ModelInput(label, categories, default=None):
    return Input(label=label, type="model", categories=categories, default=default)
