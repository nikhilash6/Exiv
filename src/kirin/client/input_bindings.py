from ..utils.enum import ExtendedEnum


class FieldType(ExtendedEnum):
    INT = 'int'
    FLOAT = 'float'
    IMAGE = 'img'
    FILE = 'file'


class InputField:
    def __init__(self, title: str, input_type: FieldType, *args, **kwargs):
        if "desc" not in kwargs: kwargs["desc"] = ""

class OutputField:
    def __init__(self, title: str, output_type: FieldType, *args, **kwargs):
        if "desc" not in kwargs: kwargs["desc"] = ""
        
class ActionButton:
    def __init__(self, title: str, callback, desc=""):
        pass