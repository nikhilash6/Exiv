from enum import Enum


# ----- base
class ExtendedEnum(Enum):

    @classmethod
    def value_list(cls):
        return list(map(lambda c: c.value, cls))

    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_
    
    
# -------
class QuantizationMethod(ExtendedEnum):
    BITSANDBYTES = "bitsandbytes"
    QUANTO = "quanto"
    TORCHAO = "torchao"