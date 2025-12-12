from .caching.taylor_seer.hook import enable_taylor_seer_cache
from ..utils.enum import ExtendedEnum


class CacheType(ExtendedEnum):
    TAYLOR_SEER = "taylor_seer"


def add_cache_hooks(model, cache_type: str):
    assert cache_type in CacheType.value_list(), f"unsupported cache type {cache_type}"
    
    if cache_type == CacheType.TAYLOR_SEER.value:
        enable_taylor_seer_cache(model)