from .caching.taylor_seer.hook import enable_taylor_seer_cache
from ..utils.enum import ExtendedEnum


class CacheType(ExtendedEnum):
    TAYLOR_SEER = "taylor_seer"


def enable_step_caching(model, cache_type: str | None = None):
    cache_type = cache_type or CacheType.TAYLOR_SEER.value
    assert cache_type in CacheType.value_list(), f"unsupported cache type {cache_type}"
    
    if cache_type == CacheType.TAYLOR_SEER.value:
        enable_taylor_seer_cache(model)