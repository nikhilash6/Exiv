from .caching.taylor_seer_lite.main import get_taylor_seer_lite_data
from .caching.taylor_seer.main import get_taylor_seer_data
from ..utils.enum import ExtendedEnum
from ..utils.logging import app_logger
from ..model_patching.hook_registry import HookRegistry, HookType

class CacheType(ExtendedEnum):
    TAYLOR_SEER = "taylor_seer"                 # NOTE: holy shit, this takes crazy vram (almost 15 GB more in case of wan)
    TAYLOR_SEER_LITE = "taylor_seer_lite"       # almost 0 vram with same quality as the full method

DEFAULT_CACHE_METHOD = CacheType.TAYLOR_SEER_LITE.value     # model agnostic, caches the entire output

cache_method_dict = {
    CacheType.TAYLOR_SEER.value: get_taylor_seer_data,
    CacheType.TAYLOR_SEER_LITE.value: get_taylor_seer_lite_data,
}

def clear_prev_cache_hooks(model):
    # atm there are no two compatible methods, so clearing everything
    for cache_type, cache_method in cache_method_dict.items():
        try:
            module_list, module_hook, model_hook = cache_method(model)
            for m, mh in zip(module_list, module_hook):
                HookRegistry.remove_hook_from_module(m, mh.hook_type)
            
            if model_hook:
                HookRegistry.remove_hook_from_module(model, model_hook)
        except Exception as e:
            # not all cache methods support all model types
            pass

def enable_step_caching(model, cache_type: str | None = None):
    cache_type = cache_type or DEFAULT_CACHE_METHOD
    assert cache_type in CacheType.value_list(), f"unsupported cache type {cache_type}"
    
    clear_prev_cache_hooks(model)
    
    # module_list, module_hook_list -> module hooks and list of modules on which they are to be applied
    # model_hook -> hook to be applied to the entire model
    cache_method = cache_method_dict.get(cache_type, None)
    assert cache_method is not None, f"{cache_type} has no defined interface method"
    module_list, module_hook_list, model_hook = cache_method(model)
    
    # applying hooks
    for m, mh in zip(module_list, module_hook_list):
        HookRegistry.apply_hook_to_module(m, mh)
    
    if model_hook:
        HookRegistry.apply_hook_to_module(model, model_hook)
    
    app_logger.info(f"{cache_type} cache hooks applied")
    