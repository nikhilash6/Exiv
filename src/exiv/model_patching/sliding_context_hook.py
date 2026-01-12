from exiv.model_patching.hook_registry import HookRegistry, HookType, ModelHook


class SlidingContextHook(ModelHook):
    def __init__(self):
        super().__init__()
        self.hook_type = HookType.SLIDING_CONTEXT.value
        
    def wrap_model_run(self, mod_run, x, t, **input):
        return super().wrap_model_run(mod_run, x, t, **input)

def enable_sliding_context(model: 'ModelMixin', config = None):
    """
    Adds generation of output in smaller context chunks and blends the overlapping regions
    """
    
    HookRegistry.remove_hook_from_module(model, HookType.SLIDING_CONTEXT.value)
    context_hook = SlidingContextHook()
    HookRegistry.apply_hook_to_module(model, context_hook)
    
    