from .state import TaylorSeerState
from ...hook_registry import HookType, ModelHook

class TaylorSeerLiteModelHook(ModelHook):
    def __init__(self, n_derivatives=1, max_warmup_steps=3, skip_interval_steps=2):
        super().__init__()
        self.hook_type = HookType.TAYLOR_SEER_LITE_MODEL_HOOK.value
        self.seer_state = TaylorSeerState(
            n_derivatives=n_derivatives, 
            max_warmup_steps=max_warmup_steps,
            skip_interval_steps=skip_interval_steps
        )

    def new_forward(self, module, *args, **kwargs):
        self.seer_state.mark_step_begin()
        
        if self.seer_state.should_compute():
            hidden_states = module._original_forward(*args, **kwargs)
            self.seer_state.update(hidden_states)
            return hidden_states
        else:
            return self.seer_state.approximate()