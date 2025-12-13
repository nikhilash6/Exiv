import torch

from .state import TaylorSeerState
from ...hook_registry import HookRegistry, HookType, ModelHook
from ....utils.logging import app_logger
from ....components.enum import Model
from ....components.models.wan.main import repeat_e, sinusoidal_embedding_1d
from ....utils.tensor import pad_to_patch_size

class TaylorSeerModuleHook(ModelHook):
    def __init__(self, n_derivatives=1, max_warmup_steps=3, skip_interval_steps=2):
        super().__init__()
        self.hook_type = HookType.TAYLOR_SEER_MODULE_HOOK.value
        self.seer_state = TaylorSeerState(
            n_derivatives=n_derivatives, 
            max_warmup_steps=max_warmup_steps,
            skip_interval_steps=skip_interval_steps
        )
        
    def reset(self):
        self.seer_state.reset()
        
    def _fused_operation(self, module, hidden_states, temb, rotary_emb, encoder_hidden_states, control_hidden_states_list, context_img_len):
        """
        Executes the block logic using 'module' attributes, with VACE fusion injected.
        This logic mirrors exiv.components.models.wan.main.WanAttentionBlock.forward
        """
        x = hidden_states
        e = temb
        freqs = rotary_emb
        context = encoder_hidden_states
        
        if e.ndim < 4:
            e = (module.modulation.to(dtype=x.dtype, device=x.device) + e).chunk(6, dim=1)
        else:
            e = (module.modulation.to(dtype=x.dtype, device=x.device).unsqueeze(0) + e).unbind(2)

        # 1. Self-Attention
        # modulated x = shift + (normalized_x * scale)
        y = module.self_attn(
            torch.addcmul(repeat_e(e[0], x), module.norm1(x), 1 + repeat_e(e[1], x)),
            freqs
        )
        x = torch.addcmul(x, y, repeat_e(e[2], x))
        del y

        # 2. Cross-Attention
        x = x + module.cross_attn(module.norm3(x), context, context_img_len=context_img_len)
        
        # 3. FFN
        y = module.ffn(torch.addcmul(repeat_e(e[3], x), module.norm2(x), 1 + repeat_e(e[4], x)))
        x = torch.addcmul(x, y, repeat_e(e[5], x))
        
        # --- VACE Fusion ---
        if control_hidden_states_list:
            # Check if this block is targeted by VACE (using hook logic or config)
            # Assuming 'vace_layers' logic handled by ModelHook sending explicit list
            control_hint, scale = control_hidden_states_list.pop()
            x = x + control_hint * scale

        return x
        
    def new_forward(self, module, *args, **kwargs):
        
        # Extract Arguments matching WanAttentionBlock.forward(x, e, freqs, context, context_img_len)
        x = args[0] if len(args) > 0 else kwargs.get("x")
        e = args[1] if len(args) > 1 else kwargs.get("e")
        freqs = args[2] if len(args) > 2 else kwargs.get("freqs")
        context = args[3] if len(args) > 3 else kwargs.get("context")
        context_img_len = args[4] if len(args) > 4 else kwargs.get("context_img_len", 257)

        vace_hints = kwargs.get("vace_hints", [])
        
        self.seer_state.mark_step_begin()
        
        if self.seer_state.should_compute():
            hidden_states = self._fused_operation(module, x, e, freqs, context, vace_hints, context_img_len)
            self.seer_state.update(hidden_states)
            return hidden_states
        else:
            return self.seer_state.approximate()

# --- 3. Model Hook ---
class TaylorSeerModelHook(ModelHook):
    def __init__(self):
        super().__init__()
        self.hook_type = HookType.TAYLOR_SEER_MODEL_HOOK.value

    def new_forward(self, module, *args, **kwargs):
        # Extract Args matching Wan21Model.forward(x, timestep, context...)
        x = args[0]
        timestep = args[1]
        context = args[2] if len(args) > 2 else kwargs.get("context", None)
        clip_fea = args[3] if len(args) > 3 else kwargs.get("clip_fea", None)

        # Re-implement Wan21Model logic
        bs, c, t, h, w = x.shape
        x = pad_to_patch_size(x, module.patch_size)

        t_len = t
        if module.ref_conv is not None and "reference_latent" in kwargs:
            t_len += 1
        
        freqs = module.rope_encode(t_len, h, w, device=x.device, dtype=x.dtype)

        with torch.autocast(device_type="cuda", dtype=torch.float32):
            x = module.patch_embedding(x.float()).to(x.dtype)
        grid_sizes = x.shape[2:]
        x = x.flatten(2).transpose(1, 2)

        time_embed = sinusoidal_embedding_1d(module.freq_dim, timestep.flatten()).to(dtype=x[0].dtype)
        e = module.time_embedding(time_embed)
        e = e.reshape(timestep.shape[0], -1, e.shape[-1])
        e0 = module.time_projection(e).unflatten(2, (6, module.dim))

        context = module.text_embedding(context)

        context_img_len = None
        if hasattr(module, "img_emb") and module.img_emb is not None and clip_fea is not None:
             context_clip = module.img_emb(clip_fea)
             context = torch.concat([context_clip, context], dim=1)
             context_img_len = clip_fea.shape[-2]
             
        full_ref = None
        if module.ref_conv is not None:
            full_ref = kwargs.get("reference_latent", None)
            if full_ref is not None:
                full_ref = module.ref_conv(full_ref).flatten(2).transpose(1, 2)
                x = torch.concat((full_ref, x), dim=1)

        # --- VACE Fusion Prep (Placeholder) ---
        # populate this list using your VACE blocks logic
        vace_hints = []
        
        # Run Blocks with Hints
        for i, block in enumerate(module.blocks):
            x = block(
                x, 
                e=e0, 
                freqs=freqs, 
                context=context, 
                context_img_len=context_img_len,
                vace_hints=vace_hints # Passed to Module Hook
            )

        x = module.head(x, e)
        if full_ref is not None:
            x = x[:, full_ref.shape[1]:]

        x = module.unpatchify(x, grid_sizes)
        return x[:, :, :t, :h, :w]

# --- 4. Helpers ---
def reset_taylor_seer_states(model):
    for module in model.modules():
        if hasattr(module, "hook_registry"):
            hook = module.hook_registry.get_hook(HookType.TAYLOR_SEER_MODULE_HOOK.value)
            if hook:
                hook.reset()
    
def wan_module_filter(model):
    return model.blocks

def enable_taylor_seer_cache(model):
    module_list = []
    if model.model_type in [Model.WANT2V.value, Model.WANTI2V.value]:
        module_list = wan_module_filter(model)
    else:
        raise Exception(f"{model.model_type} is not supported by Taylor Seer caching atm")
        
    for m in module_list:
        HookRegistry.apply_hook_to_module(m, TaylorSeerModuleHook())
        
    HookRegistry.apply_hook_to_module(model, TaylorSeerModelHook())
    app_logger.info("Taylor seer cache hooks applied")