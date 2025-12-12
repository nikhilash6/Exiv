import torch

from exiv.components.models.wan.main import sinusoidal_embedding_1d
from exiv.utils.tensor import pad_to_patch_size

from .state import TaylorSeerState
from ...hook_registry import HookRegistry, HookType
from ....utils.logging import app_logger
from ....components.enum import Model


class TaylorSeerModuleHook:
    def __init__(self, n_derivatives=1, max_warmup_steps=3):
        super().__init__()
        self.hook_type = HookType.TAYLOR_SEER_MODULE_HOOK.value
        # each hook manages its own state for its specific block
        self.seer_state = TaylorSeerState(
            n_derivatives=n_derivatives, 
            max_warmup_steps=max_warmup_steps
        )
        
    def reset(self):
        # NOTE: call this before every new generation !
        self.seer_state.reset()
        
    def _fused_operation(self, hidden_states, temb, rotary_emb, encoder_hidden_states, control_hidden_states_list, **fwd_kwargs):
        # we fuse the block output calculation alongwith 
        # the VACE control signals injection, so the state is easier to manage
        
        if temb.ndim == 4:
            # temb: batch_size, seq_len, 6, inner_dim (wan2.2 ti2v)
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
                self.scale_shift_table.unsqueeze(0) + temb.float()
            ).chunk(6, dim=2)
            # batch_size, seq_len, 1, inner_dim
            shift_msa = shift_msa.squeeze(2)
            scale_msa = scale_msa.squeeze(2)
            gate_msa = gate_msa.squeeze(2)
            c_shift_msa = c_shift_msa.squeeze(2)
            c_scale_msa = c_scale_msa.squeeze(2)
            c_gate_msa = c_gate_msa.squeeze(2)
        else:
            # temb: batch_size, 6, inner_dim (wan2.1/wan2.2 14B)
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
                self.scale_shift_table + temb.float()
            ).chunk(6, dim=1)

        # 1. Self-attention
        norm_hidden_states = (self.norm1(hidden_states.float()) * (1 + scale_msa) + shift_msa).type_as(
            hidden_states
        )
        attn_output = self.attn1(norm_hidden_states, None, None, rotary_emb)
        hidden_states = (hidden_states.float() + attn_output * gate_msa).type_as(hidden_states)

        # 2. Cross-attention
        norm_hidden_states = self.norm2(hidden_states.float()).type_as(hidden_states)
        attn_output = self.attn2(norm_hidden_states, encoder_hidden_states, None, None)
        hidden_states = hidden_states + attn_output

        # 3. Feed-forward
        norm_hidden_states = (
            self.norm3(hidden_states.float()) * (1 + c_scale_msa) + c_shift_msa
        ).type_as(hidden_states)
        ff_output = self.ffn(norm_hidden_states)
        hidden_states = (hidden_states.float() + ff_output.float() * c_gate_msa).type_as(hidden_states)

        # NOTE(DefTruth): Fused VACE into block forward to support caching.
        i = self._i
        vace_layers = self._vace_layers
        if i in vace_layers:
            control_hint, scale = control_hidden_states_list.pop()
            hidden_states = hidden_states + control_hint * scale

        return hidden_states
        
    def new_forward(self, *args, **kwargs):
        # TODO: check during run and adapt accordingly
        # Extract 'x' (hidden_states)
        if len(args) > 0:
            hidden_states = args[0]
        else:
            hidden_states = kwargs.get("x", kwargs.get("hidden_states"))

        # Extract 'e' (temb)
        if len(args) > 1:
            temb = args[1]
        else:
            temb = kwargs.get("e", kwargs.get("temb"))

        # Extract 'freqs' (rotary_emb)
        if len(args) > 2:
            rotary_emb = args[2]
        else:
            rotary_emb = kwargs.get("freqs", kwargs.get("rotary_emb"))

        # Extract 'context' (encoder_hidden_states)
        if len(args) > 3:
            encoder_hidden_states = args[3]
        else:
            encoder_hidden_states = kwargs.get("context", kwargs.get("encoder_hidden_states"))

        # NOTE: this is definitely NOT called vace_hints
        # Extract VACE specific kwargs (will be None/Empty for normal Wan)
        vace_hints = kwargs.get("vace_hints", [])
        block_index = kwargs.get("block_index", getattr(self, "_i", -1))
        
        # update step
        self.seer_state.mark_step_begin()
        
        if self.seer_state.should_compute():
            fwd_kwargs = {k: v for k, v in kwargs.items() if k not in ["vace_hints", "block_index"]}
            call_kwargs = fwd_kwargs.copy()
            if "x" not in call_kwargs and len(args) == 0: call_kwargs["x"] = hidden_states
            if "e" not in call_kwargs and len(args) <= 1: call_kwargs["e"] = temb
            if "freqs" not in call_kwargs and len(args) <= 2: call_kwargs["freqs"] = rotary_emb
            if "context" not in call_kwargs and len(args) <= 3: call_kwargs["context"] = encoder_hidden_states
            
            hidden_states = self._fused_operation(hidden_states, temb, rotary_emb, encoder_hidden_states, vace_hints, **call_kwargs)
            self.seer_state.update(hidden_states)
            return hidden_states
        else:
            hidden_states = self.seer_state.approximate()
            return hidden_states
        
class TaylorSeerModelHook:
    def __init__(self):
        super().__init__()
        self.hook_type = HookType.TAYLOR_SEER_MODEL_HOOK.value

    # The hook receives (module, *args, **kwargs)
    def new_forward(self, module, *args, **kwargs):
        x = args[0]          # 'x'
        timestep = args[1]   # 'timestep'
        context = args[2]    # 'context'
        
        # 'clip_fea' might be in args[3] or kwargs
        clip_fea = args[3] if len(args) > 3 else kwargs.get("clip_fea", None)

        bs, c, t, h, w = x.shape
        x = pad_to_patch_size(x, module.patch_size)

        t_len = t
        if module.ref_conv is not None and "reference_latent" in kwargs:
            t_len += 1
        
        # RoPE encoding
        freqs = module.rope_encode(t_len, h, w, device=x.device, dtype=x.dtype)

        # Patch Embed
        with torch.autocast(device_type="cuda", dtype=torch.float32):
            x = module.patch_embedding(x.float()).to(x.dtype)
        grid_sizes = x.shape[2:]
        x = x.flatten(2).transpose(1, 2)

        time_embed = sinusoidal_embedding_1d(module.freq_dim, timestep.flatten()).to(dtype=x[0].dtype)
        e = module.time_embedding(time_embed)
        e = e.reshape(timestep.shape[0], -1, e.shape[-1])
        e0 = module.time_projection(e).unflatten(2, (6, module.dim))

        # Text Context
        context = module.text_embedding(context)

        # Image Context (for I2V)
        context_img_len = None
        if hasattr(module, "img_emb") and module.img_emb is not None and clip_fea is not None:
             context_clip = module.img_emb(clip_fea)
             context = torch.concat([context_clip, context], dim=1)
             context_img_len = clip_fea.shape[-2]
             
        # Inject Reference Latent (I2V)
        full_ref = None
        if module.ref_conv is not None:
            full_ref = kwargs.get("reference_latent", None)
            if full_ref is not None:
                full_ref = module.ref_conv(full_ref).flatten(2).transpose(1, 2)
                x = torch.concat((full_ref, x), dim=1)

        vace_hints = []
        for i, block in enumerate(module.blocks):
            x = block(
                x, 
                e=e0, 
                freqs=freqs, 
                context=context, 
                context_img_len=context_img_len,
                vace_hints=vace_hints
            )
        

        x = module.head(x, e)
        if full_ref is not None:
            x = x[:, full_ref.shape[1]:]

        x = module.unpatchify(x, grid_sizes)
        
        # Crop to original size
        return x[:, :, :t, :h, :w]
    

def reset_taylor_seer_states(model):
    for module in model.modules():
        if hasattr(module, "hook_registry"):
            hook = module.hook_registry.get_hook(HookType.TAYLOR_SEER_MODULE_HOOK.value)
            if hook:
                hook.reset()
    
def wan_module_filter(model: 'ModelMixin'):
    # returns a list of modules on which the Hook should be applied
    return model.blocks

def enable_taylor_seer_cache(model: 'ModelMixin'):
    module_list = []
    if model.type in [Model.WANT2V.value, Model.WANTI2V.value]:
        module_list = wan_module_filter(model)
        
    for m in module_list:
        HookRegistry.apply_hook_to_module(m, TaylorSeerModuleHook())
        
    HookRegistry.apply_hook_to_module(model, TaylorSeerModelHook())
    
    app_logger.info("Taylor seer cache hooks applied")