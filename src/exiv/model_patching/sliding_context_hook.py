import torch

from dataclasses import dataclass

from ..utils.enum import ExtendedEnum
from ..utils.device import OFFLOAD_DEVICE, VRAM_DEVICE
from ..utils.logging import app_logger
from ..model_patching.hook_registry import HookLocation, HookRegistry, HookType, ModelHook

# debug var
low_blend = 0
left_blend_factor = 2

class BlendType(ExtendedEnum):
    LINEAR = "linear"
    PYRAMIND = "pyramid"
    NO_BLEND = "no_blend"

@dataclass
class SlidingContextConfig:
    ctx_len: int = 20
    ctx_overlap: int = 5
    frame_dim: int = 2           # for (B, C, T, H, W). 1 if (B, T, C, H, W)
    blend_type: str = "pyramid"
    
    @property
    def stride(self):
        return self.ctx_len - self.ctx_overlap

class SlidingContextHook(ModelHook):
    def __init__(self, config = None):
        super().__init__()
        self.hook_type = HookType.SLIDING_CONTEXT.value
        self.hook_location = HookLocation.INNER_SAMPLER_STEP.value
        
        self.config: SlidingContextConfig = config or SlidingContextConfig()
        self.device = VRAM_DEVICE
        
    def _get_linear_mask(self, window_length, overlap):
        """
        Creates a trapezoidal mask: 
        - Ramps up from 0 to 1 over 'overlap' frames
        - Stays at 1.0 in the middle
        - Ramps down from 1 to 0 over 'overlap' frames
        """
        mask = torch.ones(window_length, device=self.device)
        if overlap > 0:
            # +2 and [1:-1] ensures we don't start at strictly 0.0 effectively avoiding div/0 issues later
            ramp = torch.linspace(low_blend, 1, overlap + 2, device=self.device)[1:-1]
            mask[:overlap] *= ramp
            mask[-overlap:] *= ramp.flip(0)
            
        return mask

    def _get_pyramid_mask(self, window_length, overlap):
        """
        Creates a triangle/pyramid mask:
        - Peaked at 1.0 in the center of the window.
        - Linearly decreases to near 0 at the edges.
        - Ignores 'overlap' parameter as it affects the whole window.
        """
        mid = window_length // 2
        # handle odd vs even window lengths to ensure peak is centered
        if window_length % 2 == 0:
            # even: e.g. 4 -> [0.33, 0.66, 0.66, 0.33] (approx)
            ramp = torch.linspace(low_blend, 1, mid + 1, device=self.device)[1:]
            mask = torch.cat([ramp, ramp.flip(0)])
        else:
            # odd: e.g. 5 -> [0.33, 0.66, 1.0, 0.66, 0.33]
            ramp = torch.linspace(low_blend, 1, mid + 2, device=self.device)[1:]
            mask = torch.cat([ramp[:-1], ramp.flip(0)])
            
        return mask
    
    def _get_no_blend_mask(self, window_length, overlap, is_first):
        """
        no_blend/Autoregressive Mask:
        - First Window: All 1s (writes everything).
        - Subsequent Windows: 0s on the left overlap (preserves previous history), 1s on the rest.
        """
        mask = torch.ones(window_length, device=self.device)
        if not is_first and overlap > 0:
            mask[:overlap] = 0.0  # Don't overwrite the left context
        return mask
        
    def _get_mask(self, window_length, is_first = False, is_last = False):
        blend_type = self.config.blend_type
        if blend_type not in BlendType.value_list():
            app_logger.warning(f"blend_type {blend_type} not supported. Defaulting to linear")
            blend_type = "linear"
            
        if blend_type == "linear":
            return self._get_linear_mask(window_length, self.config.ctx_overlap)
        elif blend_type == "pyramid":
            return self._get_pyramid_mask(window_length, self.config.ctx_overlap)
        elif blend_type == "no_blend":
            return self._get_no_blend_mask(window_length, self.config.ctx_overlap, is_first)
        
    def execute(self, module, mod_run, x, t, **input):
        if len(x.shape) != 5:
            app_logger.warning(f"Shape {x.shape} is not supported by this hook, skipping processing")
        else:
            num_frames = x.shape[self.config.frame_dim]
            if num_frames <= self.config.ctx_len:
                out = super().execute(module, mod_run, x, t, **input)
                return out
            
            # creating a list of windows to iterate upon
            windows = []
            for start_idx in range(0, num_frames, self.config.stride):
                end_idx = start_idx + self.config.ctx_len
                # fix overflow
                if end_idx > num_frames:
                    end_idx = num_frames
                    start_idx = max(0, end_idx - self.config.ctx_len)
                
                indices = list(range(start_idx, end_idx))
                windows.append(indices)
                if end_idx == num_frames: break
            
            # accumulate and blend the outputs
            final_output = torch.zeros_like(x, device=OFFLOAD_DEVICE)
            count_mask = torch.zeros_like(x, device=OFFLOAD_DEVICE)
            for i, indices in enumerate(windows):
                # slice the window for inputs
                slices = [slice(None)] * x.ndim
                slices[self.config.frame_dim] = indices

                x_slice = x[tuple(slices)]
                
                # slicing relevant inputs in input if they match video length (e.g., embeddings)
                input_slice = {}
                for k, v in input.items():
                    if isinstance(v, torch.Tensor) and \
                        v.shape[self.config.frame_dim] == num_frames:
                        # NOTE: checking if this tensor actually corresponds to frames
                        # this is a heuristic; be careful with other tensors of same size
                        slices_k = [slice(None)] * v.ndim
                        slices_k[self.config.frame_dim] = indices
                    else:
                        input_slice[k] = v
                
                input_slice['t_start'] = indices[0]
                # TODO: complete latent injection
                # if input_slice['t_start'] > 0:
                #     input_slice['t_start'] -= 1
                #     input_slice['time_indices_map'] = {0:0}
                #     t_anchor = torch.zeros((t.shape[0], 1), device=t.device, dtype=t.dtype)
                #     t = torch.cat([t_anchor, t], dim=1)
                #     anchor_frame = x[:, :, 0:1] 
                #     x_slice = torch.cat([anchor_frame, x_slice], dim=2)
                # TODO: t needs to be sliced here
                output_slice = super().execute(module, mod_run, x_slice, t, **input_slice)
                
                if input_slice.get('time_indices_map') is not None:
                    # slicing [1:] removes the first frame (anchor)
                    output_slice = output_slice.narrow(self.config.frame_dim, 1, output_slice.shape[self.config.frame_dim] - 1)
                
                # (20)
                window_mask = self._get_mask(len(indices), is_first=i == 0, is_last=i == len(windows) - 1)
                # reshape to (1, 1, 20, 1, 1)
                mask_shape = [1] * x.ndim
                mask_shape[self.config.frame_dim] = len(indices)
                window_mask = window_mask.view(mask_shape)
                
                temp_out = output_slice * window_mask
                final_output[tuple(slices)] += temp_out.to(OFFLOAD_DEVICE)
                count_mask[tuple(slices)] += window_mask.to(OFFLOAD_DEVICE)
                print("here")
                
            return final_output / (count_mask + 1e-5)
            
        return super().execute(module, mod_run, x, t, **input)

def enable_sliding_context(model: 'ModelMixin', config = None):
    """
    Adds generation of output in smaller context chunks and blends the overlapping regions
    """
    
    HookRegistry.remove_hook_from_module(model, HookType.SLIDING_CONTEXT.value)
    context_hook = SlidingContextHook(config)
    HookRegistry.apply_hook_to_module(model, context_hook)
    
    