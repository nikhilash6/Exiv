import torch

from typing import Any, Callable

from .hook_registry import HookRegistry, HookType

# TODO: test this code when an image model is added
class InpaintingHook:
    def __init__(self):
        super().__init__()
        self.hook_type = HookType.INPAINT_HOOK.value
        
    def call_wrapper(self, module: torch.nn.Module, og_call: Callable, *args, **kwargs):
        # NOTE: this wrapper requires args and kwargs to be a certain format
        # run and update them as neccessary
        denoise_mask = kwargs.get("denoise_mask", None)
        noisy_input = args[0]
        sigma = kwargs.get("sigma", None)
        og_input = kwargs.get("og_input", None)
        
        assert og_input is not None and sigma is not None, "og_input and sigma is required to process inpainting"
        
        if denoise_mask is not None:
            latent_mask = 1. - denoise_mask
            # sigma is defined for each batch, for a latent image of size [batch_size, channels, height, width]
            # so reshaping sigma from [4] -> [4, 1, 1, 1]   (assuming batch_size = 4)
            reshaped_sigma = sigma.reshape([sigma.shape[0]] + [1] * (len(self.noise.shape) - 1))
            scaled_noise = self.model_sampling.noise_scaling(reshaped_sigma, self.noise, self.latent_image)
            noisy_input = noisy_input * denoise_mask + scaled_noise * latent_mask
        
        modified_args = (noisy_input,) + args[1:]
        out = og_call(modified_args, **kwargs)
        
        # discarding changes in the unmasked areas
        if denoise_mask is not None:
            out = out * denoise_mask + og_input * latent_mask
            
        return out
    
# these are applied to the entire model and are generally __call__ wrappers
def add_preprocessing_hooks(model):
    hook_handle_list = [InpaintingHook]
    
    for hook_cls in hook_handle_list:
        module_hook = hook_cls()
        HookRegistry.apply_hook_to_module(model, module_hook)