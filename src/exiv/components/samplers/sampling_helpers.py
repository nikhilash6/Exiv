import torch

import collections

from ...utils.tensor import common_upscale, repeat_to_batch_size


def preprocess_cond(conds, x_in):
    '''
    - calculates conditionals strength
    - reshapes model conditionals to match bs
    - separates controlnet conditionals
    '''
    # ---- strength calc
    strength = conds.get('strength', 1.0)
    mult = torch.ones_like(x_in) * strength

    # ---- prepare conditioning
    conditioning = {}
    model_conds = conds["model_conds"]
    for c in model_conds:
        conditioning[c] = model_conds[c].process_cond(batch_size=x_in.shape[0], device=x_in.device)

    # ---- controlnets
    control = conds.get('control', None)

    cond_obj = collections.namedtuple('cond_obj', ['input_x', 'mult', 'conditioning', 'control'])
    return cond_obj(x_in, mult, conditioning, control)


def prepare_mask(noise_mask, shape, device):
    """ensures noise mask is of proper dimensions"""
    noise_mask = torch.nn.functional.interpolate(noise_mask.reshape((-1, 1, noise_mask.shape[-2], noise_mask.shape[-1])), size=(shape[2], shape[3]), mode="bilinear")
    noise_mask = torch.cat([noise_mask] * shape[1], dim=1)
    noise_mask = repeat_to_batch_size(noise_mask, shape[0])
    noise_mask = noise_mask.to(device)
    return noise_mask


def process_conds(conds, noise_shape, device):
    process_masks(conds, noise_shape[2:], device)
    prepare_controlnet(conds)
    return conds

# NOTE: not very relevant atm, but will update as more models are added
def process_masks(conds, latent_dims, device):
    """
    - moves the mask tensor to the target device (e.g., GPU).
    - resizes the mask to match the dimensions of the latent image.
    """
    for k in conds:
        for cond in conds[k]:
            if 'mask' in cond:
                mask = cond['mask']
                mask = mask.to(device=device)

                # adding a batch dimension
                if len(mask.shape) == len(latent_dims):
                    mask = mask.unsqueeze(0)

                if mask.shape[1:] != latent_dims:
                    if mask.ndim < 4:
                        # adding the channel dim then removing it
                        mask = common_upscale(mask.unsqueeze(1), latent_dims[-1], latent_dims[-2], 'bilinear', 'none').squeeze(1)
                    else:
                        mask = common_upscale(mask, latent_dims[-1], latent_dims[-2], 'bilinear', 'none')

                cond['mask'] = mask

# TODO: test when proper controlnet support is added
def prepare_controlnet(conds):
    """
    Ensures ControlNet is applied symmetrically to positive and negative conds.
    If a ControlNet guides the positive prompt (e.g., with a depth map), the 
    negative prompt also needs a corresponding "empty" ControlNet. This gives 
    the model a clean baseline to push away from, making the ControlNet's guidance 
    much more effective.
    """
    positive_conds = conds.get("positive", [])
    negative_conds = conds.get("negative", [])

    positive_controls = [
        c['control'] for c in positive_conds
        if 'control' in c and c['control'] is not None
    ]

    if not positive_controls:
        return

    neg_chunks_without_control = [
        (chunk, i) for i, chunk in enumerate(negative_conds)
        if chunk.get('control') is None
    ]

    if not neg_chunks_without_control:
        return

    # for each positive ControlNet, apply it to a corresponding negative prompt.
    for i, control_to_add in enumerate(positive_controls):
        # cycle through the available negative prompts.
        target_chunk, chunk_index = neg_chunks_without_control[i % len(neg_chunks_without_control)]
        new_chunk = target_chunk.copy()
        new_chunk['control'] = control_to_add
        conds["negative"][chunk_index] = new_chunk
