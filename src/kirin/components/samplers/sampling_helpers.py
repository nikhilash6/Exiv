import torch

import collections

from ...utils.tensor import repeat_to_batch_size


# TODO: refactor these methods neatly

def get_area_and_mult(conds, x_in, timestep_in):
    '''
    basically tells where the conds are being applied in the current timestep
    (if it is being applied) and by how much (strength). 
    returns
    - input_x : actual patch of data the cond is being applied on
    - mult : strength of cond
    - conditioning : conds
    - area : bounds of the application area
    - control : controlnet conds ?
    '''
    
    # shape of x_in - [batch_size, channels, height, width, ...]    (maybe seq_len or other dims)
    dims = tuple(x_in.shape[2:])    # skipping batch_size and channels
    area = None
    strength = 1.0

    # ---- checking if cond is applicable in the current timestep
    if 'timestep_start' in conds:
        timestep_start = conds['timestep_start']
        if timestep_in[0] > timestep_start:
            return None

    if 'timestep_end' in conds:
        timestep_end = conds['timestep_end']
        if timestep_in[0] < timestep_end:
            return None

    # ---- finding the are, the cond applies to
    if 'area' in conds:
        area = list(conds['area'])
    if 'strength' in conds:
        strength = conds['strength']

    input_x = x_in
    if area is not None:
        for i in range(len(dims)):
            # input_x.shape[i + 2] -> skipping bs and ch, area[len(dims) + i] -> i is the offset of the ith dim in 'dims'
            area[i] = min(input_x.shape[i + 2] - area[len(dims) + i], area[i])
            input_x = input_x.narrow(i + 2, area[len(dims) + i], area[i])

    # ---- multiplier for different parts in the area
    if 'mask' in conds:
        mask_strength = conds.get("mask_strength", 1.0)
        mask = conds['mask']
        assert(mask.shape[1:] == x_in.shape[2:])    # batch is the first dim in mask

        # crop the mask
        if area is not None:
            for i in range(len(dims)):
                mask = mask.narrow(i + 1, area[len(dims) + i], area[i])

        mask = mask * mask_strength
        mult = mask.unsqueeze(1)
    else:
        # apply cond everywhere equally in the area
        mult = torch.ones_like(input_x)

    mult *= strength

    # ---- soften area edge
    # soften the edges of the area, however we don't do it if the mask is present
    if 'mask' not in conds and area is not None:
        rr = 8 # 8 pixels
        # e.g. loop over height and width
        for i in range(len(dims)):
            # starting edge (left or top)
            if area[len(dims) + i] != 0:
                for t in range(rr):
                    m = mult.narrow(i + 2, t, 1)
                    m *= ((1.0/rr) * (t + 1)) # linear gradient
            
            # ending edge (right or bottom)
            if (area[i] + area[len(dims) + i]) < x_in.shape[i + 2]:
                for t in range(rr):
                    m = mult.narrow(i + 2, area[i] - 1 - t, 1)
                    m *= ((1.0/rr) * (t + 1)) # linear gradient

    # ---- prepare conditioning tensors
    conditioning = {}
    model_conds = conds["model_conds"]
    for c in model_conds:
        conditioning[c] = model_conds[c].process_cond(batch_size=x_in.shape[0], device=x_in.device, area=area)

    # controlnet ?
    control = conds.get('control', None)

    cond_obj = collections.namedtuple('cond_obj', ['input_x', 'mult', 'conditioning', 'area', 'control'])
    return cond_obj(input_x, mult, conditioning, area, control)


def prepare_mask(noise_mask, shape, device):
    """ensures noise mask is of proper dimensions"""
    noise_mask = torch.nn.functional.interpolate(noise_mask.reshape((-1, 1, noise_mask.shape[-2], noise_mask.shape[-1])), size=(shape[2], shape[3]), mode="bilinear")
    noise_mask = torch.cat([noise_mask] * shape[1], dim=1)
    noise_mask = repeat_to_batch_size(noise_mask, shape[0])
    noise_mask = noise_mask.to(device)
    return noise_mask


def process_conds(model, noise, conds, device, latent_image=None, denoise_mask=None, seed=None):
    for k in conds:
        conds[k] = conds[k][:]
        resolve_areas_and_cond_masks_multidim(conds[k], noise.shape[2:], device)

    for k in conds:
        calculate_start_end_timesteps(model, conds[k])

    if hasattr(model, 'extra_conds'):
        for k in conds:
            conds[k] = encode_model_conds(model.extra_conds, conds[k], noise, device, k, latent_image=latent_image, denoise_mask=denoise_mask, seed=seed)

    #make sure each cond area has an opposite one with the same area
    for k in conds:
        for c in conds[k]:
            for kk in conds:
                if k != kk:
                    create_cond_with_same_area_if_none(conds[kk], c)

    for k in conds:
        pre_run_control(model, conds[k])

    if "positive" in conds:
        positive = conds["positive"]
        for k in conds:
            if k != "positive":
                apply_empty_x_to_equal_area(list(filter(lambda c: c.get('control_apply_to_uncond', False) == True, positive)), conds[k], 'control', lambda cond_cnets, x: cond_cnets[x])
                apply_empty_x_to_equal_area(positive, conds[k], 'gligen', lambda cond_cnets, x: cond_cnets[x])

    return conds


def apply_empty_x_to_equal_area(conds, uncond, name, uncond_fill_func):
    cond_cnets = []
    cond_other = []
    uncond_cnets = []
    uncond_other = []
    for t in range(len(conds)):
        x = conds[t]
        if 'area' not in x:
            if name in x and x[name] is not None:
                cond_cnets.append(x[name])
            else:
                cond_other.append((x, t))
    for t in range(len(uncond)):
        x = uncond[t]
        if 'area' not in x:
            if name in x and x[name] is not None:
                uncond_cnets.append(x[name])
            else:
                uncond_other.append((x, t))

    if len(uncond_cnets) > 0:
        return

    for x in range(len(cond_cnets)):
        temp = uncond_other[x % len(uncond_other)]
        o = temp[0]
        if name in o and o[name] is not None:
            n = o.copy()
            n[name] = uncond_fill_func(cond_cnets, x)
            uncond += [n]
        else:
            n = o.copy()
            n[name] = uncond_fill_func(cond_cnets, x)
            uncond[temp[1]] = n


def pre_run_control(model, conds):
    s = model.model_sampling
    for t in range(len(conds)):
        x = conds[t]

        timestep_start = None
        timestep_end = None
        percent_to_timestep_function = lambda a: s.percent_to_sigma(a)
        if 'control' in x:
            x['control'].pre_run(model, percent_to_timestep_function)


def create_cond_with_same_area_if_none(conds, c): #TODO: handle dim != 2
    if 'area' not in c:
        return

    c_area = c['area']
    smallest = None
    for x in conds:
        if 'area' in x:
            a = x['area']
            if c_area[2] >= a[2] and c_area[3] >= a[3]:
                if a[0] + a[2] >= c_area[0] + c_area[2]:
                    if a[1] + a[3] >= c_area[1] + c_area[3]:
                        if smallest is None:
                            smallest = x
                        elif 'area' not in smallest:
                            smallest = x
                        else:
                            if smallest['area'][0] * smallest['area'][1] > a[0] * a[1]:
                                smallest = x
        else:
            if smallest is None:
                smallest = x
    if smallest is None:
        return
    if 'area' in smallest:
        if smallest['area'] == c_area:
            return

    out = c.copy()
    out['model_conds'] = smallest['model_conds'].copy() #TODO: which fields should be copied?
    conds += [out]


def encode_model_conds(model_function, conds, noise, device, prompt_type, **kwargs):
    for t in range(len(conds)):
        x = conds[t]
        params = x.copy()
        params["device"] = device
        params["noise"] = noise
        default_width = None
        if len(noise.shape) >= 4: #TODO: 8 multiple should be set by the model
            default_width = noise.shape[3] * 8
        params["width"] = params.get("width", default_width)
        params["height"] = params.get("height", noise.shape[2] * 8)
        params["prompt_type"] = params.get("prompt_type", prompt_type)
        for k in kwargs:
            if k not in params:
                params[k] = kwargs[k]

        out = model_function(**params)
        x = x.copy()
        model_conds = x['model_conds'].copy()
        for k in out:
            model_conds[k] = out[k]
        x['model_conds'] = model_conds
        conds[t] = x
    return conds


def calculate_start_end_timesteps(model, conds):
    s = model.model_sampling
    for t in range(len(conds)):
        x = conds[t]

        timestep_start = None
        timestep_end = None
        if 'start_percent' in x:
            timestep_start = s.percent_to_sigma(x['start_percent'])
        if 'end_percent' in x:
            timestep_end = s.percent_to_sigma(x['end_percent'])

        if (timestep_start is not None) or (timestep_end is not None):
            n = x.copy()
            if (timestep_start is not None):
                n['timestep_start'] = timestep_start
            if (timestep_end is not None):
                n['timestep_end'] = timestep_end
            conds[t] = n


def resolve_areas_and_cond_masks_multidim(conditions, dims, device):
    # We need to decide on an area outside the sampling loop in order to properly generate opposite areas of equal sizes.
    # While we're doing this, we can also resolve the mask device and scaling for performance reasons
    for i in range(len(conditions)):
        c = conditions[i]
        if 'area' in c:
            area = c['area']
            if area[0] == "percentage":
                modified = c.copy()
                a = area[1:]
                a_len = len(a) // 2
                area = ()
                for d in range(len(dims)):
                    area += (max(1, round(a[d] * dims[d])),)
                for d in range(len(dims)):
                    area += (round(a[d + a_len] * dims[d]),)

                modified['area'] = area
                c = modified
                conditions[i] = c

        if 'mask' in c:
            mask = c['mask']
            mask = mask.to(device=device)
            modified = c.copy()
            if len(mask.shape) == len(dims):
                mask = mask.unsqueeze(0)
            if mask.shape[1:] != dims:
                mask = torch.nn.functional.interpolate(mask.unsqueeze(1), size=dims, mode='bilinear', align_corners=False).squeeze(1)

            if modified.get("set_area_to_bounds", False): #TODO: handle dim != 2
                bounds = torch.max(torch.abs(mask),dim=0).values.unsqueeze(0)
                boxes, is_empty = get_mask_aabb(bounds)
                if is_empty[0]:
                    # Use the minimum possible size for efficiency reasons. (Since the mask is all-0, this becomes a noop anyway)
                    modified['area'] = (8, 8, 0, 0)
                else:
                    box = boxes[0]
                    H, W, Y, X = (box[3] - box[1] + 1, box[2] - box[0] + 1, box[1], box[0])
                    H = max(8, H)
                    W = max(8, W)
                    area = (int(H), int(W), int(Y), int(X))
                    modified['area'] = area

            modified['mask'] = mask
            conditions[i] = modified
            
def get_mask_aabb(masks):
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device, dtype=torch.int)

    b = masks.shape[0]

    bounding_boxes = torch.zeros((b, 4), device=masks.device, dtype=torch.int)
    is_empty = torch.zeros((b), device=masks.device, dtype=torch.bool)
    for i in range(b):
        mask = masks[i]
        if mask.numel() == 0:
            continue
        if torch.max(mask != 0) == False:
            is_empty[i] = True
            continue
        y, x = torch.where(mask)
        bounding_boxes[i, 0] = torch.min(x)
        bounding_boxes[i, 1] = torch.min(y)
        bounding_boxes[i, 2] = torch.max(x)
        bounding_boxes[i, 3] = torch.max(y)

    return bounding_boxes, is_empty