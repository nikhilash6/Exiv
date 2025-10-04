import torch
from torch import Tensor

import math

from .scheduler_types import calculate_sigmas
from .sampling_helpers import get_area_and_mult, process_conds, prepare_mask
from .enum import DISCARD_PENULTIMATE_SIGMA_SAMPLERS, KSamplerType, SamplerType, SchedulerType
from .sampler_impl import ksampler_factory
from ..conditionals import can_concat_cond, cond_cat
from ...utils.tensor import fix_empty_latent_channels, prepare_noise
from ...utils.device import MemoryManager
from ...model_utils.model_wrapper import ModelWrapper
from ...model_utils.latent import Latent

class KSampler:
    def __init__(
        self,
        wrapped_model: ModelWrapper,
        seed: int,
        steps: int,
        cfg: int,
        sampler_name: str,
        scheduler_name: str,
        positive: torch.Tensor,
        negative: torch.Tensor,
        latent_image: Latent,
        denoise: int = 1.0,
    ):
        assert sampler_name in KSamplerType.value_list() + SamplerType.value_list(), f"sampler {sampler_name} not supported"
        assert scheduler_name in SchedulerType.value_list(), f"scheduler {scheduler_name} not supported"
        assert denoise >= 0.0 and denoise <= 1.0, f"denoise {denoise} out of range, should be between 0 and 1"
        
        self.wrapped_model = wrapped_model
        self.seed = seed
        self.steps = steps
        self.cfg = cfg 
        self.sampler_name = sampler_name
        self.scheduler_name = scheduler_name
        self.positive = positive
        self.negative = negative
        self.latent_image = latent_image
        self.denoise = denoise
        
        self.set_steps(steps, denoise)
        
    def calculate_sigmas(self):
        steps = self.steps
        if discard_penultimate:=(self.sampler_name in DISCARD_PENULTIMATE_SIGMA_SAMPLERS):
            steps += 1
        
        sigmas = calculate_sigmas(self.wrapped_model.model_sampling, self.scheduler, steps)
        if discard_penultimate: sigmas = torch.cat([sigmas[:-2], sigmas[-1:]])
        return sigmas
    
    def set_steps(self, steps, denoise=None):
        self.steps = steps
        if denoise is None or denoise > 0.9999:
            self.sigmas = self.calculate_sigmas(steps).to(self.device)
        else:
            if denoise <= 0.0:
                self.sigmas = torch.FloatTensor([])
            else:
                new_steps = int(steps/denoise)
                sigmas = self.calculate_sigmas(new_steps).to(self.device)
                self.sigmas = sigmas[-(steps + 1):]
    
    # TODO: allow for dynamic injection / overriding of variables such as sigma
    def run_sampling(self, disable_noise = False):
        latent_image = self.latent_image.samples
        latent_image = fix_empty_latent_channels(self.wrapped_model, latent_image)
        
        # decides between random noise or zero noise tensors
        if disable_noise:
            noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
        else:
            batch_inds = self.latent_image.batch_index
            noise = prepare_noise(latent_image, self.seed, batch_inds)

        # TODO: enable calculation of new sigmas as per vars injected at the runtime
        # such as last_step, start_step
        
        sampler = ksampler_factory(self.sampler_name)
        return sample(
            self.wrapped_model,
            noise,
            self.positive,
            self.negative,
            self.cfg,
            self.sigmas,
            sampler,
            self.model_options,
            latent_image=latent_image,
            denoise_mask=self.latent_image.noise_mask,
            callback=lambda *args, **kwargs: None,      # TODO: pass a null fn from the top
            seed=self.seed,
            denoise=self.denoise,
        )

# TODO: refactor in a clean way
def sample(
    wrapped_model: ModelWrapper,
    noise: Tensor,
    positive,
    negative,
    cfg,
    sigmas,
    sampler,
    model_options,
    latent_image,
    denoise_mask,
    callback,
    seed,
    denoise,
):
    '''
    - processes the inputs, masks and conditionals 
    - constructs the model_sampling that is ultimately passed in the sampler's sample method
    - sampler's sample returns the output after running the sampling loop
    '''
    # returning if there are 0 steps
    if sigmas.shape[-1] == 0: return latent_image
   
    og_conds = {"positive": positive, "negative": negative}
    conds = {}
    for k, v in og_conds.items():
        conds[k] = [a.copy() for a in v]
        
    if denoise_mask is not None:
        denoise_mask = prepare_mask(denoise_mask, noise.shape, wrapped_model.device)
        
    if latent_image is not None and torch.count_nonzero(latent_image) > 0:
        latent_image = wrapped_model.process_latent_in(latent_image)
        
    conds = process_conds(wrapped_model, noise, conds, wrapped_model.device, latent_image, denoise_mask, seed)
    pos_conds, neg_conds = conds.get("positive"), conds.get("negative")
    
    extra_args = {"model_options": model_options, "seed":seed}
    
    def denoiser_function(x, sigma, **kwargs):
        return model_sampling(
            wrapped_model,
            x,
            sigma,
            neg_conds,
            pos_conds,
            cfg,
            model_options=kwargs.get("model_options", {}),
            seed=kwargs.get("seed")
        )
    
    samples = sampler.sample(denoiser_function, sigmas, extra_args, callback, noise, latent_image, denoise_mask)
    return wrapped_model.process_latent_out(samples.to(torch.float32))


def model_sampling(model, x, timestep, uncond, cond, cond_scale, model_options={}, seed=None):
    '''
    single sampling step for a given model. This has to be used inside the sampler's sample methods.
    '''
    
    if math.isclose(cond_scale, 1.0) and model_options.get("disable_cfg1_optimization", False) == False:
        uncond_ = None
    else:
        uncond_ = uncond

    conds = [cond, uncond_]
    out = calc_cond_batch(model, conds, x, timestep, model_options)

    for fn in model_options.get("sampler_pre_cfg_function", []):
        args = {"conds":conds, "conds_out": out, "cond_scale": cond_scale, "timestep": timestep,
                "input": x, "sigma": timestep, "model": model, "model_options": model_options}
        out  = fn(args)

    cond_pred, uncond_pred = out[0], out[1]
    
    # ------ applying cfg ---------
    if "sampler_cfg_function" in model_options:
        args = {"cond": x - cond_pred, "uncond": x - uncond_pred, "cond_scale": cond_scale, "timestep": timestep, "input": x, "sigma": timestep,
                "cond_denoised": cond_pred, "uncond_denoised": uncond_pred, "model": model, "model_options": model_options}
        cfg_result = x - model_options["sampler_cfg_function"](args)
    else:
        cfg_result = uncond_pred + (cond_pred - uncond_pred) * cond_scale

    for fn in model_options.get("sampler_post_cfg_function", []):
        args = {"denoised": cfg_result, "cond": cond, "uncond": uncond, "model": model, "uncond_denoised": uncond_pred, "cond_denoised": cond_pred,
                "sigma": timestep, "model_options": model_options, "input": x}
        cfg_result = fn(args)

    return cfg_result


# TODO: adapt and refactor, don't need many of these functionalities
def calc_cond_batch(model, conds, x_in, timestep, model_options):
    out_conds = []
    out_counts = []
    to_run = []

    for i in range(len(conds)):
        out_conds.append(torch.zeros_like(x_in))
        out_counts.append(torch.ones_like(x_in) * 1e-37)

        cond = conds[i]
        if cond is not None:
            for x in cond:
                p = get_area_and_mult(x, x_in, timestep)
                if p is None:
                    continue

                to_run += [(p, i)]

    while len(to_run) > 0:
        first = to_run[0]
        first_shape = first[0][0].shape
        to_batch_temp = []
        for x in range(len(to_run)):
            if can_concat_cond(to_run[x][0], first[0]):
                to_batch_temp += [x]

        to_batch_temp.reverse()
        to_batch = to_batch_temp[:1]

        free_memory = MemoryManager.available_memory(x_in.device)
        for i in range(1, len(to_batch_temp) + 1):
            batch_amount = to_batch_temp[:len(to_batch_temp)//i]
            input_shape = [len(batch_amount) * first_shape[0]] + list(first_shape)[1:]
            if model.memory_required(input_shape) * 1.5 < free_memory:
                to_batch = batch_amount
                break

        input_x = []
        mult = []
        c = []
        cond_or_uncond = []
        area = []
        control = None
        patches = None
        for x in to_batch:
            o = to_run.pop(x)
            p = o[0]
            input_x.append(p.input_x)
            mult.append(p.mult)
            c.append(p.conditioning)
            area.append(p.area)
            cond_or_uncond.append(o[1])
            control = p.control
            patches = p.patches

        batch_chunks = len(cond_or_uncond)
        input_x = torch.cat(input_x)
        c = cond_cat(c)
        timestep_ = torch.cat([timestep] * batch_chunks)

        if control is not None:
            c['control'] = control.get_control(input_x, timestep_, c, len(cond_or_uncond))

        transformer_options = {}
        if 'transformer_options' in model_options:
            transformer_options = model_options['transformer_options'].copy()

        if patches is not None:
            if "patches" in transformer_options:
                cur_patches = transformer_options["patches"].copy()
                for p in patches:
                    if p in cur_patches:
                        cur_patches[p] = cur_patches[p] + patches[p]
                    else:
                        cur_patches[p] = patches[p]
                transformer_options["patches"] = cur_patches
            else:
                transformer_options["patches"] = patches

        transformer_options["cond_or_uncond"] = cond_or_uncond[:]
        transformer_options["sigmas"] = timestep

        c['transformer_options'] = transformer_options

        if 'model_function_wrapper' in model_options:
            output = model_options['model_function_wrapper'](model.apply_model, {"input": input_x, "timestep": timestep_, "c": c, "cond_or_uncond": cond_or_uncond}).chunk(batch_chunks)
        else:
            output = model.apply_model(input_x, timestep_, **c).chunk(batch_chunks)

        for o in range(batch_chunks):
            cond_index = cond_or_uncond[o]
            a = area[o]
            if a is None:
                out_conds[cond_index] += output[o] * mult[o]
                out_counts[cond_index] += mult[o]
            else:
                out_c = out_conds[cond_index]
                out_cts = out_counts[cond_index]
                dims = len(a) // 2
                for i in range(dims):
                    out_c = out_c.narrow(i + 2, a[i + dims], a[i])
                    out_cts = out_cts.narrow(i + 2, a[i + dims], a[i])
                out_c += output[o] * mult[o]
                out_cts += mult[o]

    for i in range(len(out_conds)):
        out_conds[i] /= out_counts[i]

    return out_conds
