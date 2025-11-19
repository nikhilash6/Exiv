import torch
import torch.nn as nn
from torch import Tensor

import os
import functools
import safetensors
from typing import List, Optional, Union
import uuid

from ..utils.dtype import cast_to
from ..utils.device import VRAM_DEVICE, MemoryManager, ProcDevice, is_same_device
from ..utils.file import ensure_model_available
from ..utils.logging import app_logger
from ..config import global_config, BYTES_IN_MB
from ..quantizers.base import QuantType, Quantizer, get_quantizer
from ..model_patching.efficient_loading_hook import enable_efficient_loading


class LoraMixin:
    """
    Adds support for adding multiple LoRA weights to the base model
    with varying strength and time schedules.
    """
    def __init__(self):
        # class using this mixin must call super().__init__()
        self.lora_definitions = [] 
        self.active_lora_schedule = {}
        self.current_lora_step = -1
        
    def create_lora_key_mapping(self, state_dict: dict):
        # maps the lora keys according to the model specification (needs to be overriden)
        # e.g. "lora_unet_up_blocks_1..." -> "up_blocks.1..."
        mapped_weights = {}
        for key, weight in state_dict.items():
            if "lora_" in key: 
                mapped_weights[key] = weight.to(dtype=torch.float16, device="cpu")
                
        return mapped_weights

    def add_lora(self, lora_path: str, base_strength: float | List = 1.0):
        app_logger.info(f"Loading LoRA: {lora_path}")
        
        lora_sd = get_state_dict(model_path=lora_path)
        mapped_weights = self.create_lora_key_mapping(lora_sd)

        self.lora_definitions.append({
            "name": os.path.splitext(os.path.basename(lora_path))[0],
            "weights": mapped_weights,
            "base_strength": base_strength,
            "scale": lora_sd.get("scale", 1.0)
        })
        
    def _expand_schedule(self, strength_input: Union[float, List[float]], total_steps: int) -> List[float]:
        # single float -> constant schedule
        if isinstance(strength_input, (float, int)):
            return [float(strength_input)] * total_steps
        
        # List -> expand, [1, 2, 3] -> [1, 1, 1, 2, 2, 2, 3, 3, 3]
        src_len = len(strength_input)
        new_schedule = []
        inc = src_len / total_steps
        pos = 0.0
        
        for _ in range(total_steps):
            idx = int(pos)
            if idx >= src_len: idx = src_len - 1
            
            new_schedule.append(strength_input[idx])
            pos += inc
            
        return new_schedule

    def setup_lora_schedule(self, total_steps: int):
        self.active_lora_schedule = {}
        
        temp_schedules = []
        for lora in self.lora_definitions:
            temp_schedules.append(self._expand_schedule(lora["base_strength"], total_steps))

        # identifying constant vs dynamic strength loras
        constant_indices = []
        dynamic_definitions = []
        dynamic_schedules = []
        
        for i, sched in enumerate(temp_schedules):
            if len(sched) > 0 and (max(sched) == min(sched)):
                if sched[0] != 0: # Ignore 0-strength LoRAs
                    constant_indices.append(i)
            else:
                dynamic_definitions.append(self.lora_definitions[i])
                dynamic_schedules.append(sched)

        # fusing constant loras
        if constant_indices:
            app_logger.info(f"Fusing {len(constant_indices)} constant LoRAs...")
            
            to_fuse = []
            for idx in constant_indices:
                lora = self.lora_definitions[idx]
                strength = temp_schedules[idx][0]
                to_fuse.append((lora, strength))
            
            fused_lora_def = self._merge_constant_loras(to_fuse)
            dynamic_definitions.insert(0, fused_lora_def)
            dynamic_schedules.insert(0, [1.0] * total_steps)
        
        self.lora_definitions = dynamic_definitions
        for step in range(total_steps):
            self.active_lora_schedule[step] = [
                sched[step] if step < len(sched) else sched[-1] for sched in dynamic_schedules
            ]
        
        self.current_lora_step = -1
        
    def _merge_constant_loras(self, lora_list):
        merged_weights = {}
        device = self.gpu_device
        
        down_patterns = ["lora.down", "lora_down"]

        with torch.no_grad():
            for lora_def, strength in lora_list:
                weights = lora_def["weights"]
                lora_scale = lora_def["scale"]
                
                eff_strength = strength * lora_scale
                processed_keys = set()
                
                for key in weights.keys():
                    if key in processed_keys: continue
                    
                    # --- Case 1: Wan LoRA "diff" or bias keys (Simple Addition) ---
                    if "diff" in key or "diff_b" in key:
                        w = weights[key].to(device, dtype=torch.float32)
                        delta = w * eff_strength
                        
                        if key not in merged_weights:
                            merged_weights[key] = delta.cpu()
                        else:
                            # add to accumulator (move acc to device for op, then back)
                            merged_weights[key] = (merged_weights[key].to(device) + delta).cpu()
                            
                        processed_keys.add(key)
                        del w, delta

                    # --- Case 2: Standard LoRA A/B Pairs (Matmul then Addition) ---
                    elif any(p in key for p in down_patterns):
                        down_key = key
                        
                        # find matching Up key
                        up_key = None
                        for p in down_patterns:
                            if p in down_key:
                                # distinct replacement to handle lora.down vs lora_down
                                up_p = p.replace("down", "up")
                                candidate = down_key.replace(p, up_p)
                                if candidate in weights:
                                    up_key = candidate
                                    break
                        
                        if up_key:
                            w_down = weights[down_key].to(device, dtype=torch.float32)
                            w_up = weights[up_key].to(device, dtype=torch.float32)
                            
                            # compute Dense Delta: (Up @ Down) * scale * strength
                            # shape: Up=(Out, Rank), Down=(Rank, In) -> (Out, In)
                            dense_delta = (w_up @ w_down) * eff_strength
                            
                            # create a unified key for the fused weight. 
                            # we map this to a 'diff' key so the model knows it's a residual.
                            fused_key = up_key.replace("lora_up", "fused_diff").replace("lora.up", "fused_diff")
                            
                            if fused_key not in merged_weights:
                                merged_weights[fused_key] = dense_delta.cpu()
                            else:
                                merged_weights[fused_key] = (merged_weights[fused_key].to(device) + dense_delta).cpu()
                            
                            processed_keys.add(down_key)
                            processed_keys.add(up_key)
                            del w_down, w_up, dense_delta

                torch.cuda.empty_cache()

        return {
            "name": "fused_constant_loras",
            "weights": merged_weights,
            "base_strength": 1.0, # Strength is baked into weights
            "scale": 1.0
        }
            
    # TODO: move to hooks, with modular loading
    def patch_lora_weights(self, step_index: int):
        # apply lora weights acc to current step strength
        
        prev_strengths = [0.0] * len(self.lora_definitions)
        if self.current_lora_step != -1:
            prev_strengths = self.active_lora_schedule.get(self.current_lora_step, prev_strengths)

        new_strengths = self.active_lora_schedule.get(step_index, prev_strengths)

        # Optimization: if strengths haven't changed, do nothing
        if prev_strengths == new_strengths:
            self.current_lora_step = step_index
            return

        with torch.no_grad():
            for i, lora_def in enumerate(self.lora_definitions):
                strength_diff = new_strengths[i] - prev_strengths[i]
                
                if abs(strength_diff) < 1e-6: continue

                self._apply_single_lora_delta(lora_def, strength_diff)

        self.current_lora_step = step_index
        
    def unpatch_lora_weights(self):
        # remove any applied lora weights
        
        if self.current_lora_step != -1:
            # apply negative strength of current state to return to 0
            current_strengths = self.active_lora_schedule.get(self.current_lora_step)
            
            with torch.no_grad():
                for i, strength in enumerate(current_strengths):
                    if strength != 0:
                        self._apply_single_lora_delta(self.lora_definitions[i], -strength)
        
        self.lora_definitions = []
        self.active_lora_schedule = {}
        self.current_lora_step = -1

    def _apply_single_lora_delta(self, lora_def, strength_delta):
        """
        Adds (LoRA * strength_delta) to the model weights in-place.
        """
        weights = lora_def["weights"]
        scale = lora_def["scale"]

        for name, module in self.named_modules():
            if not isinstance(module, (nn.Linear, nn.Conv2d)): continue

            up_key, down_key = self._find_lora_keys(name, weights)
            
            if up_key and down_key:
                # moving small LoRA matrices to GPU only for calculation
                w_up = weights[up_key].to(module.weight.device)
                w_down = weights[down_key].to(module.weight.device)

                if isinstance(module, nn.Linear):
                    delta = (w_up @ w_down)
                else:
                    # conceptual
                    delta = torch.mm(w_up.flatten(1), w_down.flatten(1)).reshape(module.weight.shape)
                
                # in-place addition: W_new = W_old + (Delta * Scale * Strength_Diff)
                module.weight.data.add_(delta, alpha=scale * strength_delta)

    def _find_lora_keys(self, module_name, weights):
        return None, None

# bypassing weight creation at model init
class ModuleMeta(type(nn.Module)):
    def __call__(cls, *args, **kwargs):
        model_dtype = kwargs.pop("dtype", torch.float32)
        quant_type = kwargs.get("quant_type", None)
        original_dtype = torch.get_default_dtype()
        
        try:
            torch.set_default_dtype(model_dtype)
            
            # zero init weight load
            with torch.device("meta"):
                instance = super().__call__(*args, **kwargs)
                quantizer: Quantizer = get_quantizer(quant_type=quant_type)
                instance.quantizer = quantizer
                if quantizer is not None:
                    quantizer.validate_environment()
                    quantizer.process_model_before_weight_loading(model=instance)
                
                if isinstance(instance, ModelMixin):    # mainly for safety
                    enable_efficient_loading(instance)  # kinda default hook
                    if not getattr(instance, 'dtype', None):
                        instance.dtype = model_dtype
                        
        finally:
            torch.set_default_dtype(original_dtype)
        
        return instance


class ModelMixin(nn.Module, LoraMixin, metaclass=ModuleMeta):
    '''
    Adds additional feature to the base model
    
    - (TODO) telemetry / stats
    - lora patching
    - zero init loading
    - (TODO) multi gpu sharding
    - (TODO) implement low cpu mem usage feature
    - auto block swapping during low memory
    - (TODO) priority swapping
    - (TODO) cuda streams for offloading
    - (TODO) support GGUF loading
    - quantization support
    - safetensor support
    - (TODO) improve safetensor loading
    - URL download support
    '''
    def __init__(self, device: str = None, quant_type: QuantType = None, model_path: str = None, dtype = torch.float32):     # quant_type, dtype is used by the meta class
        super().__init__()
        self.gpu_device = device or VRAM_DEVICE
        self.model_path = model_path
        self.model_arch_config = None
    
    def clear_cache(self):
        # TODO: legacy code, will remove after a final check
        pass
    
    @staticmethod
    def is_leaf_module(module: nn.Module) -> bool:
        # TODO: this needs major fixing. Rn we are considering any module with
        # a parameter as the leaf module, so we don't have to load the parameters separately,
        # but this means that modules with multiple sub modules and even just one parameter
        # count as leaf (and will be significantly heavy than a leaf), thus increasing the min mem required
        if len(list(module.parameters(recurse=False))) > 0:
            return True
        
        return len(list(module.children())) == 0

    @staticmethod
    def _module_size(module: nn.Module):
        ms = 0
        for param in module.parameters(recurse=False):
            ms += param.nelement() * param.element_size()
        return round(ms / BYTES_IN_MB, 2)
    
    def __call__(self, *args, **kwargs):
        def original_call(*args, **kwargs):
            with torch.inference_mode():
                # moving the inputs to GPU
                app_logger.debug(f"moving the inputs to {self.gpu_device}")
                new_args = tuple(cast_to(a, device=self.gpu_device, dtype=self.dtype) if torch.is_tensor(a) else a for a in args)
                new_kwargs = {k: (cast_to(v, device=self.gpu_device, dtype=self.dtype) if torch.is_tensor(v) else v) for k, v in kwargs.items()}

                return super(ModelMixin, self).__call__(*new_args, **new_kwargs)
            
        registry = getattr(self, "hook_registry", None)
        if registry and registry.head.next_hook != registry.tail:
            wrapped_call = registry.get_modified_call(original_call)
            return wrapped_call(*args, **kwargs)
        else:
            return original_call(*args, **kwargs)

    # code adapted from Huggingface Diffusers
    def load_model(
        self,
        model_path = None,              # model file path (override for flexibility)
        force_download=False,           # re_download models
        download_url=None,              # file url (optional)
        dtype=None                      # TODO: hardware specific dtype
    ):
        model_path = model_path or self.model_path
        assert model_path is not None, "model_path is required"
        # loading everything on the CPU, then modularly offloading to the GPU
        device = ProcDevice.CPU.value
        self.dtype = dtype or self.dtype
        
        model_path = ensure_model_available(model_path, download_url, force_download)
        
        state_dict = get_state_dict(model_path)
        model_state_dict = self.state_dict()
        
        for param_name, param in state_dict.items():
            if param_name not in model_state_dict: 
                app_logger.warning(f"skipping the param {param_name} as it's not present in the model definition")
                continue
            
            if self.dtype is not None:
                if self.quantizer is not None:
                    pass    # not overiding dtype of quantized models
                else:
                    param = param.to(self.dtype)
            
            # For compatibility with PyTorch load_state_dict which converts state dict dtype to existing dtype in model, and which
            # uses `param.copy_(input_param)` that preserves the contiguity of the parameter in the model
            # Reference: https://github.com/pytorch/pytorch/blob/db79ceb110f6646523019a59bbd7b838f43d4a86/torch/nn/modules/module.py#L2040C29-L2040C29
            old_param = self
            splits = param_name.split(".")
            for split in splits:
                # recursively drill down: model.down_blocks[0].attentions[0].proj_in.weight
                old_param = getattr(old_param, split)
            
            # param_name might be for a buffer or something not loadable, skip it
            if not isinstance(old_param, (torch.nn.Parameter, torch.Tensor)):
                old_param = None
                
            if old_param is not None:
                if self.dtype is None:
                    param = param.to(old_param.dtype)
                    
                if old_param.is_contiguous():
                    param = param.contiguous()
            
            # bnb params are flattened.
            # gguf quants have a different shape based on the type of quantization applied
            if model_state_dict[param_name].shape != param.shape:
                if self.quantizer is not None:
                    self.quantizer.check_quantized_param_shape(param_name, model_state_dict[param_name], param)
                else:
                    raise ValueError(
                        f"Cannot load {model_path} because {param_name} expected shape {model_state_dict[param_name].shape}, but got {param.shape}."
                    )
            
            # final assignment
            if self.quantizer is not None and self.quantizer.check_if_quantized_param(
                self, param, param_name, state_dict, dtype=self.dtype
            ):
                self.quantizer.create_quantized_param(
                    self,
                    param,
                    param_name,
                    device,
                    state_dict,
                    dtype=self.dtype
                )
            else:
                set_module_tensor_to_device(self, param_name, device, value=param, dtype=self.dtype)
    
    def process_latent_in(self, latent_in: Tensor) -> Tensor:
        assert self.model_arch_config is not None, "model_arch_config not set"
        return self.model_arch_config.latent_format.process_in(latent_in)
    
    def process_latent_out(self, latent_out: Tensor) -> Tensor:
        assert self.model_arch_config is not None, "model_arch_config not set"
        return self.model_arch_config.latent_format.process_out(latent_out)
    
    def prepare_conds_for_model(self, cond_group_name: str, cond_list: List[List], noise, **kwargs):
        out = []
        for cond in cond_list:
            # --- Step 1: Standardization ---
            # cond : [tensor, dict]
            temp = cond[1].copy()
            model_conds = {}
            
            # move the text embedding to a standard key for the model to find later
            if cond[0] is not None:
                temp["cross_attn"] = cond[0]
                
            temp["model_conds"] = model_conds
            temp["uuid"] = uuid.uuid4()         # differentiating conds

            # --- Step 2: Formatting  ---
            params = temp.copy()
            params["device"] = self.gpu_device
            params["noise"] = noise
            
            spatial_compression_factor = kwargs.get("spatial_compression_factor", 8)
            if len(noise.shape) >= 4:
                params["width"] = params.get("width", noise.shape[3] * spatial_compression_factor)
                params["height"] = params.get("height", noise.shape[2] * spatial_compression_factor)
                
            params["prompt_type"] = params.get("prompt_type", cond_group_name)
            
            # extra arguments (like latent_image, denoise_mask)
            for k in kwargs:
                if k not in params:
                    params[k] = kwargs[k]

            # model specific formatting
            formatted_results = self.format_conds(**params)

            # --- Step 3: Update the 'model_conds' dictionary ---
            current_model_conds = temp['model_conds'].copy()
            for k in formatted_results:
                current_model_conds[k] = formatted_results[k]
                
            temp['model_conds'] = current_model_conds
            out.append(temp)
            
        return out
    
    def format_conds(self):
        # this formats the conds to a format as required by the underlying model
        raise NotImplementedError("Child instance has not overriden this empty impl.")

def move_model(model, device):
    # handling device movement through our custom logic
    for name, module in model.named_modules():
        move_module(
            model,
            module,
            name, 
            target_device=device,
        )

    return model

c = 0
# TODO: dtype and non_blocking params are not handled as of now
def move_module(model, module, module_name, target_device=None):
    """
    This contains the centralized logic for moving different module types 
    between devices
    """
    if module is None: return   # m_ref can turn out to be None
    
    global c
    app_logger.debug(f"Loading the current module: {c}")
    c += 1
    
    target_device = target_device or model.gpu_device
    app_logger.debug(f"Moving {module_name} to {target_device}")
    
    module_class_name = module.__class__.__name__
    is_bnb_module = module_class_name in ["Linear8bitLt", "Linear4bit"]

    if any(p.device.type == "meta" for p in module.parameters(recurse=False)):
        module.to_empty(device=target_device)
    
    elif is_bnb_module:
        device_index = torch.device(target_device).index
        if device_index is None:
             device_index = torch.cuda.current_device() # Get default index if "cuda"
        
        # .cuda(device_index) / to is overridden by bnb
        module.to(target_device)
        
        # handling the movement of linear8bit
        # after the first forward, quant weights are stored in the state
        if hasattr(module, "weight") and getattr(module.weight, "CB", None) is not None:
            module.weight.CB = module.weight.CB.to(target_device)
            
        if hasattr(module, "weight") and getattr(module.weight, "SCB", None) is not None:
            module.weight.SCB = module.weight.SCB.to(target_device)
        
        if hasattr(module, "state") and getattr(module.state, "CB", None) is not None:
            module.state.CB = module.state.CB.to(target_device)
            
        if hasattr(module, "state") and getattr(module.state, "SCB", None) is not None:
            module.state.SCB = module.state.SCB.to(target_device)

        
    else:
        # standard .to() for all other regular modules
        module.to(device=target_device)

    # app_logger.debug(f"modules rn: {[m.__class__.__name__ for mn, m in model.named_modules() if m != model]}")
    # MemoryManager.clear_memory()


# lots of checks that can be skipped
def set_module_tensor_to_device(
    module: nn.Module,
    tensor_name: str,
    device: Union[int, str, torch.device],
    value: Optional[torch.Tensor] = None,
    dtype: Optional[Union[str, torch.dtype]] = None,
    non_blocking: bool = False,
):
    # traverse the nested modules using the '.' in the tensor name (e.g., 'encoder.layer.0.weight')
    if "." in tensor_name:
        splits = tensor_name.split(".")
        for split in splits[:-1]:
            new_module = getattr(module, split)
            if new_module is None:
                raise ValueError(f"{module} has no attribute {split}.")
            module = new_module
        tensor_name = splits[-1]

    # ensure the tensor name corresponds to an existing parameter or buffer
    if tensor_name not in module._parameters and tensor_name not in module._buffers:
        raise ValueError(f"{module} does not have a parameter or a buffer named {tensor_name}.")
    is_buffer = tensor_name in module._buffers
    old_value = getattr(module, tensor_name)

    if old_value.device == torch.device("meta") and device not in ["meta", torch.device("meta")] and value is None:
        raise ValueError(f"{tensor_name} is on the meta device, we need a `value` to put in on {device}.")

    param = module._parameters[tensor_name] if tensor_name in module._parameters else None
    param_cls = type(param)

    
    if value is not None:
        if dtype is None:
            # For compatibility with PyTorch load_state_dict which converts state dict dtype to existing dtype in model
            value = value.to(old_value.dtype, non_blocking=non_blocking)
        elif not str(value.dtype).startswith(("torch.uint", "torch.int", "torch.bool")):
            value = value.to(dtype, non_blocking=non_blocking)

    device_quantization = None
    with torch.no_grad():
        # temporarily set the device to 'cpu' to handle quantization correctly before the final move.
        # if it's currently not on gpu then it needs to processed first before moving to gpu (if its the target)
        if (
            param is not None
            and param.device.type not in ("cuda", "xpu")
            and torch.device(device).type in ("cuda", "xpu")
            and param_cls.__name__ in ["Int8Params", "FP4Params", "Params4bit"]
        ):
            device_quantization = device 
            device = "cpu"

        if isinstance(value, torch.Tensor):
            new_value = value.to(device, non_blocking=non_blocking)
        else:
            new_value = torch.tensor(value, device=device)

        # revert the target device to the original GPU
        if device_quantization is not None:
            device = device_quantization

        # --- final assignment
        # simple assignment for the buffer
        if is_buffer:
            module._buffers[tensor_name] = new_value
        
        # update if a new value was provided OR if the device actually changed.
        elif value is not None or not is_same_device(torch.device(device), module._parameters[tensor_name].device):
            param_cls = type(module._parameters[tensor_name])
            kwargs = module._parameters[tensor_name].__dict__
            
            # special handling for low-precision/quantized parameter classes (e.g., bitsandbytes)
            if param_cls.__name__ in ["Int8Params", "FP4Params", "Params4bit"]:
                # cast to fp16 for 8-bit serialization/compatibility if needed
                if param_cls.__name__ == "Int8Params" and new_value.dtype == torch.float32:
                    new_value = new_value.to(torch.float16, non_blocking=non_blocking)
                
                # quantize the weights on the GPU first then move to the CPU
                if device == "cpu" and param_cls.__name__ == "Int8Params":
                    new_value = param_cls(new_value, requires_grad=old_value.requires_grad, **kwargs).to(0).to("cpu")
                    new_value.CB = new_value.CB.to("cpu")
                    new_value.SCB = new_value.SCB.to("cpu")
                else:
                    # re-wrap the tensor using its specialized class and move to the final device
                    new_value = param_cls(new_value, requires_grad=old_value.requires_grad, **kwargs).to(
                        device, non_blocking=non_blocking
                    )

            # other known quantized tensor types (affine one is from torchao)
            elif param_cls.__name__ in ["QTensor", "QBitsTensor", "AffineQuantizedTensor"]:
                new_value = torch.nn.Parameter(new_value, requires_grad=old_value.requires_grad).to(
                    device, non_blocking=non_blocking
                )
            
            # default
            else:
                new_value = param_cls(new_value, requires_grad=old_value.requires_grad).to(
                    device, non_blocking=non_blocking
                )

            module._parameters[tensor_name] = new_value

    # freeing old_value (safety check)
    # MemoryManager.clear_memory()

# code adapted from ComfyUI
def get_state_dict(model_path, device=torch.device("cpu")):
    if isinstance(device, str):
        device = torch.device(device)
    
    file_extension = os.path.basename(model_path).split(".")[-1]
    if file_extension in ["safetensors", "sft"]:
        try:
            # safetensor's zero copy loading (pt - pytorch)
            kwargs = {"framework": "pt"}
            # safetensors only support cpu and cuda, and doesn't take cpu as param
            if device.type == "cuda": kwargs["device"] = device.type
            with safetensors.safe_open(model_path, **kwargs) as f:
                sd = {}
                for k in f.keys():
                    tensor = f.get_tensor(k)    # loading one key at a time; low mem pressure
                    if global_config.disable_mmap:
                        # moving to device (no zero copying)
                        tensor = tensor.to(device=device, copy=True)
                    sd[k] = tensor
        except Exception as e:
            app_logger.error(str(e))
            raise e
    else:   # ckpt, pth, pt
        torch_args = {}
        # using simple flags rn, will fix later
        if not global_config.disable_mmap: torch_args["mmap"] = True
        if global_config.always_safe_load: torch_args["weights_only"] = True
        
        sd = torch.load(model_path, map_location=device, **torch_args)
        if "state_dict" in sd:  
            sd = sd["state_dict"]   # loading state_dict if available
        elif len(sd) == 1:          # loading the first key (if it's a dict)
            val = next(iter(sd.values()))
            sd = val if isinstance(val, dict) else sd
            
    return sd