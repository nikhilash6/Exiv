import torch
import torch.nn as nn
from abc import abstractmethod
from typing import Optional, Tuple, Dict, Any

from .helper_methods import get_state_dict, set_module_tensor_to_device
from ..utils.common import get_module_from_name
from ..utils.dtype import cast_to
from ..utils.device import VRAM_DEVICE, ProcDevice
from ..utils.logging import app_logger
from ..quantizers.base import QuantType, Quantizer, get_quantizer
from ..model_patching.hook_registry import HookLocation
from ..utils.file import ensure_model_availability
from ..components.models.common import AROutput


# TODO: keeping separate from the diffusion meta, maybe these will be merged in the future
class ARModuleMeta(type(nn.Module)):
    def __call__(cls, *args, **kwargs):
        model_dtype = kwargs.pop("dtype", torch.float32)
        quant_type = kwargs.get("quant_type", None)
        quant_config = kwargs.get("quant_config", None)
        force_load_mode = kwargs.get("force_load_mode", None)
        meta_device = kwargs.pop("meta_device", "meta")
        original_dtype = torch.get_default_dtype()
        
        try:
            torch.set_default_dtype(model_dtype)
            
            # zero init weight load
            with torch.device(meta_device):
                # Put dtype back into kwargs so __init__ receives it for nested model creation
                kwargs["dtype"] = model_dtype
                instance = super().__call__(*args, **kwargs)
                quantizer: Quantizer = get_quantizer(quant_type=quant_type, quant_config=quant_config)
                instance.quantizer = quantizer
                if isinstance(instance, ARModelMixin):    # mainly for safety
                    instance.force_load_mode = getattr(instance, "force_load_mode", None) or force_load_mode
                    if not getattr(instance, 'dtype', None):
                        instance.dtype = model_dtype
                        
        finally:
            torch.set_default_dtype(original_dtype)
        
        return instance


class ARModelArchConfig:
    def __init__(self, model_type=None):
        self.model_type = model_type
        # TODO: add more specific attributes and methods


class ARModelMixin(nn.Module, metaclass=ARModuleMeta):
    """
    - (TODO) support quantizers
    - (TODO) support GGUF loading
    
    Base mixin for autoregressive models
    
    Provides:
    - Standardized __call__ wrapper with device/dtype handling and hooks
    - Default generate() implementation that loops calling forward()
    - Model loading utilities
    
    Subclasses MUST implement:
    - forward(): Single step forward pass returning AROutput
    
    Subclasses CAN override:
    - generate(): For custom generation logic (e.g., multi-model orchestration)
    """
    
    def __init__(self, device: str = None, quant_type: QuantType = None, model_path: str = None, dtype = torch.float32, **kwargs):
        super().__init__()
        
        self.gpu_device = device or VRAM_DEVICE
        self.model_path = model_path
        self.model_arch_config = None
        
    def __call__(self, *args, **kwargs):
        """
        Wrapper that handles device/dtype casting and hooks.
        
        Calls the actual forward() implementation.
        """
        def original_call(*args, **kwargs):
            with torch.no_grad():
                app_logger.debug(f"moving the inputs to {self.gpu_device} and dtype {self.dtype}")
                new_args = tuple(cast_to(a, device=self.gpu_device, dtype=self.dtype) if torch.is_tensor(a) and idx == 0 else a for idx, a in enumerate(args))
                new_kwargs = {k: (cast_to(v, device=self.gpu_device, dtype=self.dtype) if torch.is_tensor(v) else v) for k, v in kwargs.items()}
                
                return super(ARModelMixin, self).__call__(*new_args, **new_kwargs)
            
        registry = getattr(self, "hook_registry", None)
        if registry and registry.head.next_hook != registry.tail:
            wrapped_call = registry.get_wrapped_fn(original_call, location=HookLocation.MODEL_RUN.value)
            return wrapped_call(*args, **kwargs)
        else:
            return original_call(*args, **kwargs)
    
    @abstractmethod
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Any] = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs
    ) -> AROutput:
        """
        Single forward step. Must be implemented by subclasses
        
        Args:
            input_ids: Input token IDs [batch, seq_len]
            inputs_embeds: Input embeddings (alternative to input_ids)
            past_key_values: Cached KV from previous steps
            attention_mask: Attention mask
            use_cache: Whether to return KV cache
            
        Returns:
            AROutput with:
            - logits: [batch, seq_len, vocab_size]
            - past_key_values: Updated cache (if use_cache=True)
            - extra: Dict with any additional outputs
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement forward()")
    
    @torch.inference_mode()
    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        max_new_tokens: int = 100,
        min_new_tokens: int = 0,
        do_sample: bool = True,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.LongTensor:
        """
        Default autoregressive generation.
        
        Loops calling forward() to generate tokens one at a time.
        Can be overridden by subclasses for custom generation logic.
        
        Args:
            input_ids: Initial token IDs [batch, seq_len]
            inputs_embeds: Initial embeddings (alternative to input_ids)
            max_new_tokens: Maximum tokens to generate
            min_new_tokens: Minimum tokens before allowing EOS
            do_sample: If True, sample; if False, greedy
            temperature: Sampling temperature
            top_k: Top-k filtering (0 = disabled)
            top_p: Top-p (nucleus) filtering (1.0 = disabled)
            repetition_penalty: Penalty for repeating tokens
            eos_token_id: Token ID that stops generation
            pad_token_id: Token ID for padding
            attention_mask: Attention mask for initial sequence
            
        Returns:
            Generated token IDs [batch, seq_len + generated]
        """
        # Initialize
        batch_size = inputs_embeds.shape[0] if inputs_embeds is not None else input_ids.shape[0]
        device = inputs_embeds.device if inputs_embeds is not None else input_ids.device
        
        # Track generated tokens
        generated_ids = input_ids.clone() if input_ids is not None else None
        past_key_values = None
        
        # Generation loop
        for step in range(max_new_tokens):
            # Prepare inputs
            if step == 0:
                # First step: use full inputs
                model_inputs = {
                    "input_ids": input_ids,
                    "inputs_embeds": inputs_embeds,
                    "attention_mask": attention_mask,
                    "use_cache": True,
                }
            else:
                # Subsequent steps: only use last token
                model_inputs = {
                    "input_ids": next_token,
                    "past_key_values": past_key_values,
                    "use_cache": True,
                }
            
            # Remove None values
            model_inputs = {k: v for k, v in model_inputs.items() if v is not None}
            
            # Forward pass
            outputs = self(**model_inputs)
            
            # Get logits for last position
            next_token_logits = outputs.logits[:, -1, :].clone()
            past_key_values = outputs.past_key_values
            
            # Apply repetition penalty
            if repetition_penalty != 1.0 and generated_ids is not None:
                for token_id in set(generated_ids[0].tolist()):
                    next_token_logits[0, token_id] /= repetition_penalty
            
            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            
            # Top-k filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Top-p filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample
            if do_sample:
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Append to sequence
            if generated_ids is None:
                generated_ids = next_token
            else:
                generated_ids = torch.cat([generated_ids, next_token], dim=1)
            
            # Check for EOS (after min_new_tokens)
            if step >= min_new_tokens and eos_token_id is not None:
                if (next_token == eos_token_id).all():
                    break
        
        return generated_ids
    
    def load_model(
        self,
        model_path = None,
        force_download = False,
        download_url = None,
        dtype = None,
        model_type = None
    ):
        model_path = model_path or self.model_path
        assert model_path is not None, "model_path is required"
        device = ProcDevice.CPU.value
        self.dtype = dtype or self.dtype
        
        # TODO: add gguf and other quantizers support
        
        model_path = ensure_model_availability(model_path, download_url, force_download)
        state_dict = get_state_dict(model_path, model_type=model_type)
        model_state_dict = self.state_dict()
        for param_name, param in state_dict.items():
            if param_name not in model_state_dict: 
                app_logger.warning(f"Skipping param {param_name} as it's not present in the model definition")
                continue
            
            parent_module, leaf_name = get_module_from_name(self, param_name)
            param = param.to(self.dtype)    # can be removed
            old_param = getattr(parent_module, leaf_name, None)
            # skipping buffer / non-loadable entities
            if not isinstance(old_param, (torch.nn.Parameter, torch.Tensor)):
                old_param = None
                
            if old_param is not None:
                if self.dtype is None:
                    param = param.to(old_param.dtype)
                    
                if old_param.is_contiguous():
                    param = param.contiguous()
                    
            if model_state_dict[param_name].shape != param.shape:
                raise ValueError(
                        f"Cannot load {model_path} because {param_name} expected shape {model_state_dict[param_name].shape}, but got {param.shape}."
                    )

            set_module_tensor_to_device(self, param_name, device, value=param, dtype=self.dtype)
        
        del state_dict
        
        # Clean up any leftover meta tensors (e.g. non-persistent parameters not in state_dict)
        for name, param in self.named_parameters():
            if param.is_meta:
                parent_module, leaf_name = get_module_from_name(self, name)
                if parent_module is not None:
                    setattr(parent_module, leaf_name, torch.nn.Parameter(torch.zeros_like(param, device=device, dtype=self.dtype)))

        # Also clean up buffers
        for name, buf in self.named_buffers():
            if buf.is_meta:
                parent_module, leaf_name = get_module_from_name(self, name)
                if parent_module is not None:
                    setattr(parent_module, leaf_name, torch.zeros_like(buf, device=device, dtype=buf.dtype))
                    
        app_logger.info("Autoregressive Model successfully loaded to CPU.")
