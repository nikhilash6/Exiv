import torch
from torch import nn

import inspect
from inspect import signature
from transformers import is_bitsandbytes_available

from ...utils.logging import app_logger

if is_bitsandbytes_available():
    import bitsandbytes as bnb


# this code has been adapted from Huggingface Diffusers

def _replace_with_bnb_linear(
    model,
    modules_to_not_convert=None,
    current_key_name=None,
    quantization_config=None,
    has_been_replaced=False,
):
    """
    Private method that wraps the recursion for module replacement.

    Returns the converted model and a boolean that indicates if the conversion has been successful or not.
    """
    for name, module in model.named_children():
        if current_key_name is None:
            current_key_name = []
        current_key_name.append(name)

        if isinstance(module, nn.Linear) and name not in modules_to_not_convert:
            # Check if the current key is not in the `modules_to_not_convert`
            current_key_name_str = ".".join(current_key_name)
            if not any(
                (key + "." in current_key_name_str) or (key == current_key_name_str) for key in modules_to_not_convert
            ):
                in_features = module.in_features
                out_features = module.out_features

                if quantization_config.quantization_method() == "llm_int8":
                    model._modules[name] = bnb.nn.Linear8bitLt(
                        in_features,
                        out_features,
                        module.bias is not None,
                        has_fp16_weights=quantization_config.llm_int8_has_fp16_weight,
                        threshold=quantization_config.llm_int8_threshold,
                    )
                    has_been_replaced = True
                else:
                    if (
                        quantization_config.llm_int8_skip_modules is not None
                        and name in quantization_config.llm_int8_skip_modules
                    ):
                        pass
                    else:
                        extra_kwargs = (
                            {"quant_storage": quantization_config.bnb_4bit_quant_storage}
                            if "quant_storage" in list(signature(bnb.nn.Linear4bit).parameters)
                            else {}
                        )
                        model._modules[name] = bnb.nn.Linear4bit(
                            in_features,
                            out_features,
                            module.bias is not None,
                            quantization_config.bnb_4bit_compute_dtype,
                            compress_statistics=quantization_config.bnb_4bit_use_double_quant,
                            quant_type=quantization_config.bnb_4bit_quant_type,
                            **extra_kwargs,
                        )
                        has_been_replaced = True
                
                # Store the module class in case we need to transpose the weight later
                model._modules[name].source_cls = type(module)
                # Force requires grad to False to avoid unexpected errors
                model._modules[name].requires_grad_(False)
        
        if len(list(module.children())) > 0:
            _, has_been_replaced = _replace_with_bnb_linear(
                module,
                modules_to_not_convert,
                current_key_name,
                quantization_config,
                has_been_replaced=has_been_replaced,
            )
        
        # Remove the last key for recursion
        current_key_name.pop(-1)
    return model, has_been_replaced


def replace_with_bnb_linear(model, modules_to_not_convert=None, current_key_name=None, quantization_config=None):
    model, _ = _replace_with_bnb_linear(model, modules_to_not_convert, current_key_name, quantization_config)

    has_been_replaced = any(
        isinstance(replaced_module, (bnb.nn.Linear4bit, bnb.nn.Linear8bitLt))
        for _, replaced_module in model.named_modules()
    )
    if not has_been_replaced:
        app_logger.warning("You are loading your model in 8bit or 4bit but no linear modules were found in your model.")

    return model


# Adapted from PEFT: https://github.com/huggingface/peft/blob/6d458b300fc2ed82e19f796b53af4c97d03ea604/src/peft/utils/integrations.py#L81
def dequantize_bnb_weight(weight: "torch.nn.Parameter", state=None, dtype: "torch.dtype" = None):
    """
    Helper function to dequantize 4bit or 8bit bnb weights.

    If the weight is not a bnb quantized weight, it will be returned as is.
    """
    if not isinstance(weight, torch.nn.Parameter):
        raise TypeError(f"Input weight should be of type nn.Parameter, got {type(weight)} instead")

    cls_name = weight.__class__.__name__
    if cls_name not in ("Params4bit", "Int8Params"):
        return weight

    if cls_name == "Params4bit":
        output_tensor = bnb.functional.dequantize_4bit(weight.data, weight.quant_state)
        msg = f"The model is going to be dequantized in {output_tensor.dtype} - if you want to upcast it to another dtype, make sure to pass the desired dtype when quantizing the model through `bnb_4bit_quant_type` argument of `BitsAndBytesConfig`"
        if dtype:
            msg = f"The model is going to be first dequantized in {output_tensor.dtype} and type-casted to {dtype}"
            output_tensor = output_tensor.to(dtype)
        app_logger.warning_once(msg)
        return output_tensor

    if state.SCB is None:
        state.SCB = weight.SCB

    if hasattr(bnb.functional, "int8_vectorwise_dequant"):
        # Use bitsandbytes API if available (requires v0.45.0+)
        dequantized = bnb.functional.int8_vectorwise_dequant(weight.data, state.SCB)
    else:
        # Multiply by (scale/127) to dequantize.
        dequantized = weight.data * state.SCB.view(-1, 1) * 7.874015718698502e-3

    if dtype:
        dequantized = dequantized.to(dtype)
    return dequantized



def _dequantize_and_replace(
    model,
    dtype,
    modules_to_not_convert=None,
    current_key_name=None,
    quantization_config=None,
    has_been_replaced=False,
):
    """
    Converts a quantized model into its dequantized original version. The newly converted model will have some
    performance drop compared to the original model before quantization - use it only for specific usecases such as
    QLoRA adapters merging.

    Returns the converted model and a boolean that indicates if the conversion has been successful or not.
    """
    quant_method = quantization_config.quantization_method()

    target_cls = bnb.nn.Linear8bitLt if quant_method == "llm_int8" else bnb.nn.Linear4bit

    for name, module in model.named_children():
        if current_key_name is None:
            current_key_name = []
        current_key_name.append(name)

        if isinstance(module, target_cls) and name not in modules_to_not_convert:
            # Check if the current key is not in the `modules_to_not_convert`
            current_key_name_str = ".".join(current_key_name)

            if not any(
                (key + "." in current_key_name_str) or (key == current_key_name_str) for key in modules_to_not_convert
            ):
                bias = getattr(module, "bias", None)

                device = module.weight.device
                new_module = torch.nn.Linear(module.in_features, module.out_features, bias=bias is not None, device=torch.device("meta"))

                if quant_method == "llm_int8":
                    state = module.state
                else:
                    state = None

                new_module.weight = torch.nn.Parameter(dequantize_bnb_weight(module.weight, state, dtype))

                if bias is not None:
                    new_module.bias = bias

                new_module.to(device)
                model._modules[name] = new_module
                has_been_replaced = True
        
        if len(list(module.children())) > 0:
            _, has_been_replaced = _dequantize_and_replace(
                module,
                dtype=dtype,
                modules_to_not_convert=modules_to_not_convert,
                current_key_name=current_key_name,
                quantization_config=quantization_config,
                has_been_replaced=has_been_replaced,
            )
        # Remove the last key for recursion
        current_key_name.pop(-1)
    
    return model, has_been_replaced


def dequantize_and_replace(
    model,
    modules_to_not_convert=None,
    quantization_config=None,
):
    model, _ = _dequantize_and_replace(
        model,
        dtype=model.dtype,
        modules_to_not_convert=modules_to_not_convert,
        quantization_config=quantization_config,
    )
    has_been_replaced = any(
        isinstance(replaced_module, torch.nn.Linear) for _, replaced_module in model.named_modules()
    )
    if not has_been_replaced:
        app_logger.warning(
            "Some linear modules were not dequantized. This could lead to unexpected behaviour. Please check your model."
        )

    return model
