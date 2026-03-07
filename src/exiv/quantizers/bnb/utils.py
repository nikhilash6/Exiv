# Copyright 2024 The HuggingFace Team. All rights reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Note: This file was modified from its original version in the HuggingFace 
# Diffusers/Transformers library to adapt it to the Kirin architecture.

import torch
from torch import nn

import inspect
from inspect import signature
from transformers import is_bitsandbytes_available

from ...utils.logging import app_logger

if is_bitsandbytes_available():
    import bitsandbytes as bnb


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

                if quantization_config.quantization_dtype == "llm_int8":
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
