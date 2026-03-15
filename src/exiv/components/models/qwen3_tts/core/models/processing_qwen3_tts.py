# Copyright 2026 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
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

import torch
from typing import List, Optional, Union, Dict, Any
from collections import UserDict


class BatchFeature(UserDict):
    r"""
    A dictionary-like object that holds the outputs of the processor and provides
    convenience methods like `.to(device)`.
    """
    def __init__(self, data: Dict[str, Any] = None, tensor_type: Optional[str] = None):
        super().__init__(data or {})
        self.tensor_type = tensor_type

    def to(self, *args, **kwargs) -> "BatchFeature":
        r"""
        Moves all tensors in the batch to the specified device.
        """
        new_data = {}
        for k, v in self.data.items():
            if hasattr(v, "to"):
                new_data[k] = v.to(*args, **kwargs)
            else:
                new_data[k] = v
        self.data = new_data
        return self

    def __getattr__(self, item):
        try:
            return self.data[item]
        except KeyError:
            raise AttributeError(f"'BatchFeature' object has no attribute '{item}'")


class Qwen3TTSProcessor:
    r"""
    Constructs a Qwen3TTS processor.

    Args:
        tokenizer:
            The text tokenizer (e.g., Qwen2Tokenizer).
        chat_template (`str`, *optional*):
            The Jinja template to use for formatting the conversation.
    """

    attributes = ["tokenizer"]
    tokenizer_class = ("Qwen2Tokenizer", "Qwen2TokenizerFast")

    def __init__(self, tokenizer=None, chat_template=None):
        self.tokenizer = tokenizer
        self.chat_template = chat_template

    def __call__(self, text: Union[str, List[str]] = None, **kwargs) -> BatchFeature:
        r"""
        Main method to prepare text sequences for the model.
        """
        if text is None:
            raise ValueError("You need to specify a `text` input to process.")

        # Default text kwargs
        text_kwargs = {
            "padding": False,
            "padding_side": "left",
        }
        
        # Extract return_tensors if present
        return_tensors = kwargs.pop("return_tensors", None)
        
        # Handle text_kwargs if nested
        if "text_kwargs" in kwargs:
            text_kwargs.update(kwargs.pop("text_kwargs"))
        
        # Merge remaining kwargs into text_kwargs
        text_kwargs.update(kwargs)

        if not isinstance(text, list):
            text = [text]

        texts_inputs = self.tokenizer(text, **text_kwargs)

        return BatchFeature(
            data=dict(texts_inputs),
            tensor_type=return_tensors,
        )

    def batch_decode(self, *args, **kwargs):
        r"""
        Forwards all arguments to the tokenizer's `batch_decode`.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        r"""
        Forwards all arguments to the tokenizer's `decode`.
        """
        return self.tokenizer.decode(*args, **kwargs)

    def apply_chat_template(self, conversations, chat_template=None, **kwargs):
        r"""
        Apply a chat template to the conversation(s).
        """
        if isinstance(conversations, dict):
            conversations = [conversations]
        
        # Some processors wrap single conversation in a list twice for batch processing
        if isinstance(conversations, list) and len(conversations) > 0 and isinstance(conversations[0], dict):
             # Ensure it's a list of conversations (list of list of dicts) if tokenizer expects that
             # but usually tokenizer.apply_chat_template handles list of dicts as one conversation.
             pass

        return self.tokenizer.apply_chat_template(
            conversations,
            chat_template=chat_template or self.chat_template,
            **kwargs
        )

    @property
    def model_input_names(self):
        r"""
        Returns the input names required by the model, inferred from the tokenizer.
        """
        tokenizer_input_names = getattr(self.tokenizer, "model_input_names", [])
        return list(dict.fromkeys(tokenizer_input_names))

