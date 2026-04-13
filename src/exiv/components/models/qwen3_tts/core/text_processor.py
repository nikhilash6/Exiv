from typing import Union, List


class Qwen3TTSTextProcessor:
    def __init__(self, tokenizer=None, chat_template=None):
        self.tokenizer = tokenizer
        self.chat_template = chat_template

    def __call__(self, text: Union[str, List[str]] = None, **kwargs) -> dict:
        if text is None:
            raise ValueError("You need to specify a `text` input to process.")
        
        text_kwargs = {"padding": False, "padding_side": "left"}
        if "text_kwargs" in kwargs:
            text_kwargs.update(kwargs.pop("text_kwargs"))
        text_kwargs.update(kwargs)

        if not isinstance(text, list):
            text = [text]

        texts_inputs = self.tokenizer(text, **text_kwargs)
        return dict(texts_inputs)

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    def apply_chat_template(self, conversations, chat_template=None, **kwargs):
        if isinstance(conversations, dict):
            conversations = [conversations]
        return self.tokenizer.apply_chat_template(
            conversations,
            chat_template=chat_template or self.chat_template,
            **kwargs
        )
        
    def build_assistant_text(self, text: str) -> str:
        return f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"

    def build_ref_text(self, text: str) -> str:
        return f"<|im_start|>assistant\n{text}<|im_end|>\n"

    def build_instruct_text(self, instruct: str) -> str:
        return f"<|im_start|>user\n{instruct}<|im_end|>\n"
