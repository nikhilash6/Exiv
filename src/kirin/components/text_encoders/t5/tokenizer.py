# code adapted from Huggingface transformer's library
import sentencepiece as spm

from ...utils.common_struct import AddedToken


class T5Tokenizer:
    def __init__(
        self,
        vocab_file,
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        extra_ids=100,
        additional_special_tokens=None,
        add_prefix_space=True,
        **kwargs,
    ):
        pad_token = AddedToken(pad_token, special=True) if isinstance(pad_token, str) else pad_token
        unk_token = AddedToken(unk_token, special=True) if isinstance(unk_token, str) else unk_token
        eos_token = AddedToken(eos_token, special=True) if isinstance(eos_token, str) else eos_token
        
        self.vocab_file = vocab_file
        self._extra_ids = extra_ids
        
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.Load(vocab_file)
        
        if additional_special_tokens is not None:
            extra_tokens = [x for x in additional_special_tokens if "<extra_id_" in str(x)]
            if len(extra_tokens) < 1:
                additional_special_tokens += [f"<extra_id_{i}>" for i in range(extra_ids)]
            elif extra_ids > 0 and extra_ids != len(extra_tokens):
                raise ValueError(
                    f"Both extra_ids ({extra_ids}) and additional_special_tokens ({additional_special_tokens}) are"
                    " provided to T5Tokenizer. In this case the additional_special_tokens must include the extra_ids"
                    " tokens"
                )
        else:
            extra_tokens = [f"<extra_id_{i}>" for i in range(extra_ids)]
            additional_special_tokens = extra_tokens
            
        self.sp_model = self.get_spm_processor(kwargs.pop("from_slow", False))
        self.add_prefix_space = add_prefix_space
        
    def get_spm_processor(self, from_slow=False):
        pass
    
    def tokenize(self, text: str, **kwargs) -> list[str]:
        pass