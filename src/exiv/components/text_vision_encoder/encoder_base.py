import torch
from torch import Tensor

from dataclasses import dataclass

from ...utils.device import ProcDevice
from ...model_utils.model_mixin import ModelMixin

# this is the base of all the encoder models like T5 and CLIP
class TextEncoder(ModelMixin):
    def __init__(self, model_path, config, te_type):
        self.model_path = model_path
        self.config = config
        self.te_type = te_type
        super().__init__(ProcDevice.CUDA.value, None, model_path)
    
    # TODO: check if this can be diff for diff encoder archs
    def gen_empty_tokens(self, special_tokens, length):
        start_token = special_tokens.get("start", None)
        end_token = special_tokens.get("end", None)
        pad_token = special_tokens.get("pad")
        output = []
        if start_token is not None:
            output.append(start_token)
        if end_token is not None:
            output.append(end_token)
        output += [pad_token] * (length - len(output))
        return output
    
    def encode_token_weights(self, token_weight_pairs):
        to_encode = list()
        max_token_len = 0
        has_weights = False
        
        for batch in token_weight_pairs:
            # batch - list of (token, weight) pairs
            tokens = list(map(lambda a: a[0], batch))
            max_token_len = max(len(tokens), max_token_len)     # longest seq among diff batches
            has_weights = has_weights or \
                not all(map(lambda a: a[1] == 1.0, batch))      # atleast one token has custom weights
            to_encode.append(tokens)

        batch_count = len(token_weight_pairs)   # total num. of batches

        if has_weights or batch_count == 0:
            # has_weights == True : need a neutral ref point -> empty tokens
            # batch_count == 0 : empty token list
            to_encode.append(self.gen_empty_tokens(self.special_tokens, max_token_len))

        # main encoding
        o = self.forward(to_encode)
        # `out` contains the embeddings for each token.
        # `pooled` is a single summary embedding for the whole sequence (e.g., from a [CLS] token).
        out, pooled = o[:2]

        # pooled only makes sense for clip models
        if pooled is not None:
            first_pooled = pooled[0:1].to(ProcDevice.CPU.value)
        else:
            # for T5 this would be intermediate_output (None)
            first_pooled = pooled

        # apply weights
        # current out: [batch_size + 1, seq_len, embed_dim] , extra dim for the empty ref
        output = []
        for k in range(0, batch_count):
            z = out[k:k+1]          # [1, seq_len, embed_dim]
            
            if has_weights:
                z_empty = out[-1]   # empty embed added above
                for j in range(len(z[0])):      # seq_len
                    weight = token_weight_pairs[k][j][1]
                    if weight != 1.0:
                        z[0][j] = (z[0][j] - z_empty[j]) * weight + z_empty[j]
            output.append(z)

        if (len(output) == 0):
            # returning the empty neutral
            r = (out[-1:].to(ProcDevice.CPU.value), first_pooled)
        else:
            # concatenating all the processed (and weighted) embeddings together
            r = (torch.cat(output, dim=-2).to(ProcDevice.CPU.value), first_pooled)

        # extra data (like an attention mask), process and append it to the output
        if len(o) > 2:
            extra = {}
            for k in o[2]:
                v = o[2][k]
                if k == "attention_mask":   # TODO: check this while running
                    v = v[:batch_count].flatten().unsqueeze(dim=0).to(ProcDevice.CPU.value)
                extra[k] = v

            r = r + (extra,)
        return r
        

class VisionEncoder(ModelMixin):
    def __init__(self, model_path, config):
        self.model_path = model_path
        self.config = config
        super().__init__(ProcDevice.CUDA.value, None, model_path)
        
    def encode_image(self, img: Tensor):
        pass


@dataclass
class T5Config:
    d_ff = 3072,
    d_kv = 64,
    d_model = 768,
    decoder_start_token_id = 0,
    dropout_rate = 0.1,
    eos_token_id = 1,
    dense_act_fn = "relu",
    initializer_factor = 1.0,
    is_encoder_decoder = True,
    is_decoder = False,
    is_gated_act = False,
    layer_norm_epsilon = 1e-06,
    model_type = "t5",
    num_decoder_layers = 12,
    num_heads = 12,
    num_layers = 12,
    output_past = True,
    pad_token_id = 0,
    relative_attention_num_buckets = 32,
    relative_attention_max_distance = 128,
    tie_word_embeddings = False,
    vocab_size = 32128
    
@dataclass
class T5XXLConfig:
    d_ff = 10240
    d_kv = 64
    d_model = 4096
    decoder_start_token_id = 0
    dropout_rate = 0.1
    eos_token_id = 1
    dense_act_fn = "gelu_pytorch_tanh"
    initializer_factor = 1.0
    is_encoder_decoder = True
    is_gated_act = True
    layer_norm_epsilon = 1e-06
    model_type = "t5"
    num_decoder_layers = 24
    num_heads = 64
    num_layers = 24
    output_past = True
    pad_token_id = 0
    relative_attention_num_buckets = 32
    tie_word_embeddings = False
    vocab_size = 3212
