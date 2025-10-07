import torch

# this is the base of all the encoder models like T5 and CLIP
class TextEncoderModel(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        pass
    
    def encode_token_weights(self, token_weight_pairs):
        pass
    
    def load_model(self):
        pass
    
class EncoderConfig:
    pass

class T5Config(EncoderConfig):
    d_ff = 3072,
    d_kv = 64,
    d_model = 768,
    decoder_start_token_id = 0,
    dropout_rate = 0.1,
    eos_token_id = 1,
    dense_act_fn = "relu",
    initializer_factor = 1.0,
    is_encoder_decoder = True,
    is_gated_act = False,
    layer_norm_epsilon = 1e-06,
    model_type = "t5",
    num_decoder_layers = 12,
    num_heads = 12,
    num_layers = 12,
    output_past = True,
    pad_token_id = 0,
    relative_attention_num_buckets = 32,
    tie_word_embeddings = False,
    vocab_size = 32128