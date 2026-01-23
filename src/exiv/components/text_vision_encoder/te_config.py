
from dataclasses import dataclass


@dataclass
class TEConfig:
    default_model_filename = None


@dataclass
class T5Config(TEConfig):
    d_ff = 3072
    d_kv = 64
    d_model = 768
    decoder_start_token_id = 0
    dropout_rate = 0.1
    eos_token_id = 1
    dense_act_fn = "relu"
    initializer_factor = 1.0
    is_encoder_decoder = True
    is_decoder = False
    is_gated_act = False
    layer_norm_epsilon = 1e-06
    model_type = "t5"
    num_decoder_layers = 12
    num_heads = 12
    num_layers = 12
    output_past = True
    pad_token_id = 0
    relative_attention_num_buckets = 32
    relative_attention_max_distance = 128
    tie_word_embeddings = False
    vocab_size = 32128
    
@dataclass
class T5XXLConfig(TEConfig):
    d_ff = 10240
    d_kv = 64
    d_model = 4096
    decoder_start_token_id = 0
    dropout_rate = 0.1
    eos_token_id = 1
    dense_act_fn = "gelu_pytorch_tanh"
    initializer_factor = 1.0
    is_encoder_decoder = True
    is_decoder = False
    is_gated_act = True
    layer_norm_epsilon = 1e-06
    model_type = "t5_xxl"
    num_decoder_layers = 24
    num_heads = 64
    num_layers = 24
    output_past = True
    pad_token_id = 0
    relative_attention_num_buckets = 32
    relative_attention_max_distance = 128
    tie_word_embeddings = False
    vocab_size = 3212

@dataclass
class UMT5XXLConfig(T5XXLConfig):
    # default init config -----
    default_model_filename = "umt5_xxl_fp16.safetensors"
    # model config ------
    model_type = "umt5_xxl"
    vocab_size = 256384