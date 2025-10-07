import torch

# this is the base of all the encoder models like T5 and CLIP
class TextEncoderModel(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        pass
    
    def encode_token_weights(self, token_weight_pairs):
        pass
    
    def load_model(self):
        pass