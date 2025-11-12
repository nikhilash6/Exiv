import torch
from torch import Tensor

from dataclasses import dataclass

from ...utils.device import OFFLOAD_DEVICE, VRAM_DEVICE, ProcDevice
from ...model_utils.model_mixin import ModelMixin
from ...utils.logging import app_logger

# this is the base of all the encoder models like T5 and CLIP
class TextEncoder(ModelMixin):
    def __init__(self, model_path, config, te_type, **kwargs):
        self.model_path = model_path
        self.config = config
        self.te_type = te_type
        
        # not all TEs use attn mask
        self.enable_attention_masks = kwargs.get("enable_attention_masks", False)
        self.layer = kwargs.get("layer", "last")        # ['last', 'hidden']
        self.layer_idx = kwargs.get("layer_idx", -2)
        self.zero_out_masked = kwargs.get("zero_out_masked", False)     # zero out padding / eos stuff
        super().__init__(model_path=model_path)
    
    def gen_empty_tokens(self, length, special_tokens: tuple):
        start_token, end_token, pad_token = special_tokens
        assert pad_token is not None, "pad token can't be None"
        
        output = []
        if start_token is not None:
            output.append(start_token)
        if end_token is not None:
            output.append(end_token)
        output += [pad_token] * (length - len(output))
        return output
    
    def process_tokens(self, tokens: list[list[int]], special_tokens: tuple, device):
        # tokens = [[123, 1234, 12, ...], [1, 1, 1, ..]]
        start_token, end_token, pad_token = special_tokens
        cmp_token = end_token or pad_token      # to determine where the 'real' sequence ends
                                                # triggers eos flag

        embeds_out = []
        attention_masks = []
        num_tokens = []

        for seq in tokens:
            attention_mask = []
            tokens_temp = []
            custom_embeds = []
            eos = False         # eos flag
            index = 0
            
            # ------ separating normal and custom tokens/embeds
            for idx, token_id in enumerate(seq):
                if isinstance(token_id, int):
                    # int token_id
                    
                    # 0 -> masked out, 1 -> left in the seq
                    # if we are *past* the end token, mask this token out
                    if eos:
                        attention_mask.append(0)
                    else:
                        attention_mask.append(1)

                    tokens_temp += [token_id]
                    if not eos and token_id == cmp_token:
                        if end_token is None:
                            attention_mask[-1] = 0
                        eos = True
                else:
                    # custom embed from text embedding
                    custom_embeds.append((idx, token_id))

            
            # ------ creating embeddings
            tokens_embed = torch.tensor([tokens_temp], device=device, dtype=torch.long)
            # 'tokens_embed' is now a 3D tensor: (1, num_tokens_temp, embed_dim)
            # TODO: handle dtype - out_dtype=torch.float32
            input_embedding = self.get_input_embeddings()
            input_embedding.to(tokens_embed.device)
            tokens_embed = input_embedding(tokens_embed)
            del input_embedding
            
            # ------ inject in custom embeddings
            index = 0
            pad_extra = 0       # counter for padding needed due to *mismatched* embeds
            embeds_info = []    # Info about injected embeddings (for this sequence)
            
            for custom_embed_tuple in custom_embeds:
                emb = custom_embed_tuple[1]
                if torch.is_tensor(emb):
                    emb = {"type": "embedding", "data": emb}

                extra = None
                emb_type = emb.get("type", None)
                
                # handling diff embed types
                if emb_type == "embedding":
                    emb = emb.get("data", None)     # simple text embed
                else:
                    # TODO: see what different types need to be supported
                    emb = None

                if emb is None:
                    # adjust index to account for the removed item
                    index += -1
                    continue

                ind = index + custom_embed_tuple[0]
                # ensure embedding is (1, seq_len, embed_dim) and on the correct device/dtype
                emb = emb.view(1, -1, emb.shape[-1]).to(device=device, dtype=torch.float32)
                emb_shape = emb.shape[1]
                
                # --- final injection
                if emb.shape[-1] == tokens_embed.shape[-1]:
                    tokens_embed = torch.cat(
                        [tokens_embed[:, :ind],
                        emb,
                        tokens_embed[:, ind:]],
                        dim=1
                    )
                    
                    # add '1's for the new embed's length
                    attention_mask = attention_mask[:ind] + [1] * emb_shape + attention_mask[ind:]
                    index += emb_shape - 1
                    embeds_info.append({"type": emb_type, "index": ind, "size": emb_shape, "extra": extra})
                else:
                    # Shape mismatch. Cannot inject.
                    index += -1
                    pad_extra += emb_shape
                    app_logger.warning(f"WARNING: shape mismatch, embedding ignored {emb.shape[-1]} != {tokens_embed.shape[-1]}")

            
            if pad_extra > 0:
                # Get embeddings for 'pad' tokens
                padd_embed = self.transformer.get_input_embeddings()(
                    torch.tensor(
                        [[self.special_tokens["pad"]] * pad_extra], 
                        device=device, 
                        dtype=torch.long
                    ), 
                    out_dtype=torch.float32,
                )
                tokens_embed = torch.cat([tokens_embed, padd_embed], dim=1)
                # don't pay attn to these pads, adding 0
                attention_mask = attention_mask + [0] * pad_extra

            embeds_out.append(tokens_embed)
            attention_masks.append(attention_mask)
            num_tokens.append(sum(attention_mask)) # true, unmasked length

        
        return (
            torch.cat(embeds_out),                                          # (batch_size, max_seq_len, embed_dim)
            torch.tensor(attention_masks, device=device, dtype=torch.long), # (batch_size, max_seq_len)
            num_tokens,
            embeds_info     # TODO: likely a bug, overwritten every loop
        )

    def encode_token_weights(self, token_weight_pairs, special_tokens: tuple):
        to_encode = list()
        max_token_len = 0
        has_weights = False
        
        # --------- prepare tokens --------------
        for batch in token_weight_pairs:
            # batch - list of (token, weight) pairs
            tokens = list(map(lambda a: a[0], batch))
            max_token_len = max(len(tokens), max_token_len)     # longest seq among diff batches
            has_weights = has_weights or \
                not all(map(lambda a: a[1] == 1.0, batch))      # atleast one token has custom weights
            to_encode.append(tokens)

        batch_count = len(token_weight_pairs)   # total num. of batches

        if has_weights or batch_count == 0:
            # Case 1: has_weights == True
            # If any batch had custom weights (e.g., for CFG), a neutral reference point
            # (an empty prompt) is required to calculate the guidance.
            
            # Case 2: batch_count == 0
            # If the input was empty, we still need to encode something (the empty sequence)
            # to avoid errors downstream.
            to_encode.append(self.gen_empty_tokens(max_token_len, special_tokens))

        # ---------------- create token embeds -------------
        embeds, attention_mask, num_tokens, embeds_info = self.process_tokens(to_encode, special_tokens, device=VRAM_DEVICE)

        attention_mask_copy = None
        if self.enable_attention_masks:
            attention_mask_copy = attention_mask

        if self.layer == "last":
            intermediate_output_layer_idx = -1
        else:
            intermediate_output_layer_idx = self.layer_idx
        
        # ---------------- main encoding ----------------
        # TODO: handle dtype fp32
        with torch.autocast(device_type="cuda", dtype=torch.float32):
            outputs = self(
                input_ids=None, 
                attention_mask=attention_mask_copy, 
                input_embeds=embeds, 
                num_tokens=num_tokens, 
                intermediate_output_layer_idx=intermediate_output_layer_idx,
                embeds_info=embeds_info
            )
        
        if self.layer == "last":
            final_output = outputs[0].float()
        else:
            final_output = outputs[1].float()

        if self.zero_out_masked:
            # any token embedding where the mask was 0.0 is multiplied by 0.0
            final_output *= attention_mask.unsqueeze(-1).float()

        pooled_output = None
        if len(outputs) >= 3:
            # output: (final, intermediate, projected_pooled, raw_pooled)
            if not self.return_projected_pooled and len(outputs) >= 4 and outputs[3] is not None:
                pooled_output = outputs[3].float()
            elif outputs[2] is not None:
                pooled_output = outputs[2].float()

        if pooled_output is not None:
            # grabbing the first pooled (from the batch)
            first_pooled = pooled_output[0:1].to(OFFLOAD_DEVICE)
        else:
            # for T5 this would be intermediate_output (None)
            first_pooled = pooled_output

        # ---------------- apply weights ----------------
        # current out: [batch_size + 1, seq_len, embed_dim] , extra dim for the empty ref
        output = []
        for k in range(0, batch_count):
            z = final_output[k:k+1]          # [1, seq_len, embed_dim]
            
            if has_weights:
                z_empty = final_output[-1]   # empty embed added above
                for j in range(len(z[0])):      # seq_len
                    weight = token_weight_pairs[k][j][1]
                    if weight != 1.0:
                        z[0][j] = (z[0][j] - z_empty[j]) * weight + z_empty[j]
            output.append(z)

        if (len(output) == 0):
            # returning the empty neutral
            r = (final_output[-1:].to(OFFLOAD_DEVICE), first_pooled)
        else:
            # concatenating all the processed (and weighted) embeddings together
            r = (torch.cat(output, dim=-2).to(OFFLOAD_DEVICE), first_pooled)

        # extra data (like an attention mask), process and append it to the output
        if len(outputs) > 2:
            extra = {}
            for k in outputs[2]:
                v = outputs[2][k]
                if k == "attention_mask":   # TODO: check this while running
                    v = v[:batch_count].flatten().unsqueeze(dim=0).to(OFFLOAD_DEVICE)
                extra[k] = v

            r = r + (extra,)
        return r
        

class VisionEncoder(ModelMixin):
    def __init__(self, model_path, dtype=None, device=None):
        super().__init__(model_path=model_path, dtype=dtype, device=device)
    
    # common preprocessor for current vision encoders
    def clip_preprocess(self, image: Tensor, crop=True):
        assert self.config is not None, "Vision encoder config not set, unable to proceed"
        
        # setting defaults
        size = self.config.get("image_size", 224)
        mean = self.config.get("image_mean", [0.48145466, 0.4578275, 0.40821073])
        std = self.config.get("image_std", [0.26862954, 0.26130258, 0.27577711])
        
        # normalizing
        image = image[:, :, :, :3] if image.shape[3] > 3 else image
        mean = torch.tensor(mean, device=image.device, dtype=image.dtype)
        std = torch.tensor(std, device=image.device, dtype=image.dtype)
        image = image.movedim(-1, 1)
        if not (image.shape[2] == size and image.shape[3] == size):
            if crop:
                scale = (size / min(image.shape[2], image.shape[3]))
                scale_size = (round(scale * image.shape[2]), round(scale * image.shape[3]))
            else:
                scale_size = (size, size)

            image = torch.nn.functional.interpolate(image, size=scale_size, mode="bicubic", antialias=True)
            h = (image.shape[2] - size)//2
            w = (image.shape[3] - size)//2
            image = image[:,:,h:h+size,w:w+size]
        image = torch.clip((255. * image), 0, 255).round() / 255.0
        return (image - mean.view([3,1,1])) / std.view([3,1,1])
        
    def encode_image(self, image: Tensor, crop=True):
        pixel_values = self.clip_preprocess(image.to(self.gpu_device), crop=crop).float()
        out = self(pixel_values=pixel_values, intermediate_output='all' if self.return_all_hidden_states else -2)

        '''
        image_embeds - final output, designed to match text embeds
        penultimate_hidden_states - richer spatial details, preferred choice for adaptors (controlnets, IPAs)
        mm_projected - llava embedding
        all_hidden_states - full data dump
        '''
        outputs = {}
        outputs["last_hidden_state"] = out[0]
        outputs["image_embeds"] = out[2]
        
        if self.return_all_hidden_states:
            all_hs = out[1]
            outputs["penultimate_hidden_states"] = all_hs[:, -2]
            outputs["all_hidden_states"] = all_hs
        else:
            outputs["penultimate_hidden_states"] = out[1]

        outputs["mm_projected"] = out[3]
        return outputs


@dataclass
class T5Config:
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
    is_decoder = False
    is_gated_act = True
    layer_norm_epsilon = 1e-06
    model_type = "t5"
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
    vocab_size = 256384