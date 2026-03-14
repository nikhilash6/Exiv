import torch
import torch.nn as nn
from .logits_processing import TemperatureLogitsProcessor, TopKLogitsProcessor, TopPLogitsProcessor

class AutoregressiveSampler(nn.Module):
    def __init__(self, max_tokens=2048, temperature=1.0, top_k=50, top_p=0.9):
        super().__init__()
        self.max_tokens = max_tokens
        
        # Build processing pipeline
        self.processors = []
        if temperature != 1.0:
            self.processors.append(TemperatureLogitsProcessor(temperature))
        if top_k > 0:
            self.processors.append(TopKLogitsProcessor(top_k))
        if top_p < 1.0:
            self.processors.append(TopPLogitsProcessor(top_p))

    def pre_logit_process(self, next_token_logits, current_tokens, step):
        """Hook injection point before applying temp/top_p"""
        return next_token_logits

    def post_sample_step(self, next_token, current_tokens, step):
        """Hook injection point after sampling"""
        return next_token

    def sample(self, model_mixin, prompt_tokens: torch.Tensor, cond: dict = None):
        current_tokens = prompt_tokens
        kv_cache = None
        
        for step in range(self.max_tokens):
            
            # 1. Forward Pass
            logits, kv_cache = model_mixin(
                input_tokens=current_tokens, 
                cond=cond, 
                past_key_values=kv_cache
            )
            
            # Extract only the newly predicted token's logits
            next_token_logits = logits[:, -1, :] 

            # 2. Hook point for logic injection
            next_token_logits = self.pre_logit_process(next_token_logits, current_tokens, step)

            # 3. Apply Processors
            for processor in self.processors:
                next_token_logits = processor(current_tokens, next_token_logits)
            
            # 4. Sample the token
            probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # 5. Hook point for logic injection
            next_token = self.post_sample_step(next_token, current_tokens, step)
            
            # 6. Append to sequence
            current_tokens = torch.cat([current_tokens, next_token], dim=-1)
            
            # 7. Check for End Of Sequence
            if next_token.item() == model_mixin.eos_token_id:
                break
                
        return current_tokens