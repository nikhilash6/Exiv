import torch

class LogitsProcessor:
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        raise NotImplementedError

class TemperatureLogitsProcessor(LogitsProcessor):
    def __init__(self, temperature: float):
        self.temperature = max(temperature, 1e-5) # Prevent division by zero

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.temperature != 1.0:
            scores = scores / self.temperature
        return scores

class TopKLogitsProcessor(LogitsProcessor):
    def __init__(self, top_k: int, filter_value: float = -float("Inf")):
        self.top_k = max(1, top_k)
        self.filter_value = filter_value

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        top_k = min(self.top_k, scores.size(-1))  # Safety check
        # Find the threshold value for the top k tokens
        indices_to_remove = scores < torch.topk(scores, top_k)[0][..., -1, None]
        # Mask out anything below that threshold
        scores = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores

class TopPLogitsProcessor(LogitsProcessor):
    def __init__(self, top_p: float, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
        self.top_p = top_p
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.top_p >= 1.0:
            return scores
            
        sorted_logits, sorted_indices = torch.sort(scores, descending=True)
        cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > self.top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        sorted_indices_to_remove[..., :self.min_tokens_to_keep] = 0
        
        # Scatter the sorted indices back to their original positions
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        scores = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores