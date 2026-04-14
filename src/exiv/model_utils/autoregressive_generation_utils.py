"""
(will refactor later)

This module provides modular components for autoregressive text/audio generation:
- LogitsProcessors: Transform logits before sampling (temperature, top-k, top-p, etc.)
- StoppingCriteria: Determine when to stop generation (EOS, max length, etc.)
- Samplers: Select next token from processed logits (greedy, multinomial)

"""

import torch
from torch import Tensor, LongTensor, FloatTensor, BoolTensor
from abc import ABC, abstractmethod
from typing import Optional, List, Union, Callable


# =============================================================================
# Logits Processors
# =============================================================================

class LogitsProcessor(ABC):
    """
    Abstract base class for all logits processors.
    Example:
        >>> processor = TemperatureLogitsProcessor(temperature=0.8)
        >>> scores = processor(input_ids, scores)
    """
    
    @abstractmethod
    def __call__(self, input_ids: LongTensor, scores: FloatTensor) -> FloatTensor:
        raise NotImplementedError(f"{self.__class__.__name__} must implement __call__")

# added mainly for debug purposes
def apply_logits_processors(
    processors: List[LogitsProcessor],
    input_ids: LongTensor,
    scores: FloatTensor
) -> FloatTensor:
    for processor in processors:
        scores = processor(input_ids, scores)
    return scores


class TemperatureLogitsProcessor(LogitsProcessor):
    """
    apply temperature scaling to logits
    - temperature < 1.0 makes the distribution more peaked (less random)
    - temperature > 1.0 makes the distribution more flat (more random)
    """
    
    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature
    
    def __call__(self, input_ids: LongTensor, scores: FloatTensor) -> FloatTensor:
        if self.temperature == 1.0:
            return scores
        return scores / self.temperature


class TopKLogitsProcessor(LogitsProcessor):
    """
    filter logits to only keep the top k tokens
    """
    
    def __init__(self, top_k: int):
        self.top_k = top_k
    
    def __call__(self, input_ids: LongTensor, scores: FloatTensor) -> FloatTensor:
        if self.top_k <= 0:
            return scores
        
        scores = scores.clone()
        indices_to_remove = scores < torch.topk(scores, self.top_k)[0][..., -1, None]
        scores[indices_to_remove] = float('-inf')
        return scores


class TopPLogitsProcessor(LogitsProcessor):
    """
    nucleus (top-p) filtering: keep the smallest set of tokens whose 
    cumulative probability exceeds p
    """
    
    def __init__(self, top_p: float):
        self.top_p = top_p
    
    def __call__(self, input_ids: LongTensor, scores: FloatTensor) -> FloatTensor:
        if self.top_p >= 1.0:
            return scores
        
        scores = scores.clone()
        sorted_logits, sorted_indices = torch.sort(scores, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        
        sorted_indices_to_remove = cumulative_probs > self.top_p
        # shift to keep at least one token
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False
        
        # back to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        scores[indices_to_remove] = float('-inf')
        return scores


class RepetitionPenaltyLogitsProcessor(LogitsProcessor):
    """
    apply repetition penalty to ALL unique tokens that have already appeared
    - penalty > 1.0 decreases likelihood of repeated tokens
    - penalty < 1.0 increases likelihood of repeated tokens
    """
    
    def __init__(self, penalty: float):
        self.penalty = penalty
    
    def __call__(self, input_ids: LongTensor, scores: FloatTensor) -> FloatTensor:
        if self.penalty == 1.0 or input_ids is None or input_ids.numel() == 0:
            return scores
        
        scores = scores.clone()
        batch_size = input_ids.shape[0]
        
        for b in range(batch_size):
            unique_tokens = set(input_ids[b].tolist())
            for token_id in unique_tokens:
                scores[b, token_id] /= self.penalty
        
        return scores


class LastTokenRepetitionPenaltyLogitsProcessor(LogitsProcessor):
    """
    apply repetition penalty to ONLY the last generated token
    """
    
    def __init__(self, penalty: float):
        self.penalty = penalty
    
    def __call__(self, input_ids: LongTensor, scores: FloatTensor) -> FloatTensor:
        if self.penalty == 1.0 or input_ids is None or input_ids.numel() == 0:
            return scores
        
        scores = scores.clone()
        batch_size = input_ids.shape[0]
        
        # only penalize the LAST token (most recent)
        last_tokens = input_ids[:, -1]
        
        for b in range(batch_size):
            scores[b, last_tokens[b]] /= self.penalty
        
        return scores


class MinNewTokensLogitsProcessor(LogitsProcessor):
    """
    prevent EOS token from being generated until a minimum number of 
    new tokens have been generated
    
    args:
        min_new_tokens: Minimum number of new tokens before EOS is allowed
        eos_token_id: Token ID for end-of-sequence
        prompt_length: Length of the initial prompt (to count only new tokens)
    """
    
    def __init__(self, min_new_tokens: int, eos_token_id: int, prompt_length: int = 0):
        self.min_new_tokens = min_new_tokens
        self.eos_token_id = eos_token_id
        self.prompt_length = prompt_length
    
    def __call__(self, input_ids: LongTensor, scores: FloatTensor) -> FloatTensor:
        if self.min_new_tokens <= 0:
            return scores
        
        current_length = input_ids.shape[1] if input_ids is not None else 0
        new_tokens_generated = current_length - self.prompt_length
        
        if new_tokens_generated < self.min_new_tokens:
            scores = scores.clone()
            scores[:, self.eos_token_id] = float('-inf')
        
        return scores


def build_default_logits_processors(
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    min_new_tokens: int = 0,
    eos_token_id: Optional[int] = None,
    prompt_length: int = 0,
    use_last_token_repetition_penalty: bool = True,
) -> List[LogitsProcessor]:
    """
    build a LogitsProcessorList with the standard set of processors
    
    args:
        temperature: Sampling temperature
        top_k: Top-k filtering (0 = disabled)
        top_p: Top-p filtering (1.0 = disabled)
        repetition_penalty: Repetition penalty (1.0 = no penalty)
        min_new_tokens: Min tokens before EOS allowed (requires eos_token_id)
        eos_token_id: EOS token ID for min_new_tokens enforcement
        prompt_length: Length of prompt for min_new_tokens calculation
        use_last_token_repetition_penalty: If True, use LastTokenRepetitionPenaltyLogitsProcessor
            (Qwen3-TTS style, only penalizes last token). If False, use standard
            RepetitionPenaltyLogitsProcessor (penalizes all unique tokens).
    """
    processors: List[LogitsProcessor] = []
    
    if temperature != 1.0:
        processors.append(TemperatureLogitsProcessor(temperature))
    
    if repetition_penalty != 1.0:
        if use_last_token_repetition_penalty:
            processors.append(LastTokenRepetitionPenaltyLogitsProcessor(repetition_penalty))
        else:
            processors.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))
    
    if min_new_tokens > 0 and eos_token_id is not None:
        processors.append(MinNewTokensLogitsProcessor(min_new_tokens, eos_token_id, prompt_length))
    
    if top_k > 0:
        processors.append(TopKLogitsProcessor(top_k))
    
    if top_p < 1.0:
        processors.append(TopPLogitsProcessor(top_p))
    
    return processors


# =============================================================================
# Stopping Criteria
# =============================================================================

class StoppingCriteria(ABC):
    @abstractmethod
    def __call__(self, input_ids: LongTensor, **kwargs) -> BoolTensor:
        raise NotImplementedError(f"{self.__class__.__name__} must implement __call__")

# mainly for dev
def check_stopping_criteria(
    criteria: List[StoppingCriteria],
    input_ids: LongTensor,
    **kwargs
) -> BoolTensor:
    """
    check all stopping criteria. Generation stops if ANY criterion is met
        
    returns:
        Boolean tensor [batch_size] - True means stop generation for that item
    """
    if len(criteria) == 0:
        return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
    
    # stack all criteria results and take any
    results = torch.stack([criterion(input_ids, **kwargs) for criterion in criteria])
    return results.any(dim=0)


class MaxLengthCriteria(StoppingCriteria):
    """
    stop generation when maximum sequence length is reached
    """
    
    def __init__(self, max_length: int):
        self.max_length = max_length
    
    def __call__(self, input_ids: LongTensor, **kwargs) -> BoolTensor:
        return torch.full(
            (input_ids.shape[0],), 
            input_ids.shape[1] >= self.max_length,
            dtype=torch.bool,
            device=input_ids.device
        )


class MaxNewTokensCriteria(StoppingCriteria):
    """
    stop generation when maximum number of new tokens is reached
    """
    
    def __init__(self, max_new_tokens: int, prompt_length: int):
        self.max_new_tokens = max_new_tokens
        self.prompt_length = prompt_length
    
    def __call__(self, input_ids: LongTensor, **kwargs) -> BoolTensor:
        current_length = input_ids.shape[1]
        new_tokens = current_length - self.prompt_length
        return torch.full(
            (input_ids.shape[0],),
            new_tokens >= self.max_new_tokens,
            dtype=torch.bool,
            device=input_ids.device
        )


class EosTokenCriteria(StoppingCriteria):
    """
    stop generation when EOS token is generated
    """
    
    def __init__(self, eos_token_id: int, min_new_tokens: int = 0, prompt_length: int = 0):
        self.eos_token_id = eos_token_id
        self.min_new_tokens = min_new_tokens
        self.prompt_length = prompt_length
    
    def __call__(self, input_ids: LongTensor, **kwargs) -> BoolTensor:
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        last_tokens = input_ids[:, -1]
        is_eos = last_tokens == self.eos_token_id
        if self.min_new_tokens > 0:
            current_length = input_ids.shape[1]
            new_tokens = current_length - self.prompt_length
            if new_tokens < self.min_new_tokens:
                return torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        return is_eos


def build_default_stopping_criteria(
    max_new_tokens: int,
    prompt_length: int,
    eos_token_id: Optional[int] = None,
    min_new_tokens: int = 0,
) -> List[StoppingCriteria]:
    criteria: List[StoppingCriteria] = []
    
    # NOTE: always stop at max_new_tokens
    criteria.append(MaxNewTokensCriteria(max_new_tokens, prompt_length))
    
    # stop on EOS token (if provided)
    if eos_token_id is not None:
        criteria.append(EosTokenCriteria(eos_token_id, min_new_tokens, prompt_length))
    
    return criteria


# =============================================================================
# Samplers
# =============================================================================

class ARSampler(ABC):
    @abstractmethod
    def __call__(self, logits: FloatTensor) -> LongTensor:
        raise NotImplementedError(f"{self.__class__.__name__} must implement __call__")


class GreedySampler(ARSampler):
    """
    greedy sampling: always pick the highest probability token
    """
    def __call__(self, logits: FloatTensor) -> LongTensor:
        return torch.argmax(logits, dim=-1, keepdim=True)


class MultinomialSampler(ARSampler):
    """
    multinomial sampling: sample from the probability distribution
    """
    def __call__(self, logits: FloatTensor) -> LongTensor:
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)


def build_sampler(do_sample: bool = True) -> ARSampler:
    return MultinomialSampler() if do_sample else GreedySampler()


# =============================================================================
# Convenience function for building complete generation config
# =============================================================================

def build_generation_components(
    max_new_tokens: int = 100,
    min_new_tokens: int = 0,
    prompt_length: int = 0,
    do_sample: bool = True,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    eos_token_id: Optional[int] = None,
    use_last_token_repetition_penalty: bool = True,
    custom_logits_processors: Optional[List] = None,
    custom_stopping_criteria: Optional[List] = None,
    custom_sampler: Optional[ARSampler] = None,
) -> tuple[List, List, ARSampler]:
    if custom_logits_processors is not None:
        logits_processors = custom_logits_processors
    else:
        logits_processors = build_default_logits_processors(
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            min_new_tokens=min_new_tokens,
            eos_token_id=eos_token_id,
            prompt_length=prompt_length,
            use_last_token_repetition_penalty=use_last_token_repetition_penalty,
        )
    
    if custom_stopping_criteria is not None:
        stopping_criteria = custom_stopping_criteria
    else:
        stopping_criteria = build_default_stopping_criteria(
            max_new_tokens=max_new_tokens,
            prompt_length=prompt_length,
            eos_token_id=eos_token_id,
            min_new_tokens=min_new_tokens,
        )
    
    if custom_sampler is not None:
        sampler = custom_sampler
    else:
        sampler = build_sampler(do_sample)
    
    return logits_processors, stopping_criteria, sampler
