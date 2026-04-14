import re
from typing import List, Tuple, Optional, Callable
import math


def split_into_sentences(text: str) -> List[str]:
    """
    split text into sentences using regex.
    example: "hello world. how are you?" -> ["hello world.", "how are you?"]
    """
    # split on sentence boundaries, keeping the punctuation
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    # filter out empty strings
    return [s.strip() for s in sentences if s.strip()]


def chunk_text_by_sentences(
    text: str,
    max_token_chunk_size: int,
    tokenize_fn: Optional[Callable[[str], int]] = None
) -> List[str]:
    """
    split text into chunks based on sentences while respecting token limits.
    example: "first sentence. second sentence." with max_token_chunk_size=100 -> ["first sentence. second sentence."]
    """
    if not text or not text.strip():
        return []
    
    # default token estimator: ~1.3 tokens per word on average for BPE
    def estimate_tokens(t: str) -> int:
        if tokenize_fn is not None:
            return tokenize_fn(t)
        return int(len(t.split()) * 1.3)
    
    sentences = split_into_sentences(text)
    if not sentences:
        return [text.strip()]
    
    chunks = []
    current_chunk = []
    current_token_count = 0
    
    for sentence in sentences:
        sentence_tokens = estimate_tokens(sentence)
        
        # handle case where single sentence exceeds max_token_chunk_size
        # just let the complete sentence through as its own chunk
        if sentence_tokens > max_token_chunk_size and current_chunk:
            # flush current chunk first
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_token_count = 0
        
        # check if adding this sentence would exceed limit
        if current_token_count + sentence_tokens > max_token_chunk_size and current_chunk:
            # flush current chunk
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_token_count = sentence_tokens
        else:
            # add to current chunk
            current_chunk.append(sentence)
            current_token_count += sentence_tokens
    
    # don't forget the last chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks


def _chunk_sentence_by_words(
    sentence: str, 
    max_token_chunk_size: int,
    estimate_tokens: Callable[[str], int]
) -> List[str]:
    """
    split a single long sentence by words.
    tries to split at natural boundaries like commas when possible.
    example: "this is a very long sentence" with max_token_chunk_size=5 -> ["this is", "a very", "long sentence"]
    """
    words = sentence.split()
    chunks = []
    current_chunk = []
    current_token_count = 0
    
    for word in words:
        word_tokens = estimate_tokens(word)
        
        if current_token_count + word_tokens > max_token_chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_token_count = word_tokens
        else:
            current_chunk.append(word)
            current_token_count += word_tokens
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks


def chunk_embeddings_by_sentences(
    inputs_embeds,
    text: str,
    max_token_chunk_size: int,
    tokenizer=None
) -> Tuple[List, List[str]]:
    """
    split embeddings and corresponding text into chunks.
    
    example: aligns embedding tensor with text chunks based on token ratios.
    """
    batch_size = inputs_embeds.shape[0]
    seq_len = inputs_embeds.shape[1]
    
    # if no tokenizer provided, fall back to rough sentence splitting
    if tokenizer is None:
        text_chunks = chunk_text_by_sentences(text, max_token_chunk_size)
        # rough split: divide embeddings evenly
        num_chunks = len(text_chunks)
        chunk_size = seq_len // num_chunks
        embed_chunks = []
        for i in range(num_chunks):
            start = i * chunk_size
            end = seq_len if i == num_chunks - 1 else (i + 1) * chunk_size
            embed_chunks.append(inputs_embeds[:, start:end, :])
        return embed_chunks, text_chunks
    
    # with tokenizer, we can align text and embeddings properly
    # for now, use simple sentence-based chunking
    text_chunks = chunk_text_by_sentences(text, max_token_chunk_size)
    
    # rough alignment: assume tokens are distributed proportionally
    embed_chunks = []
    char_idx = 0
    token_start = 0
    
    for i, chunk_text in enumerate(text_chunks):
        # find the position of this chunk in the original text
        chunk_start = text.find(chunk_text, char_idx)
        chunk_end = chunk_start + len(chunk_text)
        
        # estimate token boundaries
        token_ratio_start = chunk_start / len(text) if len(text) > 0 else 0
        token_ratio_end = chunk_end / len(text) if len(text) > 0 else 0
        
        new_token_start = int(token_ratio_start * seq_len)
        new_token_end = int(token_ratio_end * seq_len)
        
        # ensure minimum size and no overlap
        new_token_start = max(token_start, new_token_start)
        if new_token_end <= new_token_start:
            new_token_end = min(seq_len, new_token_start + 1)
        
        embed_chunks.append(inputs_embeds[:, new_token_start:new_token_end, :])
        
        char_idx = chunk_end
        token_start = new_token_end
    
    return embed_chunks, text_chunks
