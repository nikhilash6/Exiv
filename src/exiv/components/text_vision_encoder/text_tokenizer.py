from transformers import CLIPTokenizer, AutoTokenizer

from .utils import load_embed, parse_prompt_attention

from ...utils.logging import app_logger

# NOTE: we are loading the tokenizer directly on init because generally
# they are very small compared to other parts of the workflow. But this can be 
# optimized if need be.
class SDTokenizer:
    def __init__(
        self, 
        tokenizer_path,
        max_length=77, 
        pad_with_end=True, 
        embedding_directory=None, 
        embedding_size=768, 
        embedding_key='clip_l',
        tokenizer_class=CLIPTokenizer,
    ):
        try:
            self.tokenizer = tokenizer_class.from_pretrained(tokenizer_path)
        except Exception as e:
            app_logger.error(f"Failed to load tokenizer from {tokenizer_path}: {e}")
            raise

        self.max_length = max_length

        self.start_token = self.tokenizer.bos_token_id
        self.end_token = self.tokenizer.eos_token_id
        self.pad_token = self.end_token if pad_with_end else self.tokenizer.pad_token_id

        self.inv_vocab = {v: k for k, v in self.tokenizer.get_vocab().items()}
        
        self.embedding_directory = embedding_directory
        self.embedding_identifier = "embedding:"        # trigger word
        self.embedding_size = embedding_size
        self.embedding_key = embedding_key

    def tokenize_with_weights(self, text:str, return_word_ids=False):
        '''
        returns batched token chunks along with their weights.
        e.g.
        [
            [
                (49406, 1.0),                  # Start Token ('<s>')
                (320, 1.0),                    # Token for 'A'
                (597, 1.0),                    # Token for 'cat'
                (<Tensor>, 1.0),               # The first vector of your embedding
                (<Tensor>, 1.0),               # The second vector of your embedding
                (49407, 1.0),                  # End Token ('</s>')
                (49407, 1.0),                  # Pad Token
                (49407, 1.0),                  # Pad Token
                (49407, 1.0),                  # Pad Token
                (49407, 1.0)                   # Pad Token
            ],
            ...
        ]
        '''
        parsed_weights = parse_prompt_attention(text)   # e.g.  [['a very very ', 1.0], ['big', 2.0], ['cat', 1.0]]

        tokens = []
        word_id_counter = 1
        
        for segment, weight in parsed_weights:
            # checking for the trigger word
            if self.embedding_identifier in segment and self.embedding_directory:   # e.g. segment = "a big embedding:hat"
                parts = segment.split(self.embedding_identifier)
                for i, part in enumerate(parts):
                    if i == 0:
                        # first part ("a big") is normally tokenized and added with the current weights
                        if part:
                            tokenized = self.tokenizer(part, add_special_tokens=False)["input_ids"]
                            tokens.extend([(t, weight, word_id_counter) for t in tokenized])
                            if tokenized: word_id_counter += 1
                    else:
                        # for the rest of the parts, we try to load the embedding from the file
                        # and then directly add it to the list of tokens
                        embedding_name, rest_of_segment = (part.split(" ", 1) + [""])[:2]
                        embed = load_embed(embedding_name.strip(), self.embedding_directory, self.embedding_size, self.embedding_key)
                        if embed is not None:
                            if len(embed.shape) == 1:
                                embed = embed.unsqueeze(0)
                            tokens.extend([(e, weight, word_id_counter) for e in embed])
                            word_id_counter += 1
                        else:
                            app_logger.warning(f"Embedding '{embedding_name}' not found, tokenizing as text.")
                            tokenized = self.tokenizer(embedding_name, add_special_tokens=False)["input_ids"]
                            tokens.extend([(t, weight, word_id_counter) for t in tokenized])
                            if tokenized: word_id_counter +=1

                        if rest_of_segment:
                            tokenized = self.tokenizer(rest_of_segment, add_special_tokens=False)["input_ids"]
                            tokens.extend([(t, weight, word_id_counter) for t in tokenized])
                            if tokenized: word_id_counter += 1
            else:
                tokenized = self.tokenizer(segment, add_special_tokens=False)["input_ids"]
                tokens.extend([(t, weight, word_id_counter) for t in tokenized])
                if tokenized: word_id_counter += 1

        # 'tokens' at this point is a long array of token tuples 
        # [(token, its_weight, counter), (<emb Tensor>, its_weight, counter+1) ...]
        # since 'tokens' can be 1000 token long, we batch them into max_length (77) chunks
        batched_tokens = []
        current_batch = []

        if self.start_token is not None:
            current_batch.append((self.start_token, 1.0, 0))

        for token, weight, word_id in tokens:
            if len(current_batch) >= self.max_length -1: # need space for end token
                if self.end_token is not None:
                    current_batch.append((self.end_token, 1.0, 0))
                # padding
                while len(current_batch) < self.max_length:
                    current_batch.append((self.pad_token, 1.0, 0))
                batched_tokens.append(current_batch)
                current_batch = []
                if self.start_token is not None:
                    current_batch.append((self.start_token, 1.0, 0))
            current_batch.append((token, weight, word_id if return_word_ids else 0))

        if current_batch:
            if self.end_token is not None:
                current_batch.append((self.end_token, 1.0, 0))
            while len(current_batch) < self.max_length:
                current_batch.append((self.pad_token, 1.0, 0))
            batched_tokens.append(current_batch)

        if not return_word_ids:
            batched_tokens = [[(t, w) for t, w, _ in batch] for batch in batched_tokens]
            
        return batched_tokens

    def decode_tokens(self, token_weight_pair):
        if not token_weight_pair or not token_weight_pair[0]:
            return ""
        
        # handles batched token_weight_pair (only the first batch.)
        token_ids = [t for t, w in token_weight_pair[0] if isinstance(t, int)]
        return self.tokenizer.decode(token_ids)


class UMTT5XXLTokenizer(SDTokenizer):
    def __init__(self, embedding_directory=None):
        super().__init__(
            "google/umt5-xxl",
            max_length=512, 
            pad_with_end=True, 
            embedding_directory=embedding_directory, 
            embedding_size=4096, 
            embedding_key='umt5xxl',
            tokenizer_class=AutoTokenizer,
        )
