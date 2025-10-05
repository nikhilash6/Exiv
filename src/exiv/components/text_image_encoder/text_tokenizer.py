import os

# TODO: properly refactor and fix this code
class SDTokenizer:
    def __init__(
        self, 
        tokenizer_path=None, 
        max_length=77, 
        pad_with_end=True, 
        embedding_directory=None, 
        embedding_size=768, 
        embedding_key='clip_l', 
        tokenizer_class=CLIPTokenizer, 
        has_start_token=True, 
        has_end_token=True, 
        pad_to_max_length=True, 
        min_length=None, 
        pad_token=None, 
        end_token=None, 
        min_padding=None, 
        tokenizer_data={}, 
        tokenizer_args={}
    ):
        if tokenizer_path is None:
            tokenizer_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "sd1_tokenizer")
        self.tokenizer = tokenizer_class.from_pretrained(tokenizer_path, **tokenizer_args)
        self.max_length = tokenizer_data.get("{}_max_length".format(embedding_key), max_length)
        self.min_length = tokenizer_data.get("{}_min_length".format(embedding_key), min_length)
        self.end_token = None
        self.min_padding = min_padding

        empty = self.tokenizer('')["input_ids"]
        self.tokenizer_adds_end_token = has_end_token
        if has_start_token:
            self.tokens_start = 1
            self.start_token = empty[0]
            if end_token is not None:
                self.end_token = end_token
            else:
                if has_end_token:
                    self.end_token = empty[1]
        else:
            self.tokens_start = 0
            self.start_token = None
            if end_token is not None:
                self.end_token = end_token
            else:
                if has_end_token:
                    self.end_token = empty[0]

        if pad_token is not None:
            self.pad_token = pad_token
        elif pad_with_end:
            self.pad_token = self.end_token
        else:
            self.pad_token = 0

        self.pad_with_end = pad_with_end
        self.pad_to_max_length = pad_to_max_length

        vocab = self.tokenizer.get_vocab()
        self.inv_vocab = {v: k for k, v in vocab.items()}
        self.embedding_directory = embedding_directory
        self.max_word_length = 8
        self.embedding_identifier = "embedding:"
        self.embedding_size = embedding_size
        self.embedding_key = embedding_key

    def _try_get_embedding(self, embedding_name:str):
        '''
        Takes a potential embedding name and tries to retrieve it.
        Returns a Tuple consisting of the embedding and any leftover string, embedding can be None.
        '''
        split_embed = embedding_name.split()
        embedding_name = split_embed[0]
        leftover = ' '.join(split_embed[1:])
        embed = load_embed(embedding_name, self.embedding_directory, self.embedding_size, self.embedding_key)
        if embed is None:
            stripped = embedding_name.strip(',')
            if len(stripped) < len(embedding_name):
                embed = load_embed(stripped, self.embedding_directory, self.embedding_size, self.embedding_key)
                return (embed, "{} {}".format(embedding_name[len(stripped):], leftover))
        return (embed, leftover)


    def tokenize_with_weights(self, text:str, return_word_ids=False, tokenizer_options={}, **kwargs):
        '''
        Takes a prompt and converts it to a list of (token, weight, word id) elements.
        Tokens can both be integer tokens and pre computed CLIP tensors.
        Word id values are unique per word and embedding, where the id 0 is reserved for non word tokens.
        Returned list has the dimensions NxM where M is the input size of CLIP
        '''
        min_length = tokenizer_options.get("{}_min_length".format(self.embedding_key), self.min_length)
        min_padding = tokenizer_options.get("{}_min_padding".format(self.embedding_key), self.min_padding)

        text = escape_important(text)
        if kwargs.get("disable_weights", False):
            parsed_weights = [(text, 1.0)]
        else:
            parsed_weights = token_weights(text, 1.0)

        # tokenize words
        tokens = []
        for weighted_segment, weight in parsed_weights:
            to_tokenize = unescape_important(weighted_segment)
            split = re.split(' {0}|\n{0}'.format(self.embedding_identifier), to_tokenize)
            to_tokenize = [split[0]]
            for i in range(1, len(split)):
                to_tokenize.append("{}{}".format(self.embedding_identifier, split[i]))

            to_tokenize = [x for x in to_tokenize if x != ""]
            for word in to_tokenize:
                # if we find an embedding, deal with the embedding
                if word.startswith(self.embedding_identifier) and self.embedding_directory is not None:
                    embedding_name = word[len(self.embedding_identifier):].strip('\n')
                    embed, leftover = self._try_get_embedding(embedding_name)
                    if embed is None:
                        logging.warning(f"warning, embedding:{embedding_name} does not exist, ignoring")
                    else:
                        if len(embed.shape) == 1:
                            tokens.append([(embed, weight)])
                        else:
                            tokens.append([(embed[x], weight) for x in range(embed.shape[0])])
                    #if we accidentally have leftover text, continue parsing using leftover, else move on to next word
                    if leftover != "":
                        word = leftover
                    else:
                        continue
                end = 999999999999
                if self.tokenizer_adds_end_token:
                    end = -1
                #parse word
                tokens.append([(t, weight) for t in self.tokenizer(word)["input_ids"][self.tokens_start:end]])

        #reshape token array to CLIP input size
        batched_tokens = []
        batch = []
        if self.start_token is not None:
            batch.append((self.start_token, 1.0, 0))
        batched_tokens.append(batch)
        for i, t_group in enumerate(tokens):
            #determine if we're going to try and keep the tokens in a single batch
            is_large = len(t_group) >= self.max_word_length
            if self.end_token is not None:
                has_end_token = 1
            else:
                has_end_token = 0

            while len(t_group) > 0:
                if len(t_group) + len(batch) > self.max_length - has_end_token:
                    remaining_length = self.max_length - len(batch) - has_end_token
                    #break word in two and add end token
                    if is_large:
                        batch.extend([(t,w,i+1) for t,w in t_group[:remaining_length]])
                        if self.end_token is not None:
                            batch.append((self.end_token, 1.0, 0))
                        t_group = t_group[remaining_length:]
                    #add end token and pad
                    else:
                        if self.end_token is not None:
                            batch.append((self.end_token, 1.0, 0))
                        if self.pad_to_max_length:
                            batch.extend([(self.pad_token, 1.0, 0)] * (remaining_length))
                    #start new batch
                    batch = []
                    if self.start_token is not None:
                        batch.append((self.start_token, 1.0, 0))
                    batched_tokens.append(batch)
                else:
                    batch.extend([(t,w,i+1) for t,w in t_group])
                    t_group = []

        #fill last batch
        if self.end_token is not None:
            batch.append((self.end_token, 1.0, 0))
        if min_padding is not None:
            batch.extend([(self.pad_token, 1.0, 0)] * min_padding)
        if self.pad_to_max_length and len(batch) < self.max_length:
            batch.extend([(self.pad_token, 1.0, 0)] * (self.max_length - len(batch)))
        if min_length is not None and len(batch) < min_length:
            batch.extend([(self.pad_token, 1.0, 0)] * (min_length - len(batch)))

        if not return_word_ids:
            batched_tokens = [[(t, w) for t, w,_ in x] for x in batched_tokens]

        return batched_tokens


    def untokenize(self, token_weight_pair):
        return list(map(lambda a: (a, self.inv_vocab[a[0]]), token_weight_pair))

    def state_dict(self):
        return {}


class SD1Tokenizer:
    def __init__(self, embedding_directory=None, tokenizer_data={}, clip_name="l", tokenizer=SDTokenizer, name=None):
        if name is not None:
            self.clip_name = name
            self.clip = "{}".format(self.clip_name)
        else:
            self.clip_name = clip_name
            self.clip = "clip_{}".format(self.clip_name)

        tokenizer = tokenizer_data.get("{}_tokenizer_class".format(self.clip), tokenizer)
        setattr(self, self.clip, tokenizer(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data))

    def tokenize_with_weights(self, text:str, return_word_ids=False, **kwargs):
        out = {}
        out[self.clip_name] = getattr(self, self.clip).tokenize_with_weights(text, return_word_ids, **kwargs)
        return out

    def untokenize(self, token_weight_pair):
        return getattr(self, self.clip).untokenize(token_weight_pair)

    def state_dict(self):
        return getattr(self, self.clip).state_dict()

# TODO: complete this
class WanT5Tokenizer(SD1Tokenizer):
    pass