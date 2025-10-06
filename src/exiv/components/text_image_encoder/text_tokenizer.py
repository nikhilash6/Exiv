import torch
from transformers import CLIPTokenizer

import os
import traceback
import re
import zipfile

from ...utils.logging import app_logger



def safe_load_embed_zip(embed_path):
    with zipfile.ZipFile(embed_path) as myzip:
        names = list(filter(lambda a: "data/" in a, myzip.namelist()))
        names.reverse()
        for n in names:
            with myzip.open(n) as myfile:
                data = myfile.read()
                number = len(data) // 4
                length_embed = 1024 #sd2.x
                if number < 768:
                    continue
                if number % 768 == 0:
                    length_embed = 768 #sd1.x
                num_embeds = number // length_embed
                embed = torch.frombuffer(data, dtype=torch.float)
                out = embed.reshape((num_embeds, length_embed)).clone()
                del embed
                return out

def expand_directory_list(directories):
    dirs = set()
    for x in directories:
        dirs.add(x)
        for root, subdir, file in os.walk(x, followlinks=True):
            dirs.add(root)
    return list(dirs)

def load_embed(embedding_name, embedding_directory, embedding_size, embed_key=None):
    if isinstance(embedding_directory, str):
        embedding_directory = [embedding_directory]

    embedding_directory = expand_directory_list(embedding_directory)

    valid_file = None
    for embed_dir in embedding_directory:
        embed_path = os.path.abspath(os.path.join(embed_dir, embedding_name))
        embed_dir = os.path.abspath(embed_dir)
        try:
            if os.path.commonpath((embed_dir, embed_path)) != embed_dir:
                continue
        except:
            continue
        if not os.path.isfile(embed_path):
            extensions = ['.safetensors', '.pt', '.bin']
            for x in extensions:
                t = embed_path + x
                if os.path.isfile(t):
                    valid_file = t
                    break
        else:
            valid_file = embed_path
        if valid_file is not None:
            break

    if valid_file is None:
        return None

    embed_path = valid_file

    embed_out = None

    try:
        if embed_path.lower().endswith(".safetensors"):
            import safetensors.torch
            embed = safetensors.torch.load_file(embed_path, device="cpu")
        else:
            try:
                embed = torch.load(embed_path, weights_only=True, map_location="cpu")
            except:
                embed_out = safe_load_embed_zip(embed_path)
    except Exception:
        app_logger.warning("{}\n\nerror loading embedding, skipping loading: {}".format(traceback.format_exc(), embedding_name))
        return None

    if embed_out is None:
        if 'string_to_param' in embed:
            values = embed['string_to_param'].values()
            embed_out = next(iter(values))
        elif isinstance(embed, list):
            out_list = []
            for x in range(len(embed)):
                for k in embed[x]:
                    t = embed[x][k]
                    if t.shape[-1] != embedding_size:
                        continue
                    out_list.append(t.reshape(-1, t.shape[-1]))
            embed_out = torch.cat(out_list, dim=0)
        elif embed_key is not None and embed_key in embed:
            embed_out = embed[embed_key]
        else:
            embed_out = bundled_embed(embed, 'bundle_emb.', '.string_to_param.*')
            if embed_out is None:
                embed_out = bundled_embed(embed, 'bundle_emb.', '.{}'.format(embed_key))
            if embed_out is None:
                values = embed.values()
                embed_out = next(iter(values))
    return embed_out


# TODO: properly refactor and fix this code
class SDTokenizer:
    def __init__(
        self, 
        tokenizer_path=None, 
        max_length=77, 
        pad_with_end=True, 
        embedding_directory=None, 
        embedding_size=768, 
        embedding_key='clip_l',         # encoder supported by the embedding
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
        '''
        - loads the tokenizer using the transformers library
        - sets basic tokens such as start, end, pad
        - sets basic details for embedding loading like size, trigger word, max_len etc..
        '''
        # loading the tokenizer through the transformers library
        self.tokenizer = tokenizer_class.from_pretrained(tokenizer_path, **tokenizer_args)
        
        # setting max and min len, finding them in the dict, if they are not  
        # present then using the default value
        self.max_length = tokenizer_data.get("{}_max_length".format(embedding_key), max_length)
        self.min_length = tokenizer_data.get("{}_min_length".format(embedding_key), min_length)
        self.min_padding = min_padding
        
        # 'empty' is like a structural token (with no content), that can be used for
        # marking the beginning and end of sequences
        self.end_token = None
        empty = self.tokenizer('')["input_ids"]
        self.tokenizer_adds_end_token = has_end_token
        
        # setting start and end tokens, diff TEs are trained differently
        if has_start_token:
            self.tokens_start = 1
            self.start_token = empty[0]
            self.end_token = end_token if end_token is not None else (empty[1] if has_end_token else None)
        else:
            self.tokens_start = 0
            self.start_token = None
            self.end_token = end_token if end_token is not None else (empty[0] if has_end_token else None)

        # setting the pad token
        if pad_token is not None:
            self.pad_token = pad_token
        elif pad_with_end:
            self.pad_token = self.end_token
        else:
            self.pad_token = 0

        self.pad_with_end = pad_with_end
        self.pad_to_max_length = pad_to_max_length

        # returns a dict of tokens and their IDs, e.g. {'cat': 5678, 'sky': 1234}
        vocab = self.tokenizer.get_vocab()
        self.inv_vocab = {v: k for k, v in vocab.items()}   # {5678: 'cat', 1234: 'sky'}
        
        # custom embeddings (textual inversion)
        self.embedding_directory = embedding_directory
        self.max_word_length = 8
        self.embedding_identifier = "embedding:"    # keyword to trigger the embedding load
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
                        app_logger.warning(f"warning, embedding:{embedding_name} does not exist, ignoring")
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