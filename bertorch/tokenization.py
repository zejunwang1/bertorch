# coding=utf-8

import copy
import json
import os
import re
import requests
import unicodedata
import numpy as np

from loguru import logger 
from collections import OrderedDict, UserDict
from typing import Any, Dict, List, Optional, Tuple, Union
from .pretraining import torch_cache_home
from .utils import (
    VOCAB_NAME,
    TOKENIZER_CONFIG_NAME,
    SPECIAL_TOKENS_MAP_NAME,
    cached_path, hf_bucket_url
)


def _is_whitespace(char):
    """
    Checks whether `char` is a whitespace character.
    """
    # \t, \n, and \r are technically control characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def _is_control(char):
    """
    Checks whether `char` is a control character.
    """
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False


def _is_punctuation(char):
    """
    Checks whether `char` is a punctuation character.
    """
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if (cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False


def load_vocab(vocab_file):
    """
    Loads a vocabulary file into a dictionary.
    """
    vocab = OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip("\n")
        vocab[token] = index
    return vocab


def whitespace_tokenize(text):
    """
    Runs basic whitespace cleaning and splitting on a piece of text.
    """
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


def _is_numpy(x):
    return isinstance(x, np.ndarray)


def _is_torch_device(x):
    import torch
    return isinstance(x, torch.device)


class BatchEncoding(UserDict):
    """
    Holds the output of the method `BertTokenizer.encode_plus` and `BertTokenizer.batch_encode_plus`.
    This class is derived from a python dictionary and can be used as a dictionary.
    """
    def __init__(
        self,
        data: Optional[Dict[str, Any]] = None,
        tensor_type: Optional[str] = None,
        prepend_batch_axis: bool = False
    ):
        super().__init__(data)
        self.convert_to_tensors(tensor_type=tensor_type, prepend_batch_axis=prepend_batch_axis)
    
    def __getattr__(self, item: str):
        try:
            return self.data[item]
        except KeyError:
            raise AttributeError

    def convert_to_tensors(
        self, tensor_type: Optional[str] = None, prepend_batch_axis: bool = False
    ):
        """
        Convert the inner content to tensors.
        """
        if tensor_type is None:
            return self
        elif tensor_type == 'pt':
            import torch
            as_tensor = torch.tensor
            is_tensor = torch.is_tensor
        elif tensor_type == 'np':
            as_tensor = np.asarray
            is_tensor = _is_numpy
        else:
            raise ValueError("Invalid tensor type: {}, only supported 'pt' and 'np'.".format(tensor_type))
        
        # Do the tensor conversion in batc
        for key, value in self.items():
            try:
                if prepend_batch_axis:
                    value = [value]
                
                if not is_tensor(value):
                    tensor = as_tensor(value)
                    self[key] = tensor
            except:
                raise ValueError(
                    "Unable to create tensor, you should probably activate truncation and/or padding "
                    "with 'padding=True' 'truncation=True' to have batched tensors with the same length."
                )
        
        return self
    
    def to(self, device):
        if isinstance(device, str) or isinstance(device, int) or _is_torch_device(device):
            self.data = {k: v.to(device=device) for k, v in self.data.items()}
        else:
            raise ValueError("Invalid device parameter.")
        return self
        

class BertTokenizer(object):
    """
    Construct a BERT tokenizer. Based on WordPiece.
    """
    def __init__(
        self,
        vocab_file,
        do_lower_case=True,
        do_basic_tokenize=True,
        never_split=None,
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        model_max_length=512,
        tokenize_chinese_chars=True
    ):
        """
        Args:
            vocab_file (type: str):
                File containing the vocabulary.
            do_lower_case (type: bool, optional, defaults to True):
                Whether or not to lowercase the input when tokenizing.
            do_basic_tokenize (type: bool, optional, defaults to True):
                Whether or not to do basic tokenization before WordPiece.
            never_split (type: Iterable, optional, defaults to None):
                Collection of tokens which will never be split during tokenization. Only has an effect when 
                do_basic_tokenize=True.
            unk_token (type: str, optional, defaults to "[UNK]"):
                The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this 
                token instead.
            sep_token (type: str, optional, defaults to "[SEP]"):
                The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
                sequence classification or for a text and a question for question answering. It is also used as the last
                token of a sequence built with special tokens.
            pad_token (type: str, optional, defaults to "[PAD]"):
                The token used for padding, for example when batching sequences of different lengths.
            cls_token (type: str, optional, defaults to "[CLS]"):
                The classifier token which is used when doing sequence classification (classification of the whole sequence
                instead of per-token classification). It is the first token of the sequence when built with special tokens.
            mask_token (type: str, optional, defaults to "[MASK]"):
                The token used for masking values. This is the token used when training this model with masked language
                modeling. This is the token which the model will try to predict.
            tokenize_chinese_chars (type: bool, optional, defaults to True):
                Whether or not to tokenize Chinese characters.
        """
        if not os.path.isfile(vocab_file):
            raise ValueError(
                "Can't find a vocabulary file at path {}. To load the vocabulary from a huggingface pretrained "
                "model use `tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`".format(vocab_file)
            )
        
        self.unk_token = unk_token
        self.sep_token = sep_token
        self.pad_token = pad_token
        self.cls_token = cls_token
        self.mask_token = mask_token
        self.all_special_tokens = [unk_token, sep_token, pad_token, cls_token, mask_token]
        self.special_tokens_map = {
            "unk_token": unk_token, "sep_token": sep_token, "pad_token": pad_token,
            "cls_token": cls_token, "mask_token": mask_token        
        }

        if never_split is None:
            self.never_split = self.all_special_tokens
        else:
            self.never_split = list(set(never_split).union(self.all_special_tokens)) 
        
        self.vocab = load_vocab(vocab_file)
        self.ids_to_tokens = OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])
        self.model_max_length = model_max_length
        self.do_basic_tokenize = do_basic_tokenize
        if do_basic_tokenize:
            self.basic_tokenizer = BasicTokenizer(
                do_lower_case=do_lower_case,
                tokenize_chinese_chars=tokenize_chinese_chars
            )
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab, unk_token=self.unk_token)

    @property
    def do_lower_case(self):
        return self.basic_tokenizer.do_lower_case
    
    @property
    def tokenize_chinese_chars(self):
        return self.basic_tokenizer.tokenize_chinese_chars

    @property
    def vocab_size(self):
        return len(self.vocab)
    
    @property
    def unk_token_id(self):
        return self._convert_token_to_id(self.unk_token)
    
    @property
    def sep_token_id(self):
        return self._convert_token_to_id(self.sep_token)
    
    @property
    def pad_token_id(self):
        return self._convert_token_to_id(self.pad_token)
    
    @property
    def cls_token_id(self):
        return self._convert_token_to_id(self.cls_token)
    
    @property
    def mask_token_id(self):
        return self._convert_token_to_id(self.mask_token)
    
    @property
    def all_special_ids(self):
        all_toks = self.all_special_tokens
        all_ids = self.convert_tokens_to_ids(all_toks)
        return all_ids
    
    @property
    def split_pattern(self):
        """
        Construct regular expression according to never split tokens.
        """
        pat_list = ['(']
        for tok in self.never_split:
            pat_list.append(tok)
            pat_list.append('|')

        pat_list.pop()
        pat_list.append(')')
        pat = ''.join(pat_list)
        pat = pat.replace('[', '\[')
        pat = pat.replace(']', '\]')
        return pat
    
    def tokenize(self, text):
        output_tokens = []
        text_list = re.split(self.split_pattern, text)
        for sub_text in text_list:
            if sub_text in self.never_split:
                output_tokens.append(sub_text)
            else:
                if self.do_basic_tokenize:
                    for token in self.basic_tokenizer.tokenize(sub_text):
                        output_tokens.extend(self.wordpiece_tokenizer.tokenize(token))
                else:
                    output_tokens.extend(self.wordpiece_tokenizer.tokenize(sub_text))
        return output_tokens
    
    def _convert_token_to_id(self, token):
        """
        Converts a token (str) in an id using the vocab.
        """
        return self.vocab.get(token, self.vocab.get(self.unk_token))
    
    def _convert_id_to_token(self, index):
        """
        Converts an index (integer) in a token (str) using the vocab.
        """
        return self.ids_to_tokens.get(index, self.unk_token)

    def convert_tokens_to_ids(
        self, tokens: Union[str, List[str]], add_special_tokens: bool = False
    ) -> Union[int, List[int]]:
        """
        Converts a token string (or a sequence of tokens) in a single integer id (or a sequence of ids), using the
        vocabulary.
        """
        if tokens is None:
            return None
        
        if isinstance(tokens, str):
            return self._convert_token_to_id(tokens)
        
        ids = [self.cls_token_id] if add_special_tokens else []
        for token in tokens:
            ids.append(self._convert_token_to_id(token))
        if add_special_tokens:
            ids.append(self.sep_token_id)
        return ids
    
    def convert_ids_to_tokens(
        self, ids: Union[int, List[int]], skip_special_tokens: bool = False
    ) -> Union[str, List[str]]:
        """
        Converts a single index or a sequence of indices in a token or a sequence of tokens, using the vocabulary and
        added tokens.
        """ 
        if isinstance(ids, int):
            return self._convert_id_to_token(ids)
        
        tokens = []
        for index in ids:
            index = int(index)
            if skip_special_tokens and index in self.all_special_ids:
                continue
            tokens.append(self._convert_id_to_token(index))
        return tokens
    
    def encode(
        self, 
        text: str, 
        text_pair: Optional[str] = None, 
        add_special_tokens: bool = True,
        truncation: bool = False,
        max_length: Optional[int] = None,
        truncate_from: Union[int, str] = 'right'
    ) -> List[List[int]]:
        """
        Converts a string to a sequence of ids (integer), using the tokenizer and vocabulary.
        Returns: [input_ids, token_type_ids, attention_mask]
        """
        first_tokens = self.tokenize(text)
        second_tokens = None
        if text_pair is not None:
            second_tokens = self.tokenize(text_pair)
        
        if truncation:
            if max_length is None:
                max_length = self.model_max_length
            if truncate_from == "right":
                index = -1
            elif truncate_from == "left":
                index = 0
            else:
                index = int(truncate_from)
            
            seq = [first_tokens, second_tokens] if second_tokens is not None else [first_tokens]
            if add_special_tokens:
                if second_tokens is None:
                    max_length -= 2
                else:
                    max_length -= 3
            while True:
                seq_len = [len(s) for s in seq]
                if sum(seq_len) > max_length:
                    i = np.argmax(seq_len)
                    seq[i].pop(index)
                else:
                    break
            first_tokens = seq[0]
            if len(seq) > 1:
                second_tokens = seq[1]
        
        input_ids = self.convert_tokens_to_ids(first_tokens, add_special_tokens)
        token_type_ids = [0] * len(input_ids)
        if second_tokens is not None:
            second_ids = self.convert_tokens_to_ids(second_tokens, add_special_tokens=False)
            if add_special_tokens:
                second_ids.append(self.sep_token_id)
            input_ids.extend(second_ids)
            token_type_ids.extend([1] * len(second_ids))

        attention_mask = [1] * len(input_ids)
        return [input_ids, token_type_ids, attention_mask]
    
    def __call__(
        self,
        text: Union[str, List[str], Tuple[str], List[Tuple[str]]],
        text_pair: Optional[Union[str, List[str], Tuple[str]]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str] = False,
        truncation: bool = False,
        max_length: Optional[int] = None,
        return_tensors: Optional[str] = None,
        truncate_from: Union[int, str] = 'right'
    ):
        # Input type checking for clearer error
        def _is_valid_text_input(t):
            if isinstance(t, str):
                # Strings are fine
                return True
            elif isinstance(t, (list, tuple)):
                # List are fine as long as they are...
                if len(t) == 0:
                    # ... empty
                    return False
                elif isinstance(t[0], str):
                    # ... list of strings
                    return True
                elif isinstance(t[0], (list, tuple)):
                    # ... list with an empty list or with a list of strings
                    for pair in t:
                        if len(pair) == 0:
                            # ... empty
                            return False
                    return isinstance(t[0][0], str)
                else:
                    return False
            else:
                return False 
        
        def _is_valid_text_pair_input(t):
            if isinstance(t, str):
                # Strings are fine
                return True
            elif isinstance(t, (list, tuple)):
                if len(t) == 0:
                    # ... empty
                    return False
                elif isinstance(t[0], str):
                    # ... list of strings
                    return True
                else:
                    return False
            else:
                return False

        if not _is_valid_text_input(text):
            raise ValueError("text input must of type `str`, `List[str]`, `List[List[str]]`.")
        
        if text_pair is not None and not _is_valid_text_pair_input(text_pair):
            raise ValueError("text pair input must of type `str` or `List[str]`.")
        
        is_batched = isinstance(text, (list, tuple))
        
        if is_batched:
            if isinstance(text_pair, str):
                raise TypeError(
                    "when tokenizing batches of text, `text_pair` must be a list or tuple with the same length as `text`."
                )
            if text_pair is not None and len(text) != len(text_pair):
                raise ValueError(
                    "batch length of `text`: {} does not match batch length of `text_pair`: {}.".format(len(text), len(text_pair))
                )
            batch_text_or_text_pairs = list(zip(text, text_pair)) if text_pair is not None else text
            return self.batch_encode_plus(
                batch_text_or_text_pairs=batch_text_or_text_pairs,
                add_special_tokens=add_special_tokens,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                return_tensors=return_tensors,
                truncate_from=truncate_from
            )
        else:
            return self.encode_plus(
                text,
                text_pair=text_pair,
                add_special_tokens=add_special_tokens,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                return_tensors=return_tensors,
                truncate_from=truncate_from
            )
            
    def encode_plus(
        self,
        text: str,
        text_pair: Optional[str] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str] = False,
        truncation: bool = False,
        max_length: Optional[int] = None,
        return_tensors: Optional[str] = None,
        truncate_from: Union[int, str] = 'right'
    ):
        encode_outputs = self.encode(
            text,
            text_pair=text_pair,
            add_special_tokens=add_special_tokens,
            truncation=truncation,
            max_length=max_length,
            truncate_from=truncate_from
        )

        input_ids = encode_outputs[0]
        token_type_ids = encode_outputs[1]
        attention_mask = encode_outputs[2]
                
        if padding == "max_length":
            pad_num = max_length - len(input_ids)
            if pad_num > 0:
                input_ids.extend([self.pad_token_id] * pad_num)
                token_type_ids.extend([0] * pad_num)
                attention_mask.extend([0] * pad_num)
        
        data = {
            'input_ids': input_ids,
            'token_type_ids': token_type_ids,
            'attention_mask': attention_mask
        }
        return BatchEncoding(data, tensor_type=return_tensors, prepend_batch_axis=True)
    
    def batch_encode_plus(
        self,
        batch_text_or_text_pairs: Union[List[str], Tuple[str], List[Tuple[str]]],
        add_special_tokens: bool = True,
        padding: Union[bool, str] = False,
        truncation: bool = False,
        max_length: Optional[int] = None,
        return_tensors: Optional[str] = None,
        truncate_from: Union[int, str] = 'right'
    ):
        input_ids, token_type_ids, attention_mask = [], [], []
        for text_or_text_pairs in batch_text_or_text_pairs:
            if isinstance(text_or_text_pairs, str):
                encode_outputs = self.encode(
                    text_or_text_pairs,
                    add_special_tokens=add_special_tokens,
                    truncation=truncation,
                    max_length=max_length,
                    truncate_from=truncate_from
                )
            else:
                encode_outputs = self.encode(
                    text_or_text_pairs[0],
                    text_pair=text_or_text_pairs[1],
                    add_special_tokens=add_special_tokens,
                    truncation=truncation,
                    max_length=max_length,
                    truncate_from=truncate_from
                )

            input_ids.append(encode_outputs[0])
            token_type_ids.append(encode_outputs[1])
            attention_mask.append(encode_outputs[2])

        max_seq_len = max(len(l) for l in input_ids)
        if padding == "max_length":
            seq_len = max_length
        elif padding == True:
            seq_len = max_seq_len
        if padding == "max_length" or padding == True:
            for i in range(len(batch_text_or_text_pairs)):
                pad_num = seq_len - len(input_ids[i])
                input_ids[i].extend([self.pad_token_id] * pad_num)
                token_type_ids[i].extend([0] * pad_num)
                attention_mask[i].extend([0] * pad_num)
        
        data = {
            'input_ids': input_ids,
            'token_type_ids': token_type_ids,
            'attention_mask': attention_mask
        }
        return BatchEncoding(data, tensor_type=return_tensors, prepend_batch_axis=False)  
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs):
        """
        Instantiate a BertTokenizer class from a pretrained model name or path.
        """
        cache_dir = kwargs.pop("cache_dir", None)
        if cache_dir is None:
            cache_dir = os.path.join(torch_cache_home, "bertorch")
            cache_dir = os.path.join(cache_dir, pretrained_model_name_or_path.replace("/", "_"))
        os.makedirs(cache_dir, exist_ok=True)

        pretrained_model_name_or_path = str(pretrained_model_name_or_path)
        if os.path.isdir(pretrained_model_name_or_path):
            vocab_file = os.path.join(pretrained_model_name_or_path, VOCAB_NAME)
            tokenizer_config_file = os.path.join(pretrained_model_name_or_path, TOKENIZER_CONFIG_NAME)
            special_tokens_map_file = os.path.join(pretrained_model_name_or_path, SPECIAL_TOKENS_MAP_NAME)
        elif os.path.isfile(pretrained_model_name_or_path):
            vocab_file = pretrained_model_name_or_path
            tokenizer_config_file = None
            special_tokens_map_file = None
        else:
            vocab_url = hf_bucket_url(pretrained_model_name_or_path, VOCAB_NAME)
            tokenizer_config_url = hf_bucket_url(pretrained_model_name_or_path, TOKENIZER_CONFIG_NAME)
            special_tokens_map_url = hf_bucket_url(pretrained_model_name_or_path, SPECIAL_TOKENS_MAP_NAME)
            
            # Load `vocab.txt` `tokenizer_config.json` `special_tokens_map.json` from URL
            try:
                vocab_file = cached_path(vocab_url, filename=VOCAB_NAME, cache_dir=cache_dir)
            except EnvironmentError as err:
                logger.error(err)
                msg = (
                    "Can't load vocab.txt for '{}'. Make sure that: '{}' is a correct model identifier "
                    "listed on https://huggingface.co/models".format(pretrained_model_name_or_path, pretrained_model_name_or_path)
                )
                raise EnvironmentError(msg)

            try:
                tokenizer_config_file = cached_path(tokenizer_config_url, filename=TOKENIZER_CONFIG_NAME, cache_dir=cache_dir)
            except requests.exceptions.HTTPError as err:
                if "404 Client Error" in str(err):
                    tokenizer_config_file = None                   
            
            try:
                special_tokens_map_file = cached_path(special_tokens_map_url, filename=SPECIAL_TOKENS_MAP_NAME, cache_dir=cache_dir)
            except requests.exceptions.HTTPError as err:
                if "404 Client Error" in str(err):
                    special_tokens_map_file = None

        if tokenizer_config_file and os.path.isfile(tokenizer_config_file):
            with open(tokenizer_config_file, encoding="utf-8") as tokenizer_config_handle:
                init_kwargs = json.load(tokenizer_config_handle)
            keys = copy.deepcopy(list(init_kwargs.keys()))
            valid = ['do_lower_case', 'do_basic_tokenize', 'never_split', 'unk_token', 'sep_token', 'pad_token',
                     'cls_token', 'mask_token', 'model_max_length', 'tokenize_chinese_chars']
            for key in keys:
                if key not in valid:
                    init_kwargs.pop(key)

            # Update with newly provided kwargs
            init_kwargs.update(kwargs)
        else:
            init_kwargs = kwargs
        
        tokenizer = cls(vocab_file, **init_kwargs)
        
        # If there is a complementary special token map, load it
        if special_tokens_map_file and os.path.isfile(special_tokens_map_file):
            with open(special_tokens_map_file, encoding="utf-8") as special_tokens_map_handle:
                special_tokens_map = json.load(special_tokens_map_handle)
            for key, value in special_tokens_map.items():
                if not hasattr(tokenizer, key) and isinstance(value, str):
                    setattr(tokenizer, key, value)
                    tokenizer.all_special_tokens.append(value)
                    tokenizer.special_tokens_map[key] = value
                    tokenizer.never_split = list(set(tokenizer.never_split.append(value)))
        
        return tokenizer 
    
    def save_pretrained(
        self, 
        save_directory: Union[str, os.PathLike],
        save_tokenizer_config: bool = False,
        save_special_tokens_map: bool = False
    ):
        """
        Save the full tokenizer state.

        This method make sure the full tokenizer can then be re-loaded using the 
        BertTokenizer.from_pretrained class method.
        """
        if os.path.isfile(save_directory):
            raise AssertionError("Provided path ({}) should be a directory, not a file".format(save_directory))
        
        os.makedirs(save_directory, exist_ok=True)
        
        if save_tokenizer_config:
            tokenizer_config_file = os.path.join(save_directory, TOKENIZER_CONFIG_NAME)
            tokenizer_config = {
                "do_lower_case": self.do_lower_case,
                "do_basic_tokenize": self.do_basic_tokenize,
                "unk_token": self.unk_token,
                "sep_token": self.sep_token,
                "pad_token": self.pad_token,
                "cls_token": self.cls_token,
                "mask_token": self.mask_token,
                "model_max_length": self.model_max_length,
                "tokenize_chinese_chars": self.tokenize_chinese_chars
            }
            with open(tokenizer_config_file, "w", encoding="utf-8") as f:
                f.write(json.dumps(tokenizer_config, ensure_ascii=False))
            #logger.info("tokenizer config file saved in {}".format(tokenizer_config_file))
        
        if save_special_tokens_map:
            special_tokens_map_file = os.path.join(save_directory, SPECIAL_TOKENS_MAP)
            with open(special_tokens_map_file, "w", encoding="utf-8") as f:
                f.write(json.dumps(self.special_tokens_map, ensure_ascii=False))
            #logger.info("Special tokens file saved in {}".format(special_tokens_map_file))
        
        index = 0
        vocab_file = os.path.join(save_directory, VOCAB_NAME)
        with open(vocab_file, "w", encoding="utf-8") as writer:
            for token, token_index in sorted(self.vocab.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    logger.warning(
                        "Saving vocabulary to {}: vocabulary indices are not consecutive."
                        " Please check that the vocabulary is not corrupted!".format(vocab_file)
                    )
                    index = token_index
                writer.write(token + "\n")
                index += 1
        

class BasicTokenizer(object):
    """
    Constructs a BasicTokenizer that will run basic tokenization (punctuation splitting, lower casing, etc.).
    """
    def __init__(self, do_lower_case=True, tokenize_chinese_chars=True):
        """
        Args:
            do_lower_case (type: bool, optional, defaults to True):
                Whether or not to lowercase the input when tokenizing.
            tokenize_chinese_chars (type: bool, optional, defaults to True):
                Collection of tokens which will never be split during tokenization. Only has an effect when 
                do_basic_tokenize=True.
        """
        self.do_lower_case = do_lower_case
        self.tokenize_chinese_chars = tokenize_chinese_chars
    
    def tokenize(self, text):
        """
        Basic Tokenization of a piece of text. Split on "white spaces" only, for sub-word tokenization, see
        WordPieceTokenizer.
        """
        text = self._clean_text(text)
        
        if self.tokenize_chinese_chars:
            text = self._tokenize_chinese_chars(text)
        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if self.do_lower_case:
                token = token.lower()
                token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token))
        
        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens

    def _run_strip_accents(self, text):
        """
        Strips accents from a piece of text.
        """
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)
    
    def _run_split_on_punc(self, text):
        """
        Splits punctuation on a piece of text.
        """
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]
    
    def _tokenize_chinese_chars(self, text):
        """
        Adds whitespace around any CJK character.
        """
        output = []
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def _is_chinese_char(self, cp):
        """
        Checks whether CP is the codepoint of a CJK character.
        """
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all of the other languages.
        if (
            (cp >= 0x4E00 and cp <= 0x9FFF)
            or (cp >= 0x3400 and cp <= 0x4DBF)  
            or (cp >= 0x20000 and cp <= 0x2A6DF)  
            or (cp >= 0x2A700 and cp <= 0x2B73F)  
            or (cp >= 0x2B740 and cp <= 0x2B81F)  
            or (cp >= 0x2B820 and cp <= 0x2CEAF)  
            or (cp >= 0xF900 and cp <= 0xFAFF)
            or (cp >= 0x2F800 and cp <= 0x2FA1F)  
        ):
            return True

        return False
    
    def _clean_text(self, text):
        """
        Performs invalid character removal and whitespace cleanup on text.
        """
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xFFFD or _is_control(char):
                continue
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)


class WordpieceTokenizer(object):
    """
    Runs WordPiece tokenization.
    """
    def __init__(self, vocab, unk_token, max_input_chars_per_word=100):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word
    
    def tokenize(self, text):
        """
        Tokenizes a piece of text into its word pieces. This uses a greedy longest-match-first algorithm to perform
        tokenization using the given vocabulary.
        
        For example, `input = "unaffable"` wil return as output: `["un", "##aff", "##able"]`.

        Args:
          text: A single token or whitespace separated tokens. This should have
            already been passed through `BasicTokenizer`.

        Returns:
          A list of wordpiece tokens.
        """
        output_tokens = []
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens
