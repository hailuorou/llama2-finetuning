import tiktoken
from transformers import BatchEncoding
from typing import List


def HF_tuncation(self, text, max_length):
    return self.decode(self.__call__(text, truncation=True, max_length=max_length).input_ids, skip_special_tokens=False)


def TK_get_length(self, text:str):
    return len(self.__call__([text]).input_ids[0])


class TiktokenTokenizer():
    """Tokenizer from OpenAI's tiktoken implementation"""

    def __init__(self, padding_side='left', model_max_length=None):
        self.model_max_length=model_max_length
        self.tokenizer = tiktoken.get_encoding('cl100k_base') #cl100k_base
        self.padding_side = padding_side
        self.eos_token_id = self.tokenizer.eot_token
        self.pad_token_id = self.tokenizer.eot_token # tiktoken does not has pad token
        self.eos_token = '<|endoftext|>'
        self.special_token_ids = set(self.tokenizer._special_tokens.values())

        assert padding_side == 'left' or padding_side == 'right', 'invalid padding_side! choose left/right'

    @property
    def vocab_size(self):
        return self.tokenizer.n_vocab

    @property
    def vocab(self):
        raise NotImplementedError("TiktokenTokenizer does not implement vocabulary access.")

    @property
    def inv_vocab(self):
        raise NotImplementedError("TiktokenTokenizer does not implement vocabulary access. \
                To get the idx-th token in vocabulary, use tokenizer.decode([idx]) .")

    def encode(self, text, **kwargs):
        return self.tokenizer.encode(text, allowed_special="all")

    def batch_encode(self, text, **kwargs):
        return [self.tokenizer.encode(t, allowed_special="all") for t in text]

    def batch_decode(self, sequences, skip_special_tokens: bool = False):
        if skip_special_tokens:
            sequences = [[i for i in s if i not in self.special_token_ids] for s in sequences]
        return [self.tokenizer.decode(s, errors="ignore") for s in sequences]

    def decode(self, sequences, skip_special_tokens: bool = True):
        if skip_special_tokens:
            sequences = [i for i in sequences if i not in self.special_token_ids]
        return self.tokenizer.decode(sequences, errors="ignore")

    def __call__(self, text, padding=False, truncation=False, max_length=None, return_tensors=None, **kwargs):
        """
        padding: True/False, if True, will apply padding_side strategy
        max_length: int, if None: ( if padding is True, will padding to max length in batch otherwise not pad)
        truncation: True/False, if True, max length must has value otherwise not work
        kwargs: not use
        """
        if max_length is None:
            max_length = self.model_max_length

        if isinstance(text, str):
            text = [text]
        else:
            assert isinstance(text, list)

        result = self.tokenizer.encode_batch(text, allowed_special="all")

        if padding:
            if max_length is None:
                max_length = max(map(lambda x: len(x), result))

            if self.padding_side == 'left':
                attention_mask = map(lambda x: [0] * (max_length - len(x)) + [1] * len(x), result)
                input_ids = map(lambda x: [self.pad_token_id] * (max_length - len(x)) + x, result)
            else:
                attention_mask = map(lambda x: [1] * len(x) + [0] * (max_length - len(x)), result)
                input_ids = map(lambda x: x + [self.pad_token_id] * (max_length - len(x)), result)
        else:
            input_ids = result
            attention_mask = map(lambda x: [1] * len(x), result)

        if truncation:
            if self.padding_side == 'left':
                attention_mask = map(lambda x: x[-max_length:], attention_mask)
                input_ids = map(lambda x: x[-max_length:], input_ids)
            else:
                attention_mask = map(lambda x: x[:max_length], attention_mask)
                input_ids = map(lambda x: x[:max_length], input_ids)

        attention_mask = list(attention_mask)
        input_ids = list(input_ids)

        return BatchEncoding(data={'input_ids': input_ids, 'attention_mask': attention_mask}, tensor_type=return_tensors)

    def truncation(self, text, max_length=None):
        """
        :param text:
        :param max_length: int, if None will use defualt max length
        :return:
        """
        text_ids = self.tokenizer.encode(text, allowed_special='all')
        if len(text_ids) > max_length:
            if self.padding_side == 'left':
                text_ids = text_ids[-max_length:]
            else:
                text_ids = text_ids[:max_length]
            text = self.tokenizer.decode(text_ids, errors="ignore").strip('�')
        return text