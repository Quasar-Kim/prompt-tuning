import os
from transformers import T5Tokenizer

tokenizer = T5Tokenizer.from_pretrained('KETI-AIR/ke-t5-small')


class KeT5Tokenizer:
    def __init__(self, parallelism=False):
        if not parallelism:
            os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        self._tokenizer = T5Tokenizer.from_pretrained('KETI-AIR/ke-t5-small')
        self.encode = self._encode
        self.encode_batch = self._encode
    
    @property
    def pad_token_id(self):
        return self._tokenizer.pad_token_id
    
    @property
    def unk_token_id(self):
        return self._tokenizer.unk_token_id
    
    @property
    def eos_token_id(self):
        return self._tokenizer.eos_token_id
    
    @property
    def bos_token_id(self):
        # implementation detail: bos token == pad token
        return self._tokenizer.pad_token_id
    
    @property
    def pad_token(self):
        return self._tokenizer.pad_token
    
    @property
    def unk_token(self):
        return self._tokenizer.unk_token
    
    @property
    def eos_token(self):
        return self._tokenizer.eos_token
    
    @property
    def bos_token(self):
        # implementation detail: bos token == pad token
        return self._tokenizer.pad_token_id

    def _encode(self, text: str, pad_to: int | None = None, truncate = False):
        if pad_to is not None:
            encoded = self._tokenizer(text, padding='max_length', truncation=truncate, max_length=pad_to)
        else:
            encoded = self._tokenizer(text)
        return encoded

    def decode(self, token_ids: list[int], remove_special_tokens: bool = False):
        return self._tokenizer.decode(token_ids, skip_special_tokens=remove_special_tokens)
    
    def decode_batch(self, batch_token_ids: list[list[int]], **kwargs):
        return [self.decode(token_ids, **kwargs) for token_ids in batch_token_ids]
