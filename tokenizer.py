from transformers import T5TokenizerFast
from local_types import *

class KeT5Tokenizer:
    def __init__(self):
        self._tokenizer = T5TokenizerFast.from_pretrained('KETI-AIR/ke-t5-small')
    
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

    def _encode(self, text: str, pad_to: Union[int, None] = None, truncate = False):
        if pad_to is not None:
            encoded = self._tokenizer(text, padding='max_length', truncation=truncate, max_length=pad_to)
        else:
            encoded = self._tokenizer(text)
        return encoded
    
    def encode(self, *args, **kwargs) -> TokenizerEncoding:
        encoded = self._encode(*args, **kwargs)
        return TokenizerEncoding(
            input_ids=encoded['input_ids'],
            attention_mask=encoded['attention_mask']
        )
    
    def encode_batch(self, *args, **kwargs) -> BatchTokenizerEncoding:
        encoded = self._encode(*args, **kwargs)
        return BatchTokenizerEncoding(
            input_ids=encoded['input_ids'],
            attention_mask=encoded['attention_mask']
        )

    def decode(self, token_ids: 'list[int]', remove_special_tokens: bool = False) -> str:
        return self._tokenizer.decode(token_ids, skip_special_tokens=remove_special_tokens)
    
    def decode_batch(self, batch_token_ids: 'list[list[int]]', remove_special_tokens: bool = False) -> 'list[str]':
        return self._tokenizer.batch_decode(batch_token_ids, skip_special_tokens=remove_special_tokens)