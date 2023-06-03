from typing import List

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from t2tpipe.tokenizer import Tokenizer


class HfTokenizer(Tokenizer):
    _tokenizer: PreTrainedTokenizerBase

    def __init__(self, model_name: str):
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)

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
        return self._tokenizer.bos_token_id

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
        return self._tokenizer.bos_token

    def encode(self, text: str) -> List[int]:
        encoded = self._tokenizer.encode(text, add_special_tokens=False)
        return encoded

    def decode(self, ids: List[int], remove_special_tokens: bool = False) -> str:
        return self._tokenizer.decode(ids, skip_special_tokens=remove_special_tokens)

    def decode_batch(
        self, batch_ids: List[List[int]], remove_special_tokens: bool = False
    ) -> List[str]:
        return self._tokenizer.batch_decode(
            batch_ids, skip_special_tokens=remove_special_tokens
        )


class KeT5Tokenizer(HfTokenizer):
    def __init__(self):
        super().__init__("KETI-AIR/ke-t5-small")

    @property
    def bos_token_id(self):
        # implementation detail
        return self.pad_token_id

    @property
    def bos_token(self):
        # implementation detail
        return self.pad_token
