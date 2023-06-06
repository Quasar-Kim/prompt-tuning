from abc import ABC, abstractmethod
from typing import List, Union


class Tokenizer(ABC):
    @property
    def bos_token(self) -> Union[str, None]:
        return None

    @property
    def eos_token(self) -> Union[str, None]:
        return None

    @property
    def pad_token(self) -> Union[str, None]:
        return None

    @property
    def unk_token(self) -> Union[str, None]:
        return None

    @property
    def bos_token_id(self) -> Union[int, None]:
        return None

    @property
    def eos_token_id(self) -> Union[int, None]:
        return None

    @property
    def pad_token_id(self) -> Union[int, None]:
        return None

    @property
    def unk_token_id(self) -> Union[int, None]:
        return None

    @abstractmethod
    def encode(self, text: str) -> List[int]:
        pass

    def encode_batch(self, batch: List[str]) -> List[List[int]]:
        encoded = []
        for text in batch:
            encoded.append(self.encode(text))
        return encoded

    @abstractmethod
    def decode(self, ids: List[int], remove_special_tokens: bool = False) -> str:
        pass

    def decode_batch(
        self, batch_ids: List[List[int]], remove_special_tokens: bool = False
    ) -> List[str]:
        decoded = []
        for ids in batch_ids:
            decoded.append(self.decode(ids, remove_special_tokens))
        return decoded
