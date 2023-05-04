from typing import Protocol, TypedDict, runtime_checkable
from typing_extensions import NotRequired
import torch

class TextInferenceSample(TypedDict):
    x: str

class TextTrainSample(TextInferenceSample):
    y: str

class EncDecUnlabeledSample(TypedDict):
    enc_x: torch.Tensor
    enc_attention_mask: NotRequired[torch.Tensor]
    enc_x_len: torch.Tensor
    dec_attention_mask: NotRequired[torch.Tensor]

class EncDecLabeledSample(EncDecUnlabeledSample):
    y: torch.Tensor
    actual_y_len: torch.Tensor # special token을 모두 제거했을때 y의 길이

class TokenizerEncoding(TypedDict):
    input_ids: list[int]
    attention_mask: list[int]

class BatchTokenizerEncoding(TypedDict):
    input_ids: list[list[int]]
    attention_mask: list[list[int]]

class Tokenizer(Protocol):
    bos_token: str | None
    eos_token: str | None
    pad_token: str | None
    unk_token: str | None
    bos_token_id: int | None
    eos_token_id: int | None
    pad_token_id: int | None
    unk_token_id: int | None

    def encode(self, text: str, pad_to: int | None = None, **kwargs) -> TokenizerEncoding:
        pass

    def encode_batch(self, batch: list[str], pad_to: int | None = None, **kwargs) -> BatchTokenizerEncoding:
        pass

    def decode(self, token_ids: list[int], remove_special_tokens: bool = False, **kwargs) -> str:
        pass

    def decode_batch(self, batch_token_ids: list[list[int]], remove_special_tokens: bool = False, **kwargs) -> list[str]:
        pass

class FeatureConverter(Protocol):
    tokenizer: Tokenizer
    
    def __init__(self, tokenizer):
        pass
    
    def convert(self, sample: TextTrainSample | TextInferenceSample):
        pass

class ModelStepOutputs(TypedDict):
    loss: torch.Tensor
    preds: NotRequired[torch.Tensor]
    labels: NotRequired[torch.Tensor]

class PostProcessor(Protocol):
    def __call__(self, outputs, batch) -> dict:
        pass