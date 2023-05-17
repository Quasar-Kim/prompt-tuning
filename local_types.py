from __future__ import annotations
from typing import Protocol, NamedTuple, TypedDict, Type, Union
import torch
from torchdata.datapipes import iter as iterpipes
from lightning.pytorch import LightningDataModule

class TokenizerEncoding(NamedTuple):
    input_ids: 'list[int]'
    attention_mask: 'list[int]'

class BatchTokenizerEncoding(NamedTuple):
    input_ids: 'list[list[int]]'
    attention_mask: 'list[list[int]]'

class Tokenizer(Protocol):
    bos_token: Union[str, None]
    eos_token: Union[str, None]
    pad_token: Union[str, None]
    unk_token: Union[str, None]
    bos_token_id: int
    eos_token_id: int
    pad_token_id: Union[int, None]
    unk_token_id: Union[int, None]

    def encode(self, text: str, pad_to: Union[int, None] = None, **kwargs) -> TokenizerEncoding:
        ...

    def encode_batch(self, batch: 'list[str]', pad_to: Union[int, None] = None, **kwargs) -> BatchTokenizerEncoding:
        ...

    def decode(self, token_ids: 'list[int]', remove_special_tokens: bool = False, **kwargs) -> str:
        ...

    def decode_batch(self, batch_token_ids: 'list[list[int]]', remove_special_tokens: bool = False, **kwargs) -> 'list[str]':
        ...

class EncDecSample(TypedDict):
    enc_x: torch.Tensor
    enc_attention_mask: torch.Tensor
    dec_x: torch.Tensor
    dec_attention_mask: torch.Tensor
    y: torch.Tensor

class DecSample(TypedDict):
    ...

Sample = Union[EncDecSample, DecSample]

class DetokenizedOutput(TypedDict):
    y: 'list[str]'
    y_pred: 'list[str]'

class PostProcessor(Protocol):
    def __call__(self, y_or_y_pred: str, tokenizer: Tokenizer) -> int:
        ...

class Metric(Protocol):
    name: str
    reduce_fx: str

    def __call__(self, y: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        ...

class DataPipe(Protocol):
    def __call__(self, dp: iterpipes.IterDataPipe, dm: LightningDataModule) -> iterpipes.IterDataPipe:
        ...

class Task(NamedTuple):
    name: str
    source: 'dict[str, iterpipes.IterDataPipe]'
    pipes: list
    postprocessor: Union[PostProcessor, None]
    metrics: Union['list[Metric]', None]

class Model(NamedTuple):
    feature_converter: Type[DataPipe]
    module: Type[LightningDataModule]
    tokenizer: Type[Tokenizer]

class ModelStepOutput(TypedDict):
    loss: torch.Tensor
    y: torch.Tensor
    y_pred: torch.Tensor

class FeatureConverterInput(TypedDict):
    x: 'list[int]'
    x_attention_mask: 'list[int]'
    y: 'list[int]'
    y_attention_mask: 'list[int]'