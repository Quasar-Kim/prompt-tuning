import torch

from dataclasses import asdict
from typing import Union

from t2tpipe.dataclass import Env, ModelInferenceOutput, Task, Model
from t2tpipe.datasource import IterableDataSource
from t2tpipe.feature_converter import NoopFeatureConverter
from t2tpipe.base import BaseLightningModule
from t2tpipe.datamodule import T2tPipeDataModule
from t2tpipe.tokenizer import Tokenizer


class DummyModule(BaseLightningModule):
    def forward(self, batch):
        pass

    def _step_train(self, batch):
        pass

    def _step_inference(self, batch):
        pass

class DummyTokenizer(Tokenizer):
    @property
    def bos_token(self):
        return '[BOS]'
    
    @property
    def eos_token(self):
        return '[EOS]'

    @property
    def unk_token(self):
        return '[UNK]'
    
    @property
    def pad_token(self):
        return '[PAD]'
    
    @property
    def bos_token_id(self):
        return -1
    
    @property
    def eos_token_id(self):
        return -2
    
    @property
    def unk_token_id(self):
        return -3
    
    @property
    def pad_token_id(self):
        return -4

    def encode(self, *args, **kwargs):
        pass

    def encode_batch(self, *args, **kwargs):
        pass

    def decode(self, *args, **kwargs):
        pass

    def decode_batch(self, *args, **kwargs):
        pass

class DummyDataModule(T2tPipeDataModule):
    def __init__(self):
        pass

dummy_task = Task(
    name='dummy',
    source={
        'train': IterableDataSource([]),
    },
    pipes=[]
)

dummy_model = Model(
    name='dummy',
    feature_converter=NoopFeatureConverter(),
    module=DummyModule(),
    tokenizer=DummyTokenizer(),
)

dummy_env = Env(
    model=dummy_model,
    task=dummy_task,
    datamodule=DummyDataModule(),
    runtime_config={},
    inference=False
)

def dataclass_equal(a, b):
    if type(a) != type(b):
        return False
    a_dict = asdict(a)
    b_dict = asdict(b)
    for (k_a, v_a), (k_b, v_b) in zip(a_dict.items(), b_dict.items()):
        if k_a != k_b:
            return False
        if type(v_a) != type(v_b):
            return False
        if isinstance(v_a, torch.Tensor):
            if not tensor_strict_equal(v_a, v_b):
               return False
        else:
            if v_a != v_b:
                return False
    return True

def tensor_strict_equal(a: torch.Tensor, b: torch.Tensor):
    if not torch.equal(a, b):
        return False
    if a.dtype != b.dtype:
        return False
    return True