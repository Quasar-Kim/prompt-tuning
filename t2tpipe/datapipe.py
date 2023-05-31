from __future__ import annotations
import functools
from itertools import islice
from typing import Tuple, Dict, Any, Type
from abc import ABC, abstractmethod

import torch
import torch.distributed as dist
import torchdata.datapipes.iter as torchdata_pipes
# NOTE: torch.utils.data.IterDataPipe == torchdata.datapipes.iter.IterDataPipe
# but type checker complains if the latter is used
from torch.utils.data import IterDataPipe as TorchIterDataPipe
from torch.utils.data.datapipes.iter.sharding import SHARDING_PRIORITIES

from t2tpipe.mixin import SetupMixin
from t2tpipe.dataclass import EncodedSampleForTrain, EncDecSampleForTrain
from t2tpipe.type import TextSampleForTrain

class DataPipe(ABC, TorchIterDataPipe, SetupMixin):
    def connect(self, dp: TransformDataPipe):
        assert self._env is not None, "DataPipe not setup"
        dp.source_dp = self
        return dp
    
    @abstractmethod
    def __iter__(self):
        pass

    def __len__(self):
        raise NotImplementedError()

class TransformDataPipe(DataPipe):
    _source_dp: DataPipe
    
    @property
    def source_dp(self):
        return self._source_dp

    @source_dp.setter
    def source_dp(self, dp: DataPipe):
        self._source_dp = dp

    def _validate_setup_and_connected(self):
        assert self._env is not None, "DataPipe not setup"
        assert self._source_dp is not None, "DataPipe is not connected to the source"

    def __len__(self):
        return len(self._source_dp)

class TorchDataWrappingTransformDataPipe(TransformDataPipe):
    _torch_datapipe: TorchIterDataPipe
    _torch_datapipe_cls: Type[TorchIterDataPipe]
    _args: Tuple[Any, ...]
    _kwargs: Dict[str, Any]

    def __init__(self, torch_datapipe_cls: Type[TorchIterDataPipe], *args, **kwargs):
        self._torch_datapipe_cls = torch_datapipe_cls
        self._args = args
        self._kwargs = kwargs

    @TransformDataPipe.source_dp.setter
    def source_dp(self, dp: DataPipe):
        self._source_dp = dp
        self._torch_datapipe = self._torch_datapipe_cls(
            self._source_dp,
            *self._args,
            **self._kwargs
        ) # type: ignore

    def __iter__(self):
        self._validate_setup_and_connected()
        yield from self._torch_datapipe

    def __len__(self):
        return len(self._torch_datapipe) # type: ignore

class Mapper(TorchDataWrappingTransformDataPipe):
    def __init__(self, *args, **kwargs):
        super().__init__(torchdata_pipes.Mapper, *args, **kwargs)

class Shuffler(TorchDataWrappingTransformDataPipe):
    def __init__(self, *args, **kwargs):
        super().__init__(torchdata_pipes.Shuffler, *args, **kwargs)

class WorkerShardingFilter(TorchDataWrappingTransformDataPipe):
    def __init__(self, *args, **kwargs):
        super().__init__(
            torchdata_pipes.ShardingFilter, 
            *args,
            sharding_group_filter=SHARDING_PRIORITIES.MULTIPROCESSING,
            **kwargs
        )

class DistributedShardingFilter(TransformDataPipe):
    '''
    DistributedSampler-like sharding datapipe.

    Distributed 훈련 환경에서 각 프로세스가 고유한 샘플을 가지도록 함.
    
    ---
    NOTE: 각 프로세스는 동일한 개수의 샘플을 받음.
    따라서 (number of samples) % (number of processes) > 0 이면 마지막 일부 샘플은 사용되지 않음.

    예시:
        >>> torch.distributed.get_world_size()
        2
        >>> source = IterableDataSource(range(5))
        >>> source.setup(env)
        >>> sharding_filter = DistributedShardingFilter()
        >>> sharding_filter.setup(env)
        >>> dp = source.connect(sharding_filter)
        >>> len(dp)
        4
    '''

    def __iter__(self):
        rank = _get_rank()
        world_size = _get_world_size()
        n_global_samples = len(self)
        yield from islice(iter(self.source_dp), rank, n_global_samples, world_size)

    def __len__(self):
        assert hasattr(self.source_dp, '__len__'), 'DistributedShardingFilter requires source_dp to have length, but it does not'
        n_samples = len(self.source_dp)
        world_size = _get_world_size()
        n_global_samples = (n_samples // world_size) * world_size
        return n_global_samples
    
def _get_world_size() -> int:
    if _is_torch_distributed():
        return dist.get_world_size()
    elif _using_xla():
        import torch_xla.core.xla_model as xm # type: ignore
        return xm.xrt_world_size()
    else:
        return 1
    
def _get_rank() -> int:
    if _is_torch_distributed():
        return dist.get_rank()
    elif _using_xla():
        import torch_xla.core.xla_model as xm # type: ignore
        return xm.get_ordinal()
    else:
        return 0

def _is_torch_distributed():
    return dist.is_available() and dist.is_initialized()

@functools.lru_cache(maxsize=None)
def _using_xla():
    try:
        import torch_xla # type: ignore
        return True
    except ImportError:
        return False

class MappingKeyMapper(Mapper):
    def __init__(self, mapping: Dict[Any, Any], *args, **kwargs):
        self._mapping = mapping
        super().__init__(*args, fn=self._map_key, **kwargs)

    def _map_key(self, sample: Dict[Any, Any]):
        return {self._mapping[k]: v for k, v in sample.items()}


class MappingValueMapper(Mapper):
    def __init__(self, mapping: Dict[Any, Dict[Any, Any]], *args, **kwargs):
        self._mapping = mapping
        super().__init__(*args, fn=self._map_value, **kwargs)

    def _map_value(self, sample: Dict[Any, Any]):
        mapped = {}
        for k, v in sample.items():
            if k in self._mapping:
                mapping = self._mapping[k]
                if v not in mapping:
                    raise KeyError(f'Cannot find mapping for `{v}` from key `{k}`')
                v = mapping[v]
            mapped[k] = v
        return mapped


class FeatureTokenizer(Mapper):
    def __init__(self):
        super().__init__(fn=self._tokenize)

    def _tokenize(self, sample: TextSampleForTrain):
        tokenizer = self._env.model.tokenizer
        return EncodedSampleForTrain(
            x=torch.tensor(tokenizer.encode(sample['x'])),
            y=torch.tensor(tokenizer.encode(sample['y']))
        )

class Padder(Mapper):
    def __init__(self, pad_to: int):
        super().__init__(fn=self._pad_sample)
        self._pad_to = pad_to

    def _pad_sample(self, sample: EncodedSampleForTrain):
        return EncodedSampleForTrain(
            x=self._pad_tensor(sample.x),
            y=self._pad_tensor(sample.y)
        )
        
    def _pad_tensor(self, t: torch.Tensor):
        assert self.pad_value is not None
        pad_len = self._pad_to - t.shape[-1]
        if pad_len < 0:
            raise ValueError(
                f'Cannot pad tensor longer than pad_to: {t.shape[-1]} > {self._pad_to}'
            )
        return torch.nn.functional.pad(t, (0, pad_len), value=self.pad_value)
    
    @property
    def pad_value(self):
        return self._env.model.tokenizer.pad_token_id
    
class PadderForEncDecModel(Mapper):
    def __init__(self):
        super().__init__(fn=self._pad_sample)

    def _pad_sample(self, sample: EncDecSampleForTrain) -> EncDecSampleForTrain:
        padded_enc_x, enc_attention_mask = self._pad_tensor(sample.enc_x)
        padded_dec_x, dec_attention_mask = self._pad_tensor(sample.dec_x)
        padded_y, _ = self._pad_tensor(sample.y)
        return EncDecSampleForTrain(
            enc_x=padded_enc_x,
            dec_x=padded_dec_x,
            y=padded_y,
            enc_attention_mask=enc_attention_mask,
            dec_attention_mask=dec_attention_mask
        )
    
    def _pad_tensor(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        assert self.pad_value is not None, 'Tokenizer should have pad_token_id but it does not'
        assert self._env.pad_to is not None, 'Env should have pad_to but it does not'
        pad_to = self._env.pad_to
        pad_len = pad_to - t.shape[-1]
        if pad_len < 0:
            raise ValueError(
                f'Cannot pad tensor longer than pad_to: {t.shape[-1]} > {pad_to}'
            )
        padded = torch.nn.functional.pad(t, (0, pad_len), value=self.pad_value)
        mask = torch.cat([
            torch.tensor(1).expand(t.shape[-1]),
            torch.tensor(0).expand(pad_len)
        ])
        return padded, mask
    
    @property
    def pad_value(self):
        return self._env.model.tokenizer.pad_token_id

class NoopDataPipe(TransformDataPipe):
    def __iter__(self):
        yield from iter(self.source_dp)