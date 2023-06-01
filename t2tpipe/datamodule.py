from typing import Optional, Callable, List
from logging import Logger
import dataclasses
import copy

from torch.utils.data import DataLoader, default_collate
from torch.utils.data import IterDataPipe as TorchIterDataPipe
from lightning.pytorch import LightningDataModule

from t2tpipe.dataclass import Env
from t2tpipe.dataclass import EncDecSampleForTrain

log = Logger(__name__)

class T2tPipeDataModule(LightningDataModule):
    env: Env
    train_datapipe: Optional[TorchIterDataPipe] = None
    validation_datapipe: Optional[TorchIterDataPipe] = None
    test_datapipe: Optional[TorchIterDataPipe] = None

    def configure(self, env: Env):
        self.env = env
        self._validate_runtime_config()

    def _validate_runtime_config(self):
        assert 'num_workers' in self.env.runtime_config and isinstance(self.env.runtime_config['num_workers'], int)
        if self.env.runtime_config['num_workers'] > 16:
            log.warning('Too many workers (usually > 16) might result in actual batch size smaller than set batch size.')
        assert 'batch_size' in self.env.runtime_config and isinstance(self.env.runtime_config['batch_size'], int)
        assert 'is_gpu' in self.env.runtime_config and isinstance(self.env.runtime_config['is_gpu'], bool)

    def setup(self, stage):
        if stage not in ['fit', 'validate', 'test']:
            raise NotImplementedError
        if stage == 'fit':
            self.train_datapipe = self._build_datapipe('train')
        if (stage == 'fit' or stage == 'validate') and 'validation' in self.env.task.source:
            self.validation_datapipe = self._build_datapipe('validation')
        if stage == 'test':
            self.test_datapipe = self._build_datapipe('test')

    def _build_datapipe(self, stage: str) -> TorchIterDataPipe:
        assert stage in self.env.task.source, f'No data source for stage {stage}'
        dp = self.env.task.source[stage]
        dp.setup(self.env)
        
        transform_pipes = [
            *self.env.task.pipes,
            self.env.model.feature_converter
        ]
        
        padder = self.env.model.padder
        if padder is not None and self.env.pad_to is not None:
            transform_pipes.append(padder)

        transform_pipes = copy.deepcopy(transform_pipes)

        for datapipe in transform_pipes:
            datapipe.setup(self.env)
            dp = dp.connect(datapipe)
        return dp
    
    @property
    def train_dataloader(self) -> Callable:
        # NOTE: when this property is accessed by trainer,
        # setup() is not called so train_datapipe is None
        # so check env.task.source instead
        if 'train' not in self.env.task.source:
            raise AttributeError
        return self._train_dataloader

    def _train_dataloader(self):
        assert self.train_datapipe is not None
        return self._create_dataloader(
            self.train_datapipe,
            shuffle=True
        )
    
    @property
    def val_dataloader(self) -> Callable:
        if 'validation' not in self.env.task.source:
            raise AttributeError
        return self._val_dataloader

    def _val_dataloader(self):
        assert self.validation_datapipe is not None
        return self._create_dataloader(
            self.validation_datapipe,
            shuffle=False
        )
    
    @property
    def test_dataloader(self) -> Callable:
        if 'test' not in self.env.task.source:
            raise AttributeError
        return self._test_dataloader
    
    def _test_dataloader(self):
        assert self.test_datapipe is not None
        return self._create_dataloader(
            self.test_datapipe,
            shuffle=False
        )
    
    def _create_dataloader(self, datapipe, **kwargs):
        return DataLoader(
            datapipe,
            num_workers=self.env.runtime_config['num_workers'],
            batch_size=self.env.runtime_config['batch_size'],
            pin_memory=self.env.runtime_config['is_gpu'],
            collate_fn=_collate_dataclass,
            **kwargs
        )

# TODO: typing dataclass?
def _collate_dataclass(samples: List):
    assert dataclasses.is_dataclass(samples[0])
    dataclass_cls = type(samples[0])
    dict_samples = [dataclasses.asdict(s) for s in samples]
    collated = default_collate(dict_samples)
    return dataclass_cls(**collated)