from multiprocessing import cpu_count
from lightning.pytorch import LightningDataModule
from lightning.pytorch.utilities.types import EVAL_DATALOADERS
from torch.utils.data import DataLoader, default_collate
import torchdata.datapipes as dp
import pandas as pd
from local_types import *

class ParquetFileLoader(dp.iter.IterDataPipe):
    def __init__(self, parquet_path):
        self.parquet_path = parquet_path

    def __iter__(self):
        df = pd.read_parquet(self.parquet_path)
        yield from df.itertuples()

class KHSDataModule(LightningDataModule):
    tokenizer: Tokenizer
    feature_converter: FeatureConverter

    def __init__(self, batch_size: int, tokenizer, feature_converter, is_gpu=False, pad_to=128):
        super().__init__()
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.feature_converter = feature_converter
        self.feature_converter.tokenizer = tokenizer
        self.feature_converter.pad_to = pad_to
        self.is_gpu = is_gpu

    def setup(self, stage):
        if stage == 'fit':
            self.train_datapipe = self._build_datapipe('data/khs/train.parquet')
            self.validation_datapipe = self._build_datapipe('data/khs/validation.parquet')
        elif stage == 'predict':
            raise NotImplementedError('훈련 다 만들고 하기')

    def _build_datapipe(self, parquet_file: str):
        def to_feature_converter_input(sample):
            return {
                'x': sample.text,
                'y': sample.label
            }
        def map_label(sample):
            mapping = {
                'hate': '혐오',
                'offensive': '공격적',
                'none': '중립'
            }
            return {
                'x': sample['x'],
                'y': mapping[sample['y']]
            }

        pipe = ParquetFileLoader(parquet_file)
        pipe = pipe.shuffle()
        pipe = pipe.sharding_filter()
        pipe = pipe.map(to_feature_converter_input)
        pipe = pipe.map(map_label)
        pipe = pipe.map(self.feature_converter.convert)
        if self.is_gpu:
            pipe = pipe.pin_memory()
        return pipe

    def train_dataloader(self):
        return DataLoader(self.train_datapipe, num_workers=cpu_count(), batch_size=self.batch_size)
    
    def val_dataloader(self):
        return DataLoader(self.validation_datapipe, num_workers=cpu_count(), batch_size=self.batch_size)

    
