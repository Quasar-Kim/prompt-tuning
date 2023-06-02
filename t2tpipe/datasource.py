from abc import abstractmethod
from typing import Optional
from pathlib import Path

import pandas as pd

from t2tpipe.datapipe import DataPipe

class DataSource(DataPipe):
    @abstractmethod
    def __len__(self) -> int:
        pass

class IterableDataSource(DataSource):
    def __init__(self, iterable):
        self._iterable = iterable

    def __iter__(self):
        yield from self._iterable

    def __len__(self):
        return len(self._iterable)
    
class ParquetDataSource(DataSource):
    def __init__(self, path: str):
        self._path = Path(path)

    def setup(self, env):
        super().setup(env)
        self._df = pd.read_parquet(self._path)

    def __iter__(self):
        assert self._df is not None, 'setup() must be called before __iter__()'
        for row in self._df.itertuples(index=False):
            yield row._asdict()

    def __len__(self):
        assert self._df is not None, 'setup() must be called before __len__()'
        return len(self._df)