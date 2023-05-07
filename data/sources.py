import pandas as pd
from torchdata.datapipes import iter as iterpipes

class ParquetFileLoader(iterpipes.IterDataPipe):
    def __init__(self, parquet_path):
        self.parquet_path = parquet_path

    def __iter__(self):
        df = pd.read_parquet(self.parquet_path)
        for t in df.itertuples(index=False):
            yield t._asdict()

    def __len__(self):
        df = pd.read_parquet(self.parquet_path)
        return len(df)