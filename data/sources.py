from itertools import islice
import pandas as pd
from torchdata.datapipes import iter as iterpipes
import torch.distributed as dist
import functools

@functools.lru_cache(maxsize=None)
def _is_distributed():
    return dist.is_available() and dist.is_initialized()

@functools.lru_cache(maxsize=None)
def _using_xla():
    try:
        import torch_xla
        return True
    except ImportError:
        return False

class ParquetFileLoader(iterpipes.IterDataPipe):
    def __init__(self, parquet_path):
        self.parquet_path = parquet_path

    def __iter__(self):
        df = pd.read_parquet(self.parquet_path)
        if _is_distributed():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            n_actual_samples = (len(df) // world_size) * world_size
        elif _using_xla():
            import torch_xla.core.xla_model as xm
            rank = xm.get_ordinal()
            world_size = xm.xrt_world_size()
            n_actual_samples = (len(df) // world_size) * world_size
        else:
            rank = 0
            world_size = 1
            n_actual_samples = len(df)
        for t in islice(df.itertuples(index=False), rank, n_actual_samples, world_size):
            yield t._asdict()