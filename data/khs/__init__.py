from ..datamodule import InMemoryDataModule
from pathlib import Path
import torch

# TODO: remove tokenizer dependency
class KHSDataModule(InMemoryDataModule):
    def _load(self, set):
        root = Path('data/khs')
        if set == 'train':
            p = root / 'train.pt'
        elif set == 'val':
            p = root / 'validation.pt'
        else:
            raise NotImplementedError
        return torch.load(p)