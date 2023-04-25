from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader

class InMemoryDataModule(LightningDataModule):
    def __init__(self, batch_size, tokenizer, is_gpu=False):
        super().__init__()
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.is_gpu = is_gpu

    def setup(self, stage):
        if stage == 'fit':
            self.train_dataset = self._load('train')
            self.val_dataset = self._load('val')
        elif stage == 'validate':
            self.val_dataset = self._load('val')
        elif stage == 'predict':
            self.predict_dataset = self._load('predict')
        else:
            raise NotImplementedError
        
    def _load(self, set):
        raise NotImplementedError('load list here')
    
    def train_dataloader(self):
        # num_workers = 0이면 prefetch가 작동하지 않음
        return DataLoader(self.train_dataset, shuffle=True, batch_size=self.batch_size, pin_memory=self.is_gpu, num_workers=1)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, shuffle=False, batch_size=self.batch_size, pin_memory=self.is_gpu, num_workers=1)

    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, shuffle=False, batch_sampler=self.batch_size, pin_memory=self.is_gpu, num_workers=1)
    