from multiprocessing import cpu_count
from lightning.pytorch import LightningModule, LightningDataModule
from lightning.pytorch.utilities.types import EVAL_DATALOADERS
import torch
from torch.utils.data import DataLoader
from typing import Tuple, Union
from torchdata.datapipes import iter as iterpipes
from .pipes import FeatureTokenizer
from local_types import *


class BaseLightningModule(LightningModule):
    tokenizer: Tokenizer
    postprocessor: Union[PostProcessor, None]
    metrics: 'list[Metric]'
    config: dict
    _validation_outputs: 'list[ModelStepOutput]'

    def __init__(self):
        super().__init__()
        self._configured = False
        self._train_losses = []
        self._validation_outputs = []

    def configure(
        self,
        tokenizer: Tokenizer,
        config: dict,
        postprocessor: Union[PostProcessor, None] = None,
        metrics: Union['list[Metric]', None] = None,
    ):
        self.tokenizer = tokenizer
        self.postprocessor = postprocessor
        self.metrics = metrics if metrics is not None else []
        self.config = config
        self.hp = config['hp'] if 'hp' in config else {}
        self._configured = True

    def training_step(self, batch, batch_idx):
        return self._step(batch)
    
    def validation_step(self, batch, batch_idx):
        return self._step(batch)

    def forward(self, batch):
        raise NotImplementedError()
    
    def _step(self, batch):
        raise NotImplementedError()

    def on_train_batch_end(self, outputs: dict, batch, batch_idx):
        self._train_losses.append(outputs['loss'])
    
    def on_train_epoch_end(self):
        self._log_loss(torch.stack(self._train_losses), stage='train')
        self._train_losses.clear()

    def on_validation_batch_end(self, outputs, batch, batch_idx) -> None:
        self._validation_outputs.append(outputs)

    def on_validation_epoch_end(self):
        outputs = self._collate_outputs() # v: (number of samples, N)
        self._log_loss(outputs['loss'], stage='validation')
        self._compute_metrics(outputs)

    def _collate_outputs(self):
        def stack_or_cat(tensors: 'list[torch.Tensor]'):
            if tensors[0].ndim > 0:
                return torch.cat(tensors)
            else:
                return torch.stack(tensors)
        outputs = {
            k: stack_or_cat([out[k] for out in self._validation_outputs])
            for k in self._validation_outputs[0].keys()
        }
        self._validation_outputs.clear()
        return outputs

    def _log_loss(self, losses: torch.Tensor, stage):
        loss = losses.mean()
        self.log(f'{stage}/loss', loss, sync_dist=True, prog_bar=True)

    def _compute_metrics(self, outputs):
        if self.postprocessor is None:
            return
        target_keys = ['y', 'y_pred']
        out = {}
        for k in target_keys:
            detokenized = self.tokenizer.decode_batch(outputs[k].tolist(), remove_special_tokens=True)
            out[k] = torch.tensor([
                self.postprocessor(s, tokenizer=self.tokenizer) 
                for s in detokenized
            ])
        for metric in self.metrics:
            val = metric(out['y'], out['y_pred'])
            self.log(f'validation/{metric.name}', val, sync_dist=True, reduce_fx=metric.reduce_fx)

class DataPipeDataModule(LightningDataModule):
    def __init__(
        self,
        source: 'dict[str, iterpipes.IterDataPipe]',
        pipes: 'list[DataPipe]',
        feature_converter: DataPipe,
        tokenizer: Tokenizer,
        config: dict
    ):
        super().__init__()
        self.source = source
        self.pipes = pipes + [
            FeatureTokenizer(),
            feature_converter
        ]
        self.tokenizer = tokenizer
        self.config = config

    def setup(self, stage):
        if stage == 'fit':
            self.train_datapipe = self._build_datapipe('train')
        if stage == 'fit' or stage == 'validate':
            self.validation_datapipe = self._build_datapipe('validation')
        if stage == 'test':
            self.test_datapipe = self._build_datapipe('test')

    def _build_datapipe(self, stage: str) -> iterpipes.IterDataPipe:
        source_pipe = self.source[stage]
        dp = source_pipe
        for pipe in self.pipes:
            dp = pipe(dp, self) # type: ignore
        return dp # type: ignore
        
    def train_dataloader(self):
        return self._create_dataloader(
            self.train_datapipe,
            shuffle=True
        )
    
    @property
    def val_dataloader(self):
        if not hasattr(self, 'validation_datapipe'):
            raise AttributeError
        return self._val_dataloader

    def _val_dataloader(self):
        return self._create_dataloader(
            self.validation_datapipe,
            shuffle=False
        )
    
    @property
    def test_dataloader(self):
        if not hasattr(self, 'test_datapipe'):
            raise AttributeError
        return self._test_dataloader
    
    def _test_dataloader(self):
        return self._create_dataloader(
            self.test_datapipe,
            shuffle=False
        )
    
    def _create_dataloader(self, datapipe, **kwargs):
        return DataLoader(
            datapipe,
            num_workers=self.config['num_workers'], 
            batch_size=self.config['batch_size'],
            pin_memory=self.config['is_gpu'],
            **kwargs
        )
        

def create_experiment(
    task: Task,
    model: Model, 
    batch_size, 
    is_gpu=False,
    pad_to=None,
    num_workers=None,
    **kwargs
) -> Tuple[BaseLightningModule, DataPipeDataModule]:
    config = {
        'batch_size': batch_size,
        'is_gpu': is_gpu,
        # cpu count가 극단적으로 높은 경우 (예: TPU VM은 96개)
        # batch size가 지켜지지 않을 수 있으므로 max 제한 검
        # config에서 설정하는 것에 제일 좋음
        'num_workers': min(cpu_count(), 16) if num_workers is None else num_workers,
        **kwargs
    }

    tokenizer = model.tokenizer()
    feature_converter = model.feature_converter(pad_to=pad_to) # type: ignore
    datamodule = DataPipeDataModule(
        source=task.source,
        pipes=task.pipes,
        feature_converter=feature_converter,
        tokenizer=tokenizer,
        config=config
    )
    module = model.module()
    module.configure( # type: ignore
        tokenizer=tokenizer,
        config=config,
        postprocessor=task.postprocessor,
        metrics=task.metrics
    ) 
    return module, datamodule # type: ignore