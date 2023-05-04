from torch import nn
from lightning.pytorch import LightningModule
from local_types import *

class LightningModuleWithMetrics(LightningModule):
    tokenizer: Tokenizer
    postprocessor: PostProcessor | None

    def __init__(self, tokenizer, metrics, postprocessor=None):
        super().__init__()
        self.metrics = nn.ModuleDict(metrics)
        self.train_losses = []
        self.validation_losses = []
        self.tokenizer = tokenizer
        self.postprocessor = postprocessor
        self.postprocessor.tokenizer = tokenizer

    def on_train_batch_end(self, outputs, batch: EncDecLabeledSample, batch_idx):
        self.train_losses.append(outputs['loss'])
        return self._postprocess_and_update_metrics(outputs, batch)

    def on_validation_batch_end(self, outputs, batch: EncDecLabeledSample, batch_idx) -> None:
        self.validation_losses.append(outputs['loss'])
        return self._postprocess_and_update_metrics(outputs, batch)

    def _postprocess_and_update_metrics(self, outputs, batch: EncDecLabeledSample):
        if self.postprocessor is not None:
            outputs = self.postprocessor(outputs, batch)
        self._update_metrics(outputs)

    def _update_metrics(self, outputs: ModelStepOutputs):
        for metric in self.metrics.values():
            metric.update(outputs['preds'], outputs['y'])

    def on_train_epoch_end(self):
        self._log_loss(stage='train')
        self._log_metrics(stage='train')

    def on_validation_epoch_end(self):
        self._log_loss(stage='validation')
        self._log_metrics(stage='validation')

    def _log_loss(self, stage):
        if stage == 'train':
            losses = self.train_losses
        elif stage == 'validation':
            losses = self.validation_losses
        else:
            raise ValueError('typo')
        loss = torch.stack(losses).mean()
        losses.clear()
        self.log(f'{stage}/loss', loss, sync_dist=True, prog_bar=True)

    def _log_metrics(self, stage):
        for metric_name, metric in self.metrics.items():
            self.log(f'{stage}/{metric_name}', metric)
        