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
        self.validation_outputs = []
        self.tokenizer = tokenizer
        self.postprocessor = postprocessor
        self.postprocessor.tokenizer = tokenizer

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.train_losses.append(outputs['loss'])

    def on_train_epoch_end(self):
        self._log_loss(self.train_losses, stage='train')
        self.train_losses.clear()

    def on_validation_batch_end(self, outputs, batch: EncDecLabeledSample, batch_idx) -> None:
        self.validation_outputs.append(outputs)

    def on_validation_epoch_end(self):
        outputs = self._postprocess_outputs()
        self._log_loss([out['loss'] for out in outputs], stage='validation')
        self._log_metrics(outputs)
        self.validation_outputs.clear()

    def _log_loss(self, losses: list, stage):
        loss = torch.stack(losses).mean()
        self.log(f'{stage}/loss', loss, sync_dist=True, prog_bar=True)

    def _postprocess_outputs(self):
        if self.postprocessor is None:
            return self.validation_outputs
        return [self.postprocessor(out) for out in self.validation_outputs]

    def _log_metrics(self, outputs):
        for metric_name, metric in self.metrics.items():
            for out in outputs:
                metric.update(out['preds'], out['y'])
            self.log(f'validation/{metric_name}', metric)
        