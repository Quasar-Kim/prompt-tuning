from functools import partial
import torch
from torch import optim
from torch.optim.lr_scheduler import LambdaLR
from transformers import T5ForConditionalGeneration
from local_types import *
from local_types import EncDecLabeledSample, ModelStepOutput
from .base import LightningModuleWithMetrics

def inv_sqrt_lambda(epoch, warmup_epochs):
    if epoch < warmup_epochs:
        return epoch / warmup_epochs
    return (epoch ** -0.5) * (warmup_epochs ** 0.5)

def get_output_length_without_padding(decoder_attention_mask: torch.Tensor):
    return decoder_attention_mask.nonzero().squeeze()[-1:].squeeze() - torch.tensor(1)

class KeT5(LightningModuleWithMetrics):
    def __init__(self, *, variant, lr, num_warmup_epochs, **kwargs):
        super().__init__(**kwargs)
        self.save_hyperparameters(ignore=['variant', 'Tokenizer_cls', 'metrics', 'postprocessor'])
        self.model = T5ForConditionalGeneration.from_pretrained(f'KETI-AIR/ke-t5-{variant}')

    def forward(self, sample: EncDecLabeledSample):
        return self.model(
            input_ids=sample['enc_x'],
            attention_mask=sample['enc_attention_mask'],
            decoder_input_ids=sample['dec_x'],
            decoder_attention_mask=sample['dec_attention_mask'],
            labels=sample['y']
        )

    def training_step(self, batch: EncDecLabeledSample, batch_idx) -> ModelStepOutput:
        return self._step(batch)
    
    def validation_step(self, batch: EncDecLabeledSample, batch_idx) -> ModelStepOutput:
        return self._step(batch)
    
    def _step(self, batch: EncDecLabeledSample) -> ModelStepOutput:
        output = self(batch)
        ret = {'loss': output.loss, 'logits': output.logits}
        if 'cls_y' in batch:
            ret['cls_y'] = batch['cls_y']
        return ret
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        lr_scheduler = LambdaLR(optimizer, lr_lambda=partial(inv_sqrt_lambda, warmup_epochs=self.hparams.num_warmup_epochs))
        return { 'optimizer': optimizer, 'lr_scheduler': lr_scheduler }

class KeT5Small(KeT5):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, variant='small')

class KeT5Base(KeT5):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, variant='base')

class KeT5Large(KeT5):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, variant='large')