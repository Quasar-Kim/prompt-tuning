from functools import partial
from transformers import T5ForConditionalGeneration
import torch
from torch import optim
from torch.optim.lr_scheduler import LambdaLR
from data.core import BaseLightningModule
from data.feature_converters import EncDecFeatureConverter
from tokenizer import KeT5Tokenizer
from local_types import *

def inv_sqrt_lambda(epoch, warmup_epochs):
    if epoch < warmup_epochs:
        return epoch / warmup_epochs
    return (epoch ** -0.5) * (warmup_epochs ** 0.5)

class KeT5Module(BaseLightningModule):
    def __init__(self, model_name: str):
        super().__init__()
        self.module = T5ForConditionalGeneration.from_pretrained(model_name)
    
    def forward(self, batch: EncDecSample):
        return self.module(
            input_ids=batch['enc_x'],
            attention_mask=batch['enc_attention_mask'],
            decoder_input_ids=batch['dec_x'],
            decoder_attention_mask=batch['dec_attention_mask'],
            labels=batch['y']
        ) # type: ignore

    def _step(self, batch: EncDecSample) -> ModelStepOutput:
        output = self(batch)
        y_pred = torch.argmax(output.logits, dim=-1)
        return {'loss': output.loss, 'y': batch['y'], 'y_pred': y_pred} # type: ignore
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hp['lr'])
        lr_scheduler = LambdaLR(optimizer, lr_lambda=partial(inv_sqrt_lambda, warmup_epochs=self.hp['num_warmup_epochs']))
        return { 'optimizer': optimizer, 'lr_scheduler': lr_scheduler }
    
class KeT5SmallModule(KeT5Module):
    def __init__(self):
        super().__init__('KETI-AIR/ke-t5-small')

    
class KeT5BaseModule(KeT5Module):
    def __init__(self):
        super().__init__('KETI-AIR/ke-t5-base')

class KeT5LargeModule(KeT5Module):
    def __init__(self):
        super().__init__('KETI-AIR/ke-t5-large')
    
keT5Small = Model(
    feature_converter=EncDecFeatureConverter, # type: ignore
    module=KeT5SmallModule, # type: ignore
    tokenizer=KeT5Tokenizer # type: ignore
)

keT5Base = Model(
    feature_converter=EncDecFeatureConverter, # type: ignore
    module=KeT5BaseModule, # type: ignore
    tokenizer=KeT5Tokenizer # type: ignore
)

keT5Large = Model(
    feature_converter=EncDecFeatureConverter, # type: ignore
    module=KeT5LargeModule, # type: ignore
    tokenizer=KeT5Tokenizer # type: ignore
)
