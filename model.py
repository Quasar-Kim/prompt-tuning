from functools import partial
from transformers import T5ForConditionalGeneration
import torch
from torch import optim
from torch.optim.lr_scheduler import LambdaLR

from t2tpipe import feature_converter, Model
from t2tpipe.base import BaseLightningModule
from t2tpipe.dataclass import EncDecSampleForTrain, EncDecSampleForInference, ModelInferenceOutput, ModelTrainOutput
from tokenizer import KeT5Tokenizer


class InvSqrtScheduler(LambdaLR):
    def __init__(self, optimizer, warmup_epochs: int):
        super().__init__(optimizer, lr_lambda=self._lr_lambda)
        self._warmup_epochs = warmup_epochs

    def _lr_lambda(self, epoch):
        if epoch < self._warmup_epochs:
            return epoch / self._warmup_epochs
        return (epoch ** -0.5) * (self._warmup_epochs ** 0.5)

class KeT5Module(BaseLightningModule):
    def __init__(self, model_name: str):
        super().__init__()
        self.module = T5ForConditionalGeneration.from_pretrained(model_name)
    
    def forward(self, batch: EncDecSampleForTrain):
        return self.module(
            input_ids=batch.enc_x,
            attention_mask=batch.enc_attention_mask,
            decoder_input_ids=batch.dec_x,
            decoder_attention_mask=batch.dec_attention_mask,
            labels=batch.y
        ) # type: ignore

    def _step_train(self, batch: EncDecSampleForTrain) -> ModelTrainOutput:
        output = self(batch)
        y_pred = torch.argmax(output.logits, dim=-1)
        return ModelTrainOutput(
            y=batch.y,
            y_pred=y_pred,
            loss=output.loss
        )
    
    def _step_inference(self, batch: EncDecSampleForInference) -> ModelInferenceOutput:
        raise NotImplementedError()
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self._env.runtime_config['lr'])
        lr_scheduler = InvSqrtScheduler(optimizer, warmup_epochs=self._env.runtime_config['num_warmup_epochs'])
        return { 'optimizer': optimizer, 'lr_scheduler': lr_scheduler }
    
class KeT5SmallModule(KeT5Module):
    def __init__(self):
        super().__init__('KETI-AIR/ke-t5-small')

ket5_small = Model(
    feature_converter=feature_converter.EncDecFeatureConverter(),
    module=KeT5SmallModule(),
    tokenizer=KeT5Tokenizer()
)