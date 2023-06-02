from typing import Any, List

import torch
from torch import nn, optim

from t2tpipe import datasource, datapipe, postprocessor, metric, feature_converter, Task, Model
from t2tpipe.base import BaseLightningModule
from t2tpipe.tokenizer import Tokenizer
from t2tpipe.dataclass import EncDecSampleForTrain, EncDecSampleForInference, ModelInferenceOutput, ModelTrainOutput
from t2tpipe.type import TextSampleForTrain

class DummyTokenizer(Tokenizer):    
    @property
    def pad_token_id(self):
        return -1
    
    @property
    def unk_token_id(self):
        return -2
    
    @property
    def eos_token_id(self):
        return -3
    
    @property
    def bos_token_id(self):
        return -4
    
    @property
    def pad_token(self):
        return '[PAD]'
    
    @property
    def unk_token(self):
        return '[UNK]'
    
    @property
    def eos_token(self):
        return '[EOS]'
    
    @property
    def bos_token(self):
        return '[BOS]'
    
    def encode(self, text: str) -> List[int]:
        encoded = list(text.encode())
        return encoded
    
    def decode(self, ids: List[int], remove_special_tokens: bool = False) -> str:
        return ''.join([chr(i) for i in ids if i > 0])

class DummyModule(BaseLightningModule):
    def __init__(self):
        super().__init__()
        self._model = nn.Linear(1, 100)
    
    def forward(self, batch: EncDecSampleForTrain):
        x = batch.enc_x.unsqueeze(-1).float() # (B, N, 1)
        y = self._model(x) # (B, N, 100)
        return y, y.mean()

    def _step_train(self, batch: EncDecSampleForTrain) -> ModelTrainOutput:
        logits, loss = self(batch)
        y_pred = torch.argmax(logits, dim=-1)
        return ModelTrainOutput(
            y=batch.y,
            y_pred=y_pred,
            loss=loss
        )
    
    def _step_inference(self, batch: EncDecSampleForInference) -> ModelInferenceOutput:
        raise NotImplementedError()
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
class DummyPostProcessor(postprocessor.PostProcessor):
    def __init__(self):
        self._tokenizer = DummyTokenizer()
    
    def __call__(self, y_or_y_pred: str) -> Any:
        return self._tokenizer.encode(y_or_y_pred)[:1]
    
dummy_model = Model(
    name='dummy_model',
    feature_converter=feature_converter.EncDecFeatureConverter(),
    module=DummyModule(),
    tokenizer=DummyTokenizer()
)

def to_sample(sample: int) -> TextSampleForTrain:
    return {
        'x': str(sample),
        'y': str(sample)
    }

dummy_task = Task(
    name='dummy',
    source={
        'train': datasource.IterableDataSource(range(100)),
        'validation': datasource.IterableDataSource(range(100)),
        'test': datasource.IterableDataSource(range(100)),
        'prediction': datasource.IterableDataSource(range(100))
    },
    pipes=[
        datapipe.Shuffler(),
        datapipe.DistributedShardingFilter(),
        datapipe.WorkerShardingFilter(),
        datapipe.Mapper(fn=to_sample),
        datapipe.FeatureTokenizer()
    ],
    pad_to=10,
    postprocessor=DummyPostProcessor()
)