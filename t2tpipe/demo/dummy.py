from typing import Any, List, Union, TypeVar
import dataclasses

import torch
from torch import nn, optim

from t2tpipe import (
    datasource,
    datapipe,
    postprocessor,
    metric,
    feature_converter,
    Task,
    Model,
)
from t2tpipe.base import BaseLightningModule
from t2tpipe.tokenizer import Tokenizer
from t2tpipe.dataclass import (
    EncDecSampleForTrain,
    EncDecSampleForPrediction,
    ModelPredictionOutput,
    ModelTrainOutput,
)
from t2tpipe.type import TextSampleForTrain
from t2tpipe.util import join_tensors

_MODEL_OUTPUT_T = TypeVar("_MODEL_OUTPUT_T", ModelTrainOutput, ModelPredictionOutput)


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
        return "[PAD]"

    @property
    def unk_token(self):
        return "[UNK]"

    @property
    def eos_token(self):
        return "[EOS]"

    @property
    def bos_token(self):
        return "[BOS]"

    def encode(self, text: str) -> List[int]:
        encoded = list(text.encode())
        return encoded

    def decode(self, ids: List[int], remove_special_tokens: bool = False) -> str:
        return "".join([chr(i) for i in ids if i > 0])


class DummyModule(BaseLightningModule):
    def __init__(self):
        super().__init__()
        self._model = nn.Linear(1, 100)

    def forward(self, batch: EncDecSampleForTrain):
        x = batch.enc_x.unsqueeze(-1).float()  # (B, N, 1)
        y = self._model(x)  # (B, N, 100)
        return y, y.mean()

    def _step_train(self, batch: EncDecSampleForTrain) -> ModelTrainOutput:
        _, loss = self(batch)
        return ModelTrainOutput(x=batch.enc_x, y=batch.y, y_pred=batch.y, loss=loss)

    def _step_prediction(
        self, batch: EncDecSampleForPrediction
    ) -> ModelPredictionOutput:
        return ModelPredictionOutput(x=batch.enc_x, y_pred=batch.enc_x)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


class MultiplyPostProcessor(postprocessor.PostProcessor):
    def __call__(self, outputs: _MODEL_OUTPUT_T) -> _MODEL_OUTPUT_T:
        if self._env.prediction:
            assert isinstance(outputs, ModelPredictionOutput)
            return self._process_prediction_output(outputs)
        assert isinstance(outputs, ModelTrainOutput)
        return self._process_train_output(outputs)

    def _process_train_output(self, outputs: ModelTrainOutput) -> ModelTrainOutput:
        return dataclasses.replace(
            outputs,
            y=self._postprocess(outputs.y),
            y_pred=self._postprocess(outputs.y_pred),
        )

    def _process_prediction_output(
        self, outputs: ModelPredictionOutput
    ) -> ModelPredictionOutput:
        return dataclasses.replace(
            outputs,
            x=self._postprocess(outputs.x),
            y_pred=self._postprocess(outputs.y_pred),
        )

    def _postprocess(self, values: List[str]) -> torch.Tensor:
        processed = [torch.tensor(int(v) * 2) for v in values]
        return join_tensors(processed)


class AverageMetric(metric.Metric):
    @property
    def name(self) -> str:
        return "average"

    @property
    def reduce_fx(self) -> str:
        return "mean"

    def __call__(self, y: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        return ((y + y_pred) / 2).mean()


dummy_model = Model(
    name="dummy_model",
    feature_converter=feature_converter.EncDecFeatureConverter(),
    module=DummyModule(),
    tokenizer=DummyTokenizer(),
)


def to_sample(sample: int) -> TextSampleForTrain:
    return {"x": str(sample), "y": str(sample)}


dummy_task = Task(
    name="dummy",
    source={
        "train": datasource.IterableDataSource(range(100)),
        "validation": datasource.IterableDataSource(range(100)),
        "test": datasource.IterableDataSource(range(100)),
        "prediction": datasource.IterableDataSource(range(100)),
    },
    pipes=[
        datapipe.Shuffler(),
        datapipe.DistributedShardingFilter(),
        datapipe.WorkerShardingFilter(),
        datapipe.Mapper(fn=to_sample),
        datapipe.FeatureTokenizer(),
    ],
    pad_to=10,
    postprocessors=[postprocessor.DecoderPostProcessor()],
    metric_preprocessors=[MultiplyPostProcessor()],
    metrics=[AverageMetric()],
)
