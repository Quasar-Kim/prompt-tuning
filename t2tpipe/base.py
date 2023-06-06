import dataclasses
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, TypeVar, Union, cast, overload

import torch
from lightning.pytorch import LightningModule
from torch import Tensor

from t2tpipe.dataclass import Env, ModelPredictionOutput, ModelTrainOutput
from t2tpipe.postprocessor import PostProcessor
from t2tpipe.util import dataclass_to_cpu, join_tensor_dataclass

_MODEL_OUTPUT_T = TypeVar("_MODEL_OUTPUT_T", ModelTrainOutput, ModelPredictionOutput)


class BaseLightningModule(LightningModule, ABC):
    _env: Env
    _configured: bool = False
    _train_losses: List[Tensor] = []
    _validation_outputs: List[ModelTrainOutput] = []
    _test_outputs: List[ModelTrainOutput] = []
    _prediction_outputs: List[ModelPredictionOutput] = []
    predictions: Any = None

    def configure(self, env: Env):
        self._env = env
        self._configured = True

    def training_step(self, batch, batch_idx) -> dict:
        output = self._step_train(batch)
        # training_step must output dict / tensor / None
        return output.__dict__

    def validation_step(self, batch, batch_idx) -> ModelTrainOutput:
        output = self._step_train(batch)
        return output

    def test_step(self, batch, batch_idx) -> ModelTrainOutput:
        output = self._step_train(batch)
        return output

    def predict_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> ModelPredictionOutput:
        outputs = self._step_prediction(batch)
        return outputs

    @abstractmethod
    def _step_train(self, batch) -> ModelTrainOutput:
        pass

    @abstractmethod
    def _step_prediction(self, batch) -> ModelPredictionOutput:
        pass

    @abstractmethod
    def forward(self, batch):
        pass

    def on_train_batch_end(self, outputs: dict, batch, batch_idx):
        self._train_losses.append(outputs["loss"])

    def on_train_epoch_end(self):
        self._log_loss(torch.stack(self._train_losses), stage="train")
        self._train_losses.clear()

    def on_validation_batch_end(self, outputs: ModelTrainOutput, batch, batch_idx):
        self._validation_outputs.append(outputs)

    def on_test_batch_end(self, outputs: ModelTrainOutput, batch, batch_idx):
        self._test_outputs.append(outputs)

    def on_validation_epoch_end(self):
        outputs = self._postprocess_outputs(self._validation_outputs)
        self._validation_outputs.clear()
        outputs = self._preprocess_outputs_for_metric(outputs)
        self._log_loss(outputs.loss, stage="validation")
        self._compute_and_log_metrics(
            ys=outputs.y, y_preds=outputs.y_pred, stage="validation"
        )

    def on_test_epoch_end(self):
        outputs = self._postprocess_outputs(self._test_outputs)
        self._test_outputs.clear()
        outputs = self._preprocess_outputs_for_metric(outputs)
        self._log_loss(outputs.loss, stage="test")
        self._compute_and_log_metrics(
            ys=outputs.y, y_preds=outputs.y_pred, stage="test"
        )

    def on_predict_batch_end(self, outputs: ModelPredictionOutput, batch, batch_idx):
        self._prediction_outputs.append(outputs)

    def on_predict_epoch_end(self):
        self.predictions = self._postprocess_outputs(self._prediction_outputs)
        self._prediction_outputs.clear()

    def _postprocess_outputs(self, outputs: List[_MODEL_OUTPUT_T]) -> _MODEL_OUTPUT_T:
        concatenated_outputs = join_tensor_dataclass(outputs)  # (S, N)
        cpu_outputs = dataclass_to_cpu(concatenated_outputs)
        if self._env.task.postprocessors is None:
            return cpu_outputs
        postprocessed_outputs = self._apply_processors_to_outputs(
            cpu_outputs,
            processors=cast(List[PostProcessor], self._env.task.postprocessors),
        )
        return postprocessed_outputs

    def _preprocess_outputs_for_metric(
        self, outputs: _MODEL_OUTPUT_T
    ) -> _MODEL_OUTPUT_T:
        processors = self._env.task.metric_preprocessors
        assert processors is not None
        processed = self._apply_processors_to_outputs(
            outputs, processors=cast(List[PostProcessor], processors)
        )
        return processed

    def _apply_processors_to_outputs(
        self, outputs: _MODEL_OUTPUT_T, processors: List[PostProcessor]
    ) -> _MODEL_OUTPUT_T:
        value = outputs
        for processor in processors:
            value = processor(value)
        return value

    def _log_loss(self, losses: torch.Tensor, stage: str):
        loss = losses.mean()
        self.log(f"{stage}/loss", loss, sync_dist=True, prog_bar=True)

    def _compute_and_log_metrics(self, ys: Any, y_preds: Any, stage: str):
        # ensure ys and y_preds are tensor
        ys, y_preds = torch.tensor(ys), torch.tensor(y_preds)
        metrics = self._env.task.metrics
        assert metrics is not None
        for metric in metrics:
            metric_val = metric(ys, y_preds)
            self.log(
                f"{stage}/{metric.name}",
                metric_val,
                sync_dist=True,
                reduce_fx=metric.reduce_fx,
            )
