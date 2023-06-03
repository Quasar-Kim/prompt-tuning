from abc import ABC, abstractmethod
from typing import Any, List, Tuple

import torch
from torch import Tensor
from lightning.pytorch import LightningModule

from t2tpipe.dataclass import (
    ModelTrainOutput,
    ModelPredictionOutput,
    Env,
    BatchTextPrediction,
)


# TODO: support inference mode
class BaseLightningModule(LightningModule, ABC):
    _env: Env
    _configured: bool = False
    _train_losses: List[Tensor] = []
    _validation_outputs: List[ModelTrainOutput] = []
    _test_outputs: List[ModelTrainOutput] = []

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
    ) -> BatchTextPrediction:
        outputs = self._step_prediction(batch)
        cpu_outputs = _model_prediction_output_to_cpu(outputs)
        prediction = BatchTextPrediction(
            x=self._decode_batch(cpu_outputs.x),
            pred=self._decode_batch(cpu_outputs.pred),
        )
        return prediction

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
        self._log_metrics(self._validation_outputs, stage="validation")

    def on_test_epoch_end(self):
        self._log_metrics(self._test_outputs, stage="test")

    def _log_metrics(self, outputs: List[ModelTrainOutput], stage: str):
        collated_outputs = self._collate_train_outputs(outputs)
        cpu_outputs = _model_train_output_to_cpu(collated_outputs)
        ys, y_preds = self._postprocess_model_train_output(cpu_outputs)
        outputs.clear()
        self._log_loss(cpu_outputs.loss, stage=stage)
        self._compute_and_log_metrics(ys, y_preds, stage=stage)

    def _collate_prediction_outputs(
        self, outputs: List[ModelPredictionOutput]
    ) -> ModelPredictionOutput:
        ys = []  # list of (B, N)
        y_preds = []  # list of (B, N)
        for out in outputs:
            ys.append(out.x)
            y_preds.append(out.pred)
        collated_outputs = ModelPredictionOutput(
            x=torch.cat(ys),  # (number of samples, N)
            pred=torch.cat(y_preds),  # (number of samples, N)
        )
        return collated_outputs

    def _collate_train_outputs(
        self, outputs: List[ModelTrainOutput]
    ) -> ModelTrainOutput:
        ys = []  # list of (B, N)
        y_preds = []  # list of (B, N)
        losses = []  # list of (B,)
        for out in outputs:
            ys.append(out.y)
            y_preds.append(out.y_pred)
            losses.append(out.loss)
        collated_outputs = ModelTrainOutput(
            y=torch.cat(ys),  # (number of samples, N)
            y_pred=torch.cat(y_preds),  # (number of samples, N)
            loss=torch.stack(losses),  # (number of samples,)
        )
        return collated_outputs

    def _log_loss(self, losses: torch.Tensor, stage):
        loss = losses.mean()
        self.log(f"{stage}/loss", loss, sync_dist=True, prog_bar=True)

    def _postprocess_model_train_output(
        self, outputs: ModelTrainOutput
    ) -> Tuple[Tensor, Tensor]:
        ys = self._postprocess(outputs.y)
        y_preds = self._postprocess(outputs.y_pred)
        return ys, y_preds

    def _postprocess(self, inputs: Tensor) -> Tensor:
        decoded = self._decode_batch(inputs)
        postprocessor = self._env.task.postprocessor
        assert postprocessor is not None
        postprocessed = [postprocessor(sample) for sample in decoded]
        out = torch.tensor(postprocessed)
        return out

    def _decode_batch(self, inputs: Tensor) -> List[str]:
        decoded = self._env.model.tokenizer.decode_batch(
            inputs.tolist(), remove_special_tokens=True
        )
        return decoded

    def _compute_and_log_metrics(self, ys: Tensor, y_preds: Tensor, stage: str):
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


def _model_train_output_to_cpu(outputs: ModelTrainOutput) -> ModelTrainOutput:
    return ModelTrainOutput(
        y=outputs.y.cpu(), y_pred=outputs.y_pred.cpu(), loss=outputs.loss.cpu()
    )


def _model_prediction_output_to_cpu(
    outputs: ModelPredictionOutput,
) -> ModelPredictionOutput:
    return ModelPredictionOutput(
        x=outputs.x.cpu(),
        pred=outputs.pred.cpu(),
    )
