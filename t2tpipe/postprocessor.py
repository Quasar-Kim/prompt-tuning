from abc import ABC, abstractmethod
from typing import Any, Dict, Union, List, TypeVar, cast
import dataclasses

import torch
from torch import Tensor

from t2tpipe.mixin import SetupMixin
from t2tpipe.util import join_tensors
from t2tpipe.dataclass import ModelTrainOutput, ModelPredictionOutput

ModelOutput = Union[ModelTrainOutput, ModelPredictionOutput]

_MODEL_OUTPUT_T = TypeVar("_MODEL_OUTPUT_T", ModelTrainOutput, ModelPredictionOutput)


class PostProcessor(ABC, SetupMixin):
    @abstractmethod
    def __call__(self, outputs: _MODEL_OUTPUT_T) -> _MODEL_OUTPUT_T:
        pass


class DecoderPostProcessor(PostProcessor):
    def __call__(self, outputs: _MODEL_OUTPUT_T) -> _MODEL_OUTPUT_T:
        if self._env.prediction:
            assert isinstance(outputs, ModelPredictionOutput)
            return self._process_prediction_output(outputs)
        assert isinstance(outputs, ModelTrainOutput)
        return self._process_train_output(outputs)

    def _process_train_output(self, outputs: ModelTrainOutput):
        return dataclasses.replace(
            outputs,
            x=self._decode(outputs.x),
            y=self._decode(outputs.y),
            y_pred=self._decode(outputs.y_pred),
        )

    def _process_prediction_output(self, outputs: ModelPredictionOutput):
        return dataclasses.replace(
            outputs, x=self._decode(outputs.x), y_pred=self._decode(outputs.y_pred)
        )

    def _decode(self, values: Tensor) -> List[str]:
        decoded = self._env.model.tokenizer.decode_batch(
            values.tolist(), remove_special_tokens=True
        )
        return decoded


class ClassificationPostProcessor(PostProcessor):
    _label2id: Dict[str, Dict[str, int]]
    _unk_id: int

    def __init__(
        self, label2id: Union[Dict[str, Dict[str, int]], Dict[str, int]], unk_id=-1
    ):
        if any(isinstance(v, int) for v in label2id.values()):
            label2id = cast(Dict[str, int], label2id)
            default_keys = ["x", "y", "y_pred"]
            self._label2id = {k: label2id for k in default_keys}
        else:
            label2id = cast(Dict[str, Dict[str, int]], label2id)
            self._label2id = label2id
        self._unk_id = unk_id

    def __call__(self, outputs: _MODEL_OUTPUT_T) -> _MODEL_OUTPUT_T:
        if self._env.prediction:
            assert isinstance(outputs, ModelPredictionOutput)
            return self._process_prediction_output(outputs)
        assert isinstance(outputs, ModelTrainOutput)
        return self._process_train_output(outputs)

    def _process_train_output(self, outputs: ModelTrainOutput):
        return dataclasses.replace(
            outputs,
            y=self._map(outputs, "y"),
            y_pred=self._map(outputs, "y_pred"),
        )

    def _process_prediction_output(self, outputs: ModelPredictionOutput):
        return dataclasses.replace(
            outputs,
            x=self._map(outputs, "x"),
            y_pred=self._map(outputs, "y_pred"),
        )

    def _map(self, outputs: ModelOutput, key: str) -> List[int]:
        labels = getattr(outputs, key)
        mapped = []
        for label in labels:
            try:
                mapped_label = self._label2id[key][label]
            except KeyError:
                mapped_label = self._unk_id
            mapped.append(mapped_label)
        return mapped


class LMOutputSlicer(PostProcessor):
    def __call__(self, outputs: _MODEL_OUTPUT_T) -> _MODEL_OUTPUT_T:
        return dataclasses.replace(
            outputs,
            y_pred=self._slice(outputs.x, outputs.y_pred),
        )

    def _slice(self, x: Tensor, y_pred: Tensor) -> Tensor:
        return y_pred[:, x.size(1) :]
