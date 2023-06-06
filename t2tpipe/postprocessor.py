import dataclasses
from abc import ABC, abstractmethod
from typing import Any, Dict, List, TypeVar, Union, cast

import torch
from torch import Tensor

from t2tpipe.dataclass import ModelPredictionOutput, ModelTrainOutput
from t2tpipe.mixin import SetupMixin
from t2tpipe.util import join_tensors

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


class MappingPostProcessor(PostProcessor):
    def __call__(self, outputs: _MODEL_OUTPUT_T) -> _MODEL_OUTPUT_T:
        if self._env.prediction:
            assert isinstance(outputs, ModelPredictionOutput)
            return self._process_prediction_output(outputs)
        assert isinstance(outputs, ModelTrainOutput)
        return self._process_train_output(outputs)

    def _process_train_output(self, outputs: ModelTrainOutput):
        return dataclasses.replace(
            outputs,
            y=self._map_over_key(outputs, "y"),
            y_pred=self._map_over_key(outputs, "y_pred"),
        )

    def _process_prediction_output(self, outputs: ModelPredictionOutput):
        return dataclasses.replace(
            outputs,
            x=self._map_over_key(outputs, "x"),
            y_pred=self._map_over_key(outputs, "y_pred"),
        )

    def _map_over_key(self, outputs: ModelOutput, key: str) -> List[Any]:
        values = getattr(outputs, key)
        mapped = []
        for value in values:
            mapped.append(self._map(value, key))
        return mapped

    @abstractmethod
    def _map(self, outputs: ModelOutput, key: str) -> Any:
        pass


class ClassificationPostProcessor(MappingPostProcessor):
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

    def _map(self, label: str, key: str) -> int:
        try:
            mapped = self._label2id[key][label]
            return mapped
        except KeyError:
            return self._unk_id


class LMOutputSlicer(MappingPostProcessor):
    _output_prefix: str

    def __init__(self, output_prefix: str):
        self._output_prefix = output_prefix

    def _map(self, src: str, key: str):
        if key not in ["y_pred", "y"]:
            return src
        prefix_idx = src.find(self._output_prefix)
        if prefix_idx == -1:
            return src
        sliced = src[prefix_idx + len(self._output_prefix) :]
        return sliced


class PrefixRemover(MappingPostProcessor):
    x_prefix: str
    y_prefix: str

    def __init__(self, x_prefix: str, y_prefix: str):
        self.x_prefix = x_prefix
        self.y_prefix = y_prefix

    def _map(self, src: str, prefix: str):
        return src[len(prefix) :] if src.startswith(prefix) else src
