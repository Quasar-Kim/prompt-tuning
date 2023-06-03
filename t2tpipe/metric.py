from abc import ABC, abstractmethod

import torch
from torch import Tensor

from t2tpipe.mixin import SetupMixin


class Metric(ABC, SetupMixin):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def reduce_fx(self) -> str:
        pass

    @abstractmethod
    def __call__(self, y: Tensor, y_pred: Tensor) -> Tensor:
        pass


class Accuracy(Metric):
    @property
    def name(self):
        return "accuracy"

    @property
    def reduce_fx(self):
        return "mean"

    def __call__(self, y: Tensor, y_pred: Tensor) -> Tensor:
        assert y.shape == y_pred.shape
        correct = torch.sum(y == y_pred)
        total = y.numel()
        return correct.float() / total


class F1Macro(Metric):
    _n_classes: int

    @property
    def name(self):
        return "f1-macro"

    @property
    def reduce_fx(self):
        return "mean"

    def __init__(self, n_classes) -> None:
        self._n_classes = n_classes

    def __call__(self, y: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        f1s = []
        for cls_id in torch.arange(self._n_classes):
            y_trues = y == cls_id
            y_falses = y != cls_id
            pred_trues = y_pred == cls_id
            pred_falses = y_pred != cls_id
            tp = torch.logical_and(y_trues, pred_trues).count_nonzero()
            fn = torch.logical_and(y_trues, pred_falses).count_nonzero()
            fp = torch.logical_and(y_falses, pred_trues).count_nonzero()
            precision = torch.nan_to_num(tp / (tp + fp))
            recall = torch.nan_to_num(tp / (tp + fn))
            f1 = torch.nan_to_num(2 * (precision * recall) / (precision + recall))
            f1s.append(f1)
        return torch.stack(f1s).mean()
