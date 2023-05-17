from typing import Any
import torch

class Accuracy:
    name = 'accuracy'
    reduce_fx = 'mean'

    def __call__(self, y: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        assert y.shape == y_pred.shape
        correct = torch.sum(y == y_pred)
        total = y.numel()
        return correct.float() / total
    
class MacroF1:
    name = 'f1'
    reduce_fx = 'mean'

    def __init__(self, n_classes) -> None:
        self.n_classes = n_classes

    def __call__(self, y: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        f1s = []
        for cls in torch.arange(self.n_classes):
            print(cls)
            y_trues = y == cls
            y_falses = y != cls
            pred_trues = y_pred == cls
            pred_falses = y_pred != cls
            tp = torch.logical_and(y_trues, pred_trues).count_nonzero()
            fn = torch.logical_and(y_trues, pred_falses).count_nonzero()
            fp = torch.logical_and(y_falses, pred_trues).count_nonzero()
            precision = torch.nan_to_num(tp / (tp + fp))
            recall = torch.nan_to_num(tp / (tp + fn))
            f1 = torch.nan_to_num(2 * (precision * recall) / (precision + recall))
            f1s.append(f1)
        return torch.stack(f1s).mean()
