import torch

class Accuracy:
    name = 'accuracy'
    reduce_fx = 'mean'

    def __call__(self, y: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        assert y.shape == y_pred.shape
        correct = torch.sum(y == y_pred)
        total = y.numel()
        return correct.float() / total