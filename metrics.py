import torch
from torchmetrics import Metric

class NaiveAccuracy(Metric):
    is_differentiable = False
    higher_is_better = True
    full_state_update = True

    def __init__(self):
        super().__init__()
        self.add_state('correct', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape == target.shape
        self.correct += torch.sum(preds == target)
        self.total += target.numel()

    def compute(self):
        return self.correct.float() / self.total
    
