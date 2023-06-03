import torch

from t2tpipe.metric import Accuracy


def test_accuracy():
    acc = Accuracy()
    y = torch.tensor([0, 0])
    y_pred = torch.tensor([0, 1])
    torch.testing.assert_close(acc(y, y_pred), torch.tensor(0.5))
