import torchmetrics
import torch

acc = torchmetrics.classification.MulticlassAccuracy(num_classes=3)
preds = torch.tensor([[100, 100]])
y = torch.tensor([[0, 0]])
acc(preds, y)