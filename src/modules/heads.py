import torch
from torch import nn


class ProjHead(nn.Module):
  def __init__(self, d_model, d_ff, num_classes):
    super().__init__()

    self.Sequential = nn.Sequential(
        nn.Linear(d_model, d_ff),
        nn.GELU(),
        nn.Linear(d_ff, d_ff),
        nn.GELU(),
        nn.Linear(d_ff, num_classes)
    )

  def forward(self,x):
    x = self.Sequential(x)
    return x


class LinearClassifier(nn.Module):
  def __init__(self, d_model, num_classes):
    super().__init__()

    self.linear = nn.Linear(d_model,num_classes)

  def forward(self,x):
    return self.linear(x)
