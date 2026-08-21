import torch
from torch import nn


class MLP(nn.Module):
  def __init__(self, d_model, d_ff):
    super().__init__()

    self.Sequential = nn.Sequential(
        nn.Linear(d_model, d_ff),
        nn.GELU(),
        nn.Linear(d_ff,d_model)
    )

  def forward(self,x):
    x = self.Sequential(x)
    return x
