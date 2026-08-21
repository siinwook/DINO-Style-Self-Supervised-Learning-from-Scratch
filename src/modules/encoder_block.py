import torch
from torch import nn

from .attention import Attention
from .mlp import MLP


class EncoderBlock(nn.Module):
  def __init__(self, d_model, num_heads, d_ff, drop_out=0.1):
    super().__init__()

    self.Attention = Attention(d_model, num_heads)
    self.MLP = MLP(d_model, d_ff)

    self.LayerNorm1 = nn.LayerNorm(d_model)
    self.LayerNorm2 = nn.LayerNorm(d_model)

    self.Dropout = nn.Dropout(drop_out)

  def forward(self,x):
    x_LN1 = self.LayerNorm1(x)
    x = self.Attention(x_LN1, x_LN1, x_LN1, mask = None) + x

    x_LN2 = self.LayerNorm2(x)
    x = self.MLP(x_LN2) + x

    return x
