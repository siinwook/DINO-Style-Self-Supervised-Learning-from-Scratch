import torch
from torch import nn

from .encoder_block import EncoderBlock


class Encoder(nn.Module):
  def __init__(self, d_model, num_heads, d_ff, drop_out=0.1, L=12):
      super().__init__()

      self.L = L
      self.EncBlcockList = nn.ModuleList([
          EncoderBlock(d_model, num_heads, d_ff, drop_out)
          for i in range(L)
      ])

  def forward(self,x):
    for EncBlock in self.EncBlcockList:
      x = EncBlock(x)

    return x
