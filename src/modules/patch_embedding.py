import torch
from torch import nn


class PatchEmbedding(nn.Module):
  def __init__(self, img_channel, patch_size, d_model):
    super().__init__()

    self.Conv = nn.Conv2d(img_channel, d_model, patch_size, patch_size, bias=False)

  def forward(self, x):
    x = self.Conv(x)

    x = x.flatten(2).transpose(1,2)

    return x
