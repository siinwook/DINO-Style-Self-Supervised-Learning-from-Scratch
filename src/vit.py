import torch
from torch import nn
import torch.nn.functional as F
import math

from .modules.patch_embedding import PatchEmbedding
from .modules.encoder import Encoder


class VisionTransformer(nn.Module):
  def __init__(self, pt_img_size, img_channel, patch_size, d_model, num_heads, d_ff, drop_out, L):
    super().__init__()

    self.pt_img_size = pt_img_size
    self.patch_size = patch_size
    self.d_model = d_model

    self.PatchEmbedding = PatchEmbedding(img_channel, patch_size, d_model)

    self.x_cls = nn.Parameter(torch.randn(1,1,d_model) * math.sqrt(2.0 / d_model)) # He init

    pt_N = pt_img_size**2 // patch_size**2
    self.pt_PositionalEncoding = nn.Parameter(torch.randn(1,pt_N+1, d_model) * math.sqrt(2.0 / d_model)) # He init

    self.Encoder = Encoder(d_model, num_heads, d_ff, drop_out, L)

  def pos_enc_interpolate(self, x, H, W):
    if H==W==self.pt_img_size:
      return self.pt_PositionalEncoding

    original_h = self.pt_img_size // self.patch_size
    original_w = self.pt_img_size // self.patch_size

    target_h = H // self.patch_size
    target_w = W // self.patch_size

    pos_cls = self.pt_PositionalEncoding[:,:1,:]
    pos_patch = self.pt_PositionalEncoding[:,1:,:]

    pos_patch = pos_patch.reshape(1,original_h,original_w,self.d_model).permute(0,3,1,2)
    pos_patch = F.interpolate(pos_patch, size=(target_h, target_w), mode="bicubic", align_corners=False)
    pos_patch = pos_patch.reshape(1,self.d_model,target_h*target_w).permute(0,2,1)

    return torch.cat([pos_cls,pos_patch], dim=1)

  def forward(self, x):
    B,C,H,W = x.shape

    x = self.PatchEmbedding(x) # (B,N,d_model)
    x = torch.cat((self.x_cls.expand(B,1,-1),x), dim=1) # (B,N+1,d_model)
    x = self.pos_enc_interpolate(x,H,W) + x

    x = self.Encoder(x)

    x = x[:,0,:]

    return x #(B,d_model)
