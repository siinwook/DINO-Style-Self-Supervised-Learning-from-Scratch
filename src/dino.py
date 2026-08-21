import torch
import torch.nn.functional as F
from torchvision.transforms import v2


def augmentation(x, size = 84):
  if size==84:
    x = v2.RandomResizedCrop(size, scale=(0.35,1.0))(x)
  else:
    x = v2.RandomResizedCrop(size, scale=(0.05,0.35))(x)
  x = v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)(x)
  x = v2.RandomHorizontalFlip(0.5)(x)

  return x


def forward_models(models, x):
  for model in models:
    x = model(x)
  return x


def update_teacher(teacher, student, m=0.996):
  for param_t, param_s in zip(teacher.parameters(), student.parameters()):
    param_t.data = param_t.data * m + param_s.data * (1.0 - m)


def dino_loss(s,t,C,S_temp,T_temp):
  s_log_prob = F.log_softmax(s/S_temp, dim=-1)
  t_prob = F.softmax((t-C)/T_temp, dim=-1)

  loss = - (t_prob * s_log_prob).sum(dim=-1).mean()
  return loss
