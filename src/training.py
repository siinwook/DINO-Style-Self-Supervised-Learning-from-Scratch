import copy
import torch
from torch import nn

from .vit import VisionTransformer
from .modules.heads import ProjHead, LinearClassifier
from .dino import augmentation, forward_models, update_teacher, dino_loss


def pretrain_dino(unlabeled_dataloader, device):
  K = 150
  C = torch.zeros(K).to(device)
  S_temp = 0.1
  T_temp = 0.04
  T_momentum = 0.996
  C_momentum = 0.9
  epochs = 30
  warmup_epochs = 3

  student = VisionTransformer(pt_img_size=84, img_channel=3, patch_size=6, d_model=192, num_heads=6, d_ff=768, drop_out=0.1, L=8).to(device)
  teacher = copy.deepcopy(student).to(device)
  teacher.requires_grad_(False)

  student_projhead = ProjHead(d_model=192, d_ff=768, num_classes=K).to(device)
  teacher_projhead = copy.deepcopy(student_projhead).to(device)
  teacher_projhead.requires_grad_(False)

  student.train()
  student_projhead.train()
  teacher.eval()
  teacher_projhead.eval()

  loss_fn = dino_loss
  optimizer = torch.optim.AdamW(list(student.parameters())+list(student_projhead.parameters()), lr = 5e-4, weight_decay=1e-3)

  scheduler_linear = torch.optim.lr_scheduler.LinearLR(optimizer=optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
  scheduler_cos = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs-warmup_epochs, eta_min=1e-5)
  scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer=optimizer, schedulers=[scheduler_linear,scheduler_cos], milestones=[warmup_epochs])

  for epoch in range(epochs):
    epoch_loss=0
    for _, (x, label) in enumerate(unlabeled_dataloader):

      xg1, xg2, x1, x2 = augmentation(x, size=84), augmentation(x, size=84), augmentation(x, size=36), augmentation(x, size=36)
      xg1, xg2, x1, x2 = xg1.to(device), xg2.to(device), x1.to(device), x2.to(device)

      sg1, sg2, s1, s2 = forward_models([student, student_projhead],xg1), forward_models([student, student_projhead],xg2), forward_models([student, student_projhead],x1), forward_models([student, student_projhead],x2)
      with torch.no_grad():
        tg1, tg2 = forward_models([teacher, teacher_projhead],xg1), forward_models([teacher, teacher_projhead],xg2)

      optimizer.zero_grad()
      loss = [loss_fn(sg2,tg1,C,S_temp,T_temp), loss_fn(s1,tg1,C,S_temp,T_temp), loss_fn(s2,tg1,C,S_temp,T_temp), loss_fn(sg1,tg2,C,S_temp,T_temp), loss_fn(s1,tg2,C,S_temp,T_temp), loss_fn(s2,tg2,C,S_temp,T_temp)]
      loss = sum(loss) / len(loss)
      loss.backward()
      optimizer.step()

      with torch.no_grad():
        update_teacher(teacher, student, T_momentum)
        update_teacher(teacher_projhead, student_projhead, T_momentum)

        C = C_momentum * C + (1.0 - C_momentum) * torch.mean(torch.cat([tg1,tg2], dim=0), dim=0)
        epoch_loss += loss.item()

    print(f"epoch {epoch+1} loss: {epoch_loss / len(unlabeled_dataloader)}")
    scheduler.step()

  return student, teacher, student_projhead, teacher_projhead


def pretrain_dino_global_only(unlabeled_dataloader, device):
  K = 150
  C = torch.zeros(K).to(device)
  S_temp = 0.1
  T_temp = 0.04
  T_momentum = 0.996
  C_momentum = 0.9
  epochs = 30
  warmup_epochs = 3

  student = VisionTransformer(pt_img_size=84, img_channel=3, patch_size=6, d_model=192, num_heads=6, d_ff=768, drop_out=0.1, L=8).to(device)
  teacher = copy.deepcopy(student).to(device)
  teacher.requires_grad_(False)

  student_projhead = ProjHead(d_model=192, d_ff=768, num_classes=K).to(device)
  teacher_projhead = copy.deepcopy(student_projhead).to(device)
  teacher_projhead.requires_grad_(False)

  student.train()
  student_projhead.train()
  teacher.eval()
  teacher_projhead.eval()

  loss_fn = dino_loss
  optimizer = torch.optim.AdamW(list(student.parameters())+list(student_projhead.parameters()), lr = 5e-4, weight_decay=1e-3)

  scheduler_linear = torch.optim.lr_scheduler.LinearLR(optimizer=optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
  scheduler_cos = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs-warmup_epochs, eta_min=1e-5)
  scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer=optimizer, schedulers=[scheduler_linear,scheduler_cos], milestones=[warmup_epochs])

  for epoch in range(epochs):
    epoch_loss=0
    for _, (x, label) in enumerate(unlabeled_dataloader):

      xg1, xg2 = augmentation(x, size=84), augmentation(x, size=84)
      xg1, xg2 = xg1.to(device), xg2.to(device)

      sg1, sg2 = forward_models([student, student_projhead],xg1), forward_models([student, student_projhead],xg2)
      with torch.no_grad():
        tg1, tg2 = forward_models([teacher, teacher_projhead],xg1), forward_models([teacher, teacher_projhead],xg2)

      optimizer.zero_grad()
      loss = [loss_fn(sg2,tg1,C,S_temp,T_temp), loss_fn(sg1,tg2,C,S_temp,T_temp)]
      loss = sum(loss) / len(loss)
      loss.backward()
      optimizer.step()

      with torch.no_grad():
        update_teacher(teacher, student, T_momentum)
        update_teacher(teacher_projhead, student_projhead, T_momentum)

        C = C_momentum * C + (1.0 - C_momentum) * torch.mean(torch.cat([tg1,tg2], dim=0), dim=0)
        epoch_loss += loss.item()

    print(f"epoch {epoch+1} loss: {epoch_loss / len(unlabeled_dataloader)}")
    scheduler.step()

  return student, teacher, student_projhead, teacher_projhead


def train_linear_probe(teacher, train_dataloader, device, num_classes):
  teacher_linear_classifier = LinearClassifier(d_model=192, num_classes=num_classes).to(device)

  teacher.requires_grad_(False)
  teacher.eval()
  teacher_linear_classifier.train()

  epochs = 40
  warmup_epochs = 4

  loss_fn = nn.CrossEntropyLoss()
  optimizer = torch.optim.AdamW(teacher_linear_classifier.parameters(), lr=5e-4, weight_decay=0.0)

  scheduler_linear = torch.optim.lr_scheduler.LinearLR(optimizer=optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
  scheduler_cos = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs-warmup_epochs, eta_min=1e-5)
  scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer=optimizer, schedulers=[scheduler_linear,scheduler_cos], milestones=[warmup_epochs])

  for epoch in range(epochs):
    epoch_loss=0
    for _, (x, label) in enumerate(train_dataloader):
      x, label = x.to(device), label.to(device)

      logits = forward_models([teacher,teacher_linear_classifier],x)

      optimizer.zero_grad()
      loss = loss_fn(logits, label)
      loss.backward()
      optimizer.step()

      epoch_loss += loss.item()

    if (epoch+1)%10==0:
      print(f"epoch {epoch+1} loss: {epoch_loss / len(train_dataloader)}")
    scheduler.step()

  return teacher_linear_classifier


def fine_tune(teacher, train_dataloader, device, num_classes):
  teacher_linear_classifier = LinearClassifier(d_model=192, num_classes=num_classes).to(device)

  teacher.train()
  teacher_linear_classifier.train()

  epochs = 40
  warmup_epochs = 4

  loss_fn = nn.CrossEntropyLoss()
  optimizer = torch.optim.AdamW(list(teacher.parameters())+list(teacher_linear_classifier.parameters()), lr=5e-4, weight_decay=1e-3)

  scheduler_linear = torch.optim.lr_scheduler.LinearLR(optimizer=optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
  scheduler_cos = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs-warmup_epochs, eta_min=1e-5)
  scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer=optimizer, schedulers=[scheduler_linear,scheduler_cos], milestones=[warmup_epochs])

  for epoch in range(epochs):
    epoch_loss=0
    for _, (x, label) in enumerate(train_dataloader):
      x, label = x.to(device), label.to(device)

      logits = forward_models([teacher,teacher_linear_classifier],x)

      optimizer.zero_grad()
      loss = loss_fn(logits, label)
      loss.backward()
      optimizer.step()

      epoch_loss += loss.item()

    if (epoch+1)%10==0:
      print(f"epoch {epoch+1} loss: {epoch_loss / len(train_dataloader)}")
    scheduler.step()

  return teacher, teacher_linear_classifier


def train_random_vit(train_dataloader, device, num_classes):
  rand_vit = VisionTransformer(pt_img_size=84, img_channel=3, patch_size=6, d_model=192, num_heads=6, d_ff=768, drop_out=0.1, L=8).to(device)
  rand_head = ProjHead(d_model=192, d_ff=784, num_classes=num_classes).to(device)

  rand_vit.train()
  rand_head.train()

  epochs = 40
  warmup_epochs = 4

  loss_fn = nn.CrossEntropyLoss()
  optimizer = torch.optim.AdamW(list(rand_vit.parameters())+list(rand_head.parameters()), lr=5e-4, weight_decay=1e-3)

  scheduler_linear = torch.optim.lr_scheduler.LinearLR(optimizer=optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
  scheduler_cos = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs-warmup_epochs, eta_min=1e-5)
  scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer=optimizer, schedulers=[scheduler_linear,scheduler_cos], milestones=[warmup_epochs])

  for epoch in range(epochs):
    epoch_loss=0
    for _, (x, label) in enumerate(train_dataloader):
      x, label = x.to(device), label.to(device)

      logits = forward_models([rand_vit, rand_head],x)

      optimizer.zero_grad()
      loss = loss_fn(logits, label)
      loss.backward()
      optimizer.step()

      epoch_loss += loss.item()

    if (epoch+1)%10==0:
      print(f"epoch {epoch+1} loss: {epoch_loss / len(train_dataloader)}")
    scheduler.step()

  return rand_vit, rand_head
