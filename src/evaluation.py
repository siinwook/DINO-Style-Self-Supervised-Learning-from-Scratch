import torch
import torch.nn.functional as F

from .dino import forward_models


def forward_features(model, dataloader, device):
  train_features = []
  train_labels = []

  for _, (x, label) in enumerate(dataloader):
    x, label = x.to(device), label.to(device)
    with torch.no_grad():
      logits = model(x)
    train_features.append(logits)
    train_labels.append(label)

  train_features = torch.cat(train_features, dim=0).to(device)
  train_labels = torch.cat(train_labels, dim=0).to(device)

  return train_features, train_labels


def knn_eval(model, train_dataloader, test_dataloader, num_classes, device):
  model.eval()

  train_features, train_labels = forward_features(model, train_dataloader, device)
  train_features = F.normalize(train_features, dim=-1)

  current=0
  for _, (x, label) in enumerate(test_dataloader):
    x, label = x.to(device), label.to(device)

    target_features = model(x)
    target_features = F.normalize(target_features, dim=-1)

    cos_similarity = target_features @ train_features.T

    knn_similarity, index = cos_similarity.topk(k=20, dim=-1, largest=True)
    knn_label = train_labels[index]

    weights = torch.exp(knn_similarity / 0.07)

    class_score = torch.zeros(1,num_classes,device = device)
    class_score.scatter_add_(1, knn_label.long(), weights)

    prediction = class_score.argmax(dim=-1)
    current += (prediction == label).sum().item()

  return current / len(test_dataloader.dataset)


def eval_classifier(model, classifier, test_dataloader, test_dataset, device):
  model.eval()
  classifier.eval()

  current = 0
  with torch.no_grad():
    for _, (x, label) in enumerate(test_dataloader):
      x, label = x.to(device), label.to(device)

      logits = forward_models([model,classifier],x)

      current += (logits.argmax(dim=-1) == label).sum()

  return current / len(test_dataset)
