# DINO-Style-Self-Supervised-Learning-from-Scratch

This project implements the DINO-style self-supervised learning pipeline with Vision Transformer from *"Emerging Properties in Self-Supervised Vision Transformers"*, and evaluates visual representations on several datasets.

It also visualizes the evolving attention map of CLS token in last encoder layer while pretraining and conducts ablation study of multi-crop augmentation

## What I implemented

### DINO

* Teacher / Student Distillation Pipeline
* Centering and Sharpening
* Global and Local Multi-Crop Augmentation
* Teacher Momentum Encoder
* Exponential Moving Average Update of Centering Momentum

### Experiments

* Evaluation of DINO-pretrained visual representations
* Transfer-learning comparison between DINO-pretrained and randomly initialized ViTs on STL-10 and Mini-ImageNet
* Attention map evolving visualization
* Multi-crop ablation

## Key Results

### Experiment 1: DINO Representation Evaluation

| Dataset | DINO Pretrained | Random Init |
| ----- | ----- | ----- |
| Mini ImageNet | 43.5% | 36.9% |

### Experiment 2: Attention Map Evolving Visualization

<p align="center">
  <img src="results/attention_maps/1154_1.png" width="22%">
  <img src="results/attention_maps/1154_10.png" width="22%">
  <img src="results/attention_maps/1154_20.png" width="22%">
  <img src="results/attention_maps/1154_30.png" width="22%">
</p>

### Experiment 3: Multi-Crop Ablation

| Multi-Crop | k-NN |
| ----- | ------------: |
| Enabled |        45.2% |
| Disabled |        44.4% |

## Tech Stack

PyTorch, Torchvision, Hugging Face, CUDA, Matplotlib, Jupyter Notebook
