# DINO-Style-Self-Supervised-Learning-from-Scratch

This project implements the DINO-style self-supervised learning pipeline with Vision Transformer from *"Emerging Properties in Self-Supervised Vision Transformers"*, and evaluates visual representations on several datasets.

It also visualizes the evolving attention map of CLS token in last encoder layer while pretraining and conducts ablation study of multi-crop augmentation

## What I implemented

### DINO

* Teacher / Student Distillation Pipeline
* Centering and Sharpening
* Multi-Crop Augmentation
* Teacher Momentum Encoder
* Exponential Moving Average Update of Centering Momentum

### Experiments

* DINO self-supervised ViT Feature Evaluation and comparison transfer learning performance with random initialized ViT(small dataset with STL-10 and bigger dataset with Mini ImageNet)
* Attention map evolving visualization
* Multi-crop ablation

## Key Results

### Experiment 1: DINO Representation Evaluation

| Dataset | DINO | Random Init |
| ----- | ----- | ----- |
| Mini ImageNet | 43.5% | 36.9% |

### Experiment 2: Attention Map Evolving Visualization


### Experiment 3: Multi-Crop Ablation

| Multi-Crop | k-NN |
| ----- | ------------: |
| O |        45.2% |
| X |        44.4% |

## Tech Stack

PyTorch, Torchvision, Hugging Face, CUDA, Matplotlib, Jupyter Notebook
