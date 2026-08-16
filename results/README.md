# Experimental Results

This document summarizes the main experimental results and my observations.

Training conditions of experiments are indicated below

---

## 1. DINO Representation Evaluation

| Dataset | k-NN Probe | Linear Probe | Fine Tune | Random Init |
| ----- | ----- | ----- | ----- | ----- |
| Mini ImageNet | 19.2% | 24.0% | 43.5% | 36.9% |
| STL-10 | 45.2% | 47.7% | 60.0% | 51.8% |

### Interpretation

- Considering that the random shot accuracy of Mini ImageNet is 1%, DINO pretrained features can classify a significant number of unseen images only with knn probe and linear probe.
- DINO pretraining learns well generalized features that can be easily fine-tuned in transfer learning of unseen images.
- For datasets of the same category, DINO self-supervised features adopted to straightforward probing achieves comparable performance to supervised learning features(45.2%, 47.7% vs 51.8%).

---

## 2. Attention Map Evolving Visualization



### Observation

-

### Interpretation

- 

---

## 3. Multi-Crop Ablation

| Multi-Crop | k-NN Probe |
| ----- | ------------: |
| Enabled |        45.2% |
| Disabled |        44.4% |

### Interpretation

- Multi-crop encourages model to learn 'local to global' correspondences and it leads to retain more generalized visual features.

---

### Training conditions

**Pretraining**

| Conditions | Pretraining |
|---|---|
| Dataset | STL-10 (Unlabeled) |
| Model | DINO-Style ViT |
| Optimizer | AdamW |
| Learning rate scheduler | Linear warmup + Cosine annealing |
| Learning rate | Linear warmup to 0.0005(1-3 epoch), then cosine annealing to 0.00001(4-30 epoch) |
| # of teachers/students | teachers: 2  students: 2 |
| Teacher temperature | 0.04 |
| Student temperature | 0.1 |
| Teacher momentum | 0.996 |
| Centering momentum | 0.9 |
| Output dimensionality K | 150 |
| Batch size | 192 |
| Epochs | 30 |
| Data augmentation | RandomResized crop, Horizontal flip, Color jitter|
| Weight decay | 0.001 |
| Device | cuda |

**Linear Head**

| Conditions | Linear Head |
|---|---|
| Dataset | STL-10 (Unlabeled) / Mini ImageNet |
| Learning rate scheduler | Linear warmup + Cosine annealing |
| Learning rate | Linear warmup to 0.0005(1-4 epoch), then cosine annealing to 0.00001(5-40 epoch) |
| Data augmentation | Resize |
| Weight decay | 0 |

**Fine Tuning**

| Conditions | Fine Tuning |
|---|---|
| Dataset | STL-10 (Unlabeled) / Mini ImageNet |
| Learning rate scheduler | Linear warmup + Cosine annealing |
| Learning rate | Linear warmup to 0.0005(1-4 epoch), then cosine annealing to 0.00001(5-40 epoch) |
| Data augmentation | Resize |
