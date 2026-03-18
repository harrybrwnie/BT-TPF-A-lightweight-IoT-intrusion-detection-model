# BT-TPF: A Lightweight IoT Intrusion Detection Model

Implementation of **"A lightweight IoT intrusion detection model based on improved BERT-of-Theseus"** by Wang et al. (2024), published in Expert Systems With Applications.

## Paper Reference

> Wang, Z., Li, J., Yang, S., Luo, X., Li, D., & Mahmoodi, S. (2024). A lightweight IoT intrusion detection model based on improved BERT-of-Theseus. *Expert Systems With Applications, 238*, 122045.

## Overview

BT-TPF is a knowledge-distillation-based IoT intrusion detection framework that achieves high accuracy with minimal parameters, making it suitable for resource-constrained IoT devices.

### Key Features

- **Siamese Network** for feature dimensionality reduction
- **Vision Transformer (ViT)** based Predecessor (teacher) model
- **PoolFormer** based Successor (student) model
- **Improved BERT-of-Theseus** knowledge distillation with gradient optimization
- **~90% parameter reduction** while maintaining >99% accuracy

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        BT-TPF Framework                         │
├─────────────────────────────────────────────────────────────────┤
│  1. Input Data (78/43 features)                                 │
│           ↓                                                     │
│  2. Siamese Network (Dimensionality Reduction → 36 features)    │
│           ↓                                                     │
│  3. Reshape to 6×6×1 Feature Map                                │
│           ↓                                                     │
│  ┌─────────────────────┐   ┌─────────────────────┐              │
│  │ Predecessor (ViT)   │   │  Successor (Pool)   │              │
│  │ 9 layers, ~13,440   │ → │  3 layers, ~788     │              │
│  │ parameters          │   │  parameters         │              │
│  └─────────────────────┘   └─────────────────────┘              │
│           ↓                                                     │
│  4. BERT-of-Theseus Knowledge Distillation                      │
│           ↓                                                     │
│  5. Fine-tuned Successor Model (Final Classifier)               │
└─────────────────────────────────────────────────────────────────┘
```

## Model Components

### 1. Siamese Network (Section 3.1)
- 3-layer MLP with shared parameters
- **Contrastive Loss** (Equation 2):
  ```
  L = (1/2N) * Σ[y*d² + (1-y)*max(margin-d, 0)²]
  ```
- Reduces features to 36 dimensions (6×6×1)

### 2. Predecessor (Teacher) Model (Section 3.4)
- Vision Transformer architecture
- 9 layers (3 modules × 3 blocks)
- Multi-Head Attention with 2 heads
- ~13,440 parameters

### 3. Successor (Student) Model (Section 3.4)
- PoolFormer architecture
- 3 layers (3 modules × 1 block)
- Pooling layer replaces attention (no trainable params)
- **~788-918 parameters** (90% reduction)

### 4. BERT-of-Theseus Knowledge Distillation (Sections 3.2-3.3)
- Module replacement mechanism:
  ```
  r_{i+1} ~ Bernoulli(p)
  y_{i+1} = r_{i+1} * prd_i(y_i) + (1 - r_{i+1}) * scc_i(y_i)
  ```
- Gradient optimization for faster convergence (Equations 8-13)
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- tqdm

## Usage

### Quick Demo (Synthetic Data)

```bash
python main.py --demo
```

### CIC-IDS2017 Dataset

```bash
python main.py --dataset cicids2017 --data_path /path/to/Wednesday-workingHours.csv
```

### TON_IoT Dataset

```bash
python main.py --dataset toniot --data_path /path/to/Train_Test_Network.csv
```

### Full Options

```bash
python main.py \
    --dataset cicids2017 \
    --data_path /path/to/data.csv \
    --epochs 50 \
    --batch_size 1024 \
    --lr 0.0001 \
    --device cuda \
    --save_path models/bt_tpf_model.pth
```

## Hyperparameters (from Section 5.2)

| Parameter | Value | Description |
|-----------|-------|-------------|
| Batch Size | 1024 | Training batch size |
| Learning Rate | 0.0001 | Adam optimizer LR |
| Pre-training Epochs | 50 | Predecessor/Successor pre-training |
| Replacement Epochs | 250 | BERT-of-Theseus training |
| MHA Heads | 2 | Multi-Head Attention heads |
| MLP Ratio (Predecessor) | 4 | Hidden layer multiplier |
| MLP Ratio (Successor) | 1 | Minimal for lightweight |
| Siamese Hidden | 5 | Hidden layer neurons |
| Contrastive Margin | 1.0 | Margin for Contrastive Loss |


## Project Structure

```
BT-TPF-A-lightweight-IoT-intrusion-detection-model/
├── main.py                     # Main execution script
├── requirements.txt            # Dependencies
├── README.md                   # This file
└── src/
    ├── __init__.py
    ├── config.py               # Configuration and hyperparameters
    ├── trainer.py              # Training pipeline
    ├── models/
    │   ├── __init__.py
    │   ├── siamese_network.py  # Siamese Network + Contrastive Loss
    │   ├── predecessor.py      # ViT-based teacher model
    │   ├── successor.py        # PoolFormer-based student model
    │   └── bert_of_theseus.py  # Knowledge distillation framework
    ├── data/
    │   ├── __init__.py
    │   ├── preprocessing.py    # Data preprocessing
    │   └── dataset_loader.py   # Dataset loaders
    └── utils/
        ├── __init__.py
        ├── metrics.py          # Evaluation metrics
        └── visualization.py    # t-SNE visualization
```

## Equations Implemented

| Equation | Description | File |
|----------|-------------|------|
| Eq. 1 | Euclidean Distance | `siamese_network.py` |
| Eq. 2 | Contrastive Loss | `siamese_network.py` |
| Eq. 3-4 | Predecessor/Successor notation | `bert_of_theseus.py` |
| Eq. 5-6 | Module replacement | `bert_of_theseus.py` |
| Eq. 7 | MSE Loss | `bert_of_theseus.py` |
| Eq. 8-13 | Gradient optimization | `bert_of_theseus.py` |
| Eq. 14 | Patch embedding | `predecessor.py` |
| Eq. 15-17 | Positional encoding | `predecessor.py` |
| Eq. 18-19 | Transformer encoder | `predecessor.py` |
| Eq. 20 | Z-Score normalization | `preprocessing.py` |
| Eq. 21-24 | Evaluation metrics | `metrics.py` |

## Citation

If you use this implementation, please cite:

```bibtex
@article{wang2024lightweight,
  title={A lightweight IoT intrusion detection model based on improved BERT-of-Theseus},
  author={Wang, Zhendong and Li, Jingfei and Yang, Shuxin and Luo, Xiao and Li, Dahai and Mahmoodi, Soroosh},
  journal={Expert Systems with Applications},
  volume={238},
  pages={122045},
  year={2024},
  publisher={Elsevier}
}
```

## License

This implementation is for educational and research purposes.
