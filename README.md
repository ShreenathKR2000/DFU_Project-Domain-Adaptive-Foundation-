# Domain-Adaptive Foundation Models for Fine-Grained DFU Classification

Diabetic Foot Ulcer (DFU) classification using a frozen **DINOv2 ViT-B/14**
backbone with **LoRA adapters** and **Supervised Contrastive Learning**.

## Problem

The DFUC2021 dataset exhibits two key challenges:

1. **Extreme class imbalance** — the four classes (*none*, *infection*,
   *ischaemia*, *both*) are heavily skewed.
2. **Partial labeling** — only 5,955 of 9,949 images carry ground-truth
   labels; the remaining 3,994 are unlabeled.

## Approach

| Component | Detail |
|---|---|
| Backbone | `facebook/dinov2-vitb14` (frozen) |
| Adaptation | LoRA rank-16 on Q/V attention projections (`peft`) |
| Loss | 0.5 * CrossEntropy + 0.5 * SupConLoss (`pytorch-metric-learning`) |
| Head | LayerNorm + Linear (768 -> 4) |

Only the LoRA adapters and the classification head are trained, keeping GPU
memory usage low enough for a **GTX 1650 (4 GB VRAM)**.

## Repository Structure

```
DFU_Project/
├── Data/                    # DFUC2021_train/ (not tracked by git)
│   └── DFUC2021_train/
│       ├── images/
│       └── train.csv
├── Notebooks/
│   └── 01_data_explore.ipynb
├── src/
│   ├── __init__.py
│   ├── dataset.py           # DFUDataset + transforms
│   ├── model.py             # DFUDinoLoRA (DINOv2 + LoRA + head)
│   └── train.py             # Training loop with --debug flag
├── .gitignore
├── CHANGELOG.md
└── README.md
```

## Setup

### 1. Create a Conda environment

```bash
conda create -n dfu python=3.10 -y
conda activate dfu
```

### 2. Install PyTorch (CUDA 11.8 — adjust for your driver)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 3. Install project dependencies

```bash
pip install transformers peft pytorch-metric-learning pandas \
            matplotlib Pillow scikit-learn
```

### 4. Place the dataset

Download or symlink the DFUC2021 training data so that the project tree looks
like:

```
DFU_Project/Data/DFUC2021_train/images/*.jpg
DFU_Project/Data/DFUC2021_train/train.csv
```

## Usage

### Local sanity check (debug mode)

Runs 2 epochs on 20 images with `batch_size=2` — safe for 4 GB VRAM:

```bash
cd DFU_Project
python -m src.train --debug
```

### Full training run

```bash
python -m src.train
```

Override defaults:

```bash
python -m src.train --epochs 30 --batch_size 4 --lr 5e-5
```

## License

This project is for academic and research purposes.
