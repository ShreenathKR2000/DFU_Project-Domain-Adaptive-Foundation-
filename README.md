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

## Initial Results (Debug Run)

First end-to-end sanity check on the debug subset (20 labeled images, 2 epochs,
batch size 2, GTX 1650 4 GB VRAM):

```
Device : cuda
Debug  : True
Epochs : 2
Batch  : 2

trainable params: 589,824 || all params: 87,170,304 || trainable%: 0.6766

Epoch 1/2  loss=0.8843  ce=1.7687  con=0.0000  acc=0.000  (10.1 s)
Epoch 2/2  loss=0.5786  ce=1.1572  con=0.0000  acc=0.600  (2.4 s)

Model saved to checkpoints/dfu_dino_lora.pt
```

**Key observations**

- Loss drops from 0.8843 → 0.5786 (−34 %) in a single additional epoch,
  confirming the LoRA adapters and head are learning.
- Accuracy rises from 0 % → 60 % on the tiny debug split.
- SupCon loss (`con`) registers 0.0000 at this batch size / subset size — this
  is expected; richer batches (full run) are needed to form meaningful
  contrastive pairs.
- Only **0.68 % of parameters are trainable**, keeping VRAM well within the
  4 GB budget.

## License

This project is for academic and research purposes.
