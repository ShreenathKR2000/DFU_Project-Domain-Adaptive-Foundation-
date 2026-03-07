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

A **two-phase** training pipeline, both fitting within a **GTX 1650 (4 GB VRAM)**:

### Phase 1 — Domain-Adaptive Pre-training (unsupervised)

| Component | Detail |
|---|---|
| Data | 3,994 **unlabeled** DFU images |
| Task | Masked Image Modeling (SimMIM-style) |
| Masking | 60 % of 14×14 patches zeroed at the pixel level |
| Decoder | LayerNorm + Linear (768 → 588) — reconstructs RGB patches |
| Loss | L1 between predicted and original pixels (masked patches only) |
| Trainable | LoRA adapters (Q/V) + decoder head |

### Phase 2 — Supervised Fine-tuning

| Component | Detail |
|---|---|
| Data | 5,955 **labeled** DFU images |
| Backbone | `facebook/dinov2-base` (frozen) + pre-trained LoRA adapters |
| Loss | 0.5 × CrossEntropy + 0.5 × SupConLoss (`pytorch-metric-learning`) |
| Head | LayerNorm + Linear (768 → 4) |
| Trainable | LoRA adapters + classification head |

Only **0.68 % of parameters** are ever trainable, keeping GPU memory low.

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
│   ├── pretrain_model.py    # DFUDinoLoRAForMIM (MIM wrapper + decoder)
│   ├── pretrain.py          # Phase 1: unsupervised MIM pre-training
│   └── train.py             # Phase 2: supervised classification
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

### Phase 1 — Domain-Adaptive Pre-training

Debug run (20 unlabeled images, batch_size=2, 2 epochs):

```bash
cd DFU_Project
python -m src.pretrain --debug
```

Full pre-training run:

```bash
python -m src.pretrain
```

Override defaults:

```bash
python -m src.pretrain --epochs 30 --batch_size 4 --lr 5e-5 --mask_ratio 0.5
```

Outputs:
- `checkpoints/dfu_pretrain_mim.pt` — full model (backbone + decoder), for
  resuming pre-training.
- `checkpoints/dfu_pretrained_backbone.pt` — LoRA weights only, for Phase 2.

### Phase 2 — Supervised Fine-tuning

Debug run (20 labeled images, batch_size=2, 2 epochs):

```bash
python -m src.train --debug
```

Full training run with Phase-1 pre-trained LoRA weights (recommended):

```bash
python -m src.train \
    --pretrained_lora_path checkpoints/dfu_pretrained_backbone.pt
```

Full training run from scratch (no Phase-1 weights):

```bash
python -m src.train
```

Override defaults:

```bash
python -m src.train --epochs 30 --batch_size 4 --lr 5e-5 \
    --pretrained_lora_path checkpoints/dfu_pretrained_backbone.pt
```

## Initial Results (Debug Runs)

### Phase 1 — MIM Pre-training (debug)

```
Device     : cuda
Debug      : True
Epochs     : 2
Batch      : 2
Mask ratio : 0.6
Loss       : l1
Unlabeled imgs : 3,994 (debug subset: 20)

trainable params: 589,824 || all params: 87,170,304 || trainable%: 0.6766
decoder params : 453,708

Epoch 1/2  loss=0.757032  (3.0 s)
Epoch 2/2  loss=0.659234  (1.8 s)

Full pretrain model  → checkpoints/dfu_pretrain_mim.pt
Backbone LoRA weights → checkpoints/dfu_pretrained_backbone.pt
```

**Observations** — Reconstruction loss drops from 0.757 → 0.659 (−13 %) across
2 epochs on just 20 images, confirming the LoRA adapters + decoder learn to
reconstruct masked patches.

### Phase 2 — Supervised Fine-tuning (debug)

**From scratch** (no Phase-1 weights):

```
LoRA   : scratch
Epoch 1/2  loss=0.5419  ce=1.0838  con=0.0000  acc=0.550  (2.3 s)
Epoch 2/2  loss=0.4112  ce=0.8223  con=0.0000  acc=0.700  (1.9 s)
```

**With Phase-1 pre-trained LoRA weights** (`--pretrained_lora_path`):

```
LoRA   : checkpoints/dfu_pretrained_backbone.pt
Loaded Phase-1 LoRA weights from checkpoints/dfu_pretrained_backbone.pt
Epoch 1/2  loss=0.6986  ce=1.3972  con=0.0000  acc=0.400  (6.8 s)
Epoch 2/2  loss=0.4085  ce=0.8170  con=0.0000  acc=0.750  (1.9 s)
```

| Init | Epoch 2 acc | Epoch 2 loss |
|---|---|---|
| Scratch | 70.0 % | 0.4112 |
| Phase-1 LoRA | **75.0 %** | **0.4085** |

**Observations** — Domain-adapted LoRA weights give +5 pp accuracy and lower
final loss on the debug split. The gap is expected to grow on the full dataset.

## Cluster / HPC Deployment

The codebase is structured to be scheduler-agnostic. Once the target cluster
environment is confirmed, a dedicated deployment script will be written for
that specific setup.

> **Note:** The training entry-points (`src/pretrain` and `src/train`) are
> plain Python modules with no framework-specific launcher dependencies.
> They will be wrapped in a Slurm `sbatch` script, a PBS `qsub` script,
> a Kubernetes Job manifest, or equivalent — depending on the cluster type
> and configuration provided.

## License

This project is for academic and research purposes.
