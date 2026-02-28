# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] - 2026-02-28

### Verified

- First successful end-to-end debug training run on GTX 1650 (4 GB VRAM).
  - 20 labeled images, batch_size=2, 2 epochs.
  - Trainable params: 589,824 / 87,170,304 (0.68 %).
  - Epoch 1: loss=0.8843, ce=1.7687, acc=0.0 %
  - Epoch 2: loss=0.5786, ce=1.1572, acc=60.0 %
  - Checkpoint saved to `checkpoints/dfu_dino_lora.pt`.
- Confirmed `python -m src.train --debug` is the correct entry-point
  (direct `python src/train.py` triggers a relative-import error).

## [0.1.0] - 2026-02-28

### Added

- `src/dataset.py` — `DFUDataset` with labeled/unlabeled handling, ImageNet
  transforms (resize 224x224, normalise), and `load_and_split_csv` helper.
- `src/model.py` — `DFUDinoLoRA` wrapping frozen DINOv2 ViT-B/14 with LoRA
  rank-16 adapters on Q/V projections and a 4-class linear head.
- `src/train.py` — Training loop with `--debug` flag (20 images, batch_size=2,
  2 epochs) for GTX 1650 4 GB VRAM. Combined loss: CrossEntropy + SupConLoss.
- `Notebooks/01_data_explore.ipynb` — EDA notebook: CSV split, class
  distribution bar chart, sample images per class.
- `.gitignore` — Ignores `Data/`, checkpoints, `__pycache__`, Jupyter
  checkpoints, IDE files.
- `README.md` — Project overview, setup instructions, usage guide.
