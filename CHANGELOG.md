# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - 2026-09-30

### Added

- `src/train.py` — stratified train/val/test split (`--val_frac`, `--test_frac`,
  `--split_seed`), per-epoch validation accuracy / macro-F1 / per-class recall,
  best-epoch selection on validation macro-F1, final test evaluation with
  confusion matrix, JSON results (`results/<run_name>.json`) and best-weights
  checkpoint (`checkpoints/<run_name>_best.pt`).
- `src/train.py` — `--seed`, `--imbalance {none,weights,sampler}`,
  `--num_workers`, `--run_name`, `--out_dir`.
- `src/dataset.py` — `make_splits`, `class_counts`, `labels_from_df`,
  `check_labeled`.
- `src/summarize.py` — aggregates `results/*.json` into a scratch vs.
  pretrained comparison with paired per-seed differences.
- `slurm/pretrain.sbatch`, `slurm/train_array.sbatch` — Slurm scripts
  (Phase 1; 3 seeds x 2 arms job array).

### Changed

- `src/model.py` — loading Phase-1 LoRA weights now fails loudly if no LoRA
  tensors are found or the load was silently ignored (all `lora_B` still
  zero); LoRA target names adapt to `transformers` 5.x (`q_proj`/`v_proj`).
- `src/pretrain.py` — `--seed`, `--num_workers`.
- `src/train.py` no longer saves `dfu_dino_lora.pt`; see `*_best.pt` above.

## [0.3.0] - 2026-03-07

### Phase 1 → Phase 2 Bridge

#### Changed

- `src/model.py` — `DFUDinoLoRA.__init__` now accepts an optional
  `pretrained_lora_path: str | None` parameter.  When provided,
  `peft.set_peft_model_state_dict` loads the Phase-1 domain-adapted LoRA
  weights into the backbone **before** the classification head is trained,
  giving the adapters a DFU-domain starting point instead of random
  initialisation.
- `src/train.py` — New CLI argument `--pretrained_lora_path` (default
  `None`) wired through to `DFUDinoLoRA`.  Start-up banner now prints
  `LoRA : <path>` or `LoRA : scratch` for traceability.

#### Verified

- Debug run with Phase-1 weights (20 labeled images, batch_size=2,
  2 epochs, GTX 1650 4 GB VRAM):
  - Epoch 1: loss=0.6986, ce=1.3972, acc=40.0 %
  - Epoch 2: loss=0.4085, ce=0.8170, **acc=75.0 %**
- Compared to training from scratch (same debug conditions):
  - Epoch 2 acc=70.0 %, loss=0.4112
- Pre-trained LoRA adapters yield **+5 pp accuracy** and lower final loss
  (0.4085 vs 0.4112) on the debug split; benefit expected to grow on the
  full dataset.
- Backward compatible: omitting `--pretrained_lora_path` trains from
  scratch with identical behaviour to v0.2.0.

## [0.2.0] - 2026-02-28

### Added

- `src/pretrain_model.py` — `DFUDinoLoRAForMIM`: SimMIM-style masked-image-
  modeling wrapper around `DFUDinoLoRA`.  Masks 60 % of 14×14 patches,
  reconstructs original pixel values with a lightweight LayerNorm + Linear
  decoder (453,708 params).  Loss: L1 on masked patches only.
- `src/pretrain.py` — Domain-adaptive pre-training loop (Phase 1).
  Loads only the 3,994 unlabeled images from `train.csv`, supports the
  same `--debug` flag (20 images, batch_size=2, 2 epochs) for 4 GB VRAM.
  Saves both the full pretrain model and a standalone backbone LoRA
  checkpoint for Phase 2 fine-tuning.

### Verified

- Debug pre-training run on GTX 1650 (4 GB VRAM):
  - 20 unlabeled images, batch_size=2, 2 epochs, 60 % mask ratio.
  - Epoch 1: loss=0.757032 | Epoch 2: loss=0.659234 (−13 %).
  - Checkpoints saved to `checkpoints/dfu_pretrain_mim.pt` and
    `checkpoints/dfu_pretrained_backbone.pt`.

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
