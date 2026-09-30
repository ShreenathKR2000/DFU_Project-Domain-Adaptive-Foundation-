# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Project history at a glance

| Stage | Versions | What happened |
|---|---|---|
| Initial state | 0.1.0 – 0.3.0 (Feb–Mar 2026) | DINOv2 + LoRA classifier, SimMIM pre-training (Phase 1) and the Phase 1 → 2 bridge, verified only with 20-image debug runs on a GTX 1650 (4 GB). Evidence for the method was a "+5 pp" debug comparison. |
| Move to an HPC GPU (L40S) | 0.4.0 | Proper held-out evaluation (stratified train/val/test, macro-F1, per-class recall, confusion matrix), class-imbalance options, seeds, Slurm scripts, scratch-vs-pretrained summary. |
| Bug found by the new checks | 0.4.1 | With older peft the Phase-1 LoRA weights were silently not applied, so every earlier "pretrained" run equalled scratch (the debug "+5 pp" was noise). Fixed; loading is now verified. |
| First real result | 0.4.x | 3 seeds x 2 arms on ~4,100 labeled images: scratch 0.880 ± 0.017 vs pretrained 0.870 ± 0.010 test macro-F1 — **no measurable benefit of Phase 1 at full labels**. |
| Label-efficiency study | 0.5.0 | `--train_frac`, grouped summary, CSV + plot, to test whether Phase 1 helps when labels are scarce. |
| Leaderboard reality check | 0.7.0 | Live DFUC2021 leaderboard: top ≈ 0.65 / rank ~50 ≈ 0.55 macro-F1 vs 0.88 internally, so the random split leaks near-duplicates. Added near-duplicate grouping, group-aware splits, stronger augmentation and label smoothing. |
| Using the unlabeled data for supervision | 0.8.0 | Noisy-student pseudo-labelling pipeline; leaderboard-style metrics (macro/micro AUC, per-class F1); summary rows separated by experiment variant. |
| Towards a leaderboard | 0.6.0 | Higher resolution, LoRA rank, extra unlabeled data for Phase 1 (e.g. DFUC2020), train-on-everything mode, ensemble + TTA inference and a submission-style CSV. |

## [0.8.0] - 2026-09-30

### Added

- Pseudo-labelling: `src.predict --unlabeled_train` (teacher predictions for the
  unlabeled training images), `src.train --pseudo_csv / --pseudo_thresh /
  --pseudo_max_per_class / --pseudo_img_dir`, `src.dataset.load_pseudo`, and
  `run_pseudo.sh` (3 teachers -> pseudo-labels -> student + matched baseline).
- Metrics: macro and micro AUC (as on the DFUC2021 leaderboard) in
  `compute_metrics`, `src.train` output and `src.predict` output.

### Changed

- `src/summarize.py` — rows are now keyed by arm **and experiment variant**
  (group split, augmentation, label smoothing, pseudo-labels, resolution, LoRA
  rank, imbalance mode), so different experiments are never averaged together
  (previously group-split or pseudo-label runs would have been merged with the
  random-split runs); table shows per-class F1 and macro AUC like the
  leaderboard.

### Verified

- Synthetic end-to-end run (teachers -> ensemble+TTA pseudo-labels -> student
  -> summary); re-evaluating the student checkpoint with `src.predict` reproduced
  the training-time metrics. Not yet run on the real data.

## [0.7.0] - 2026-09-30

### Added

- `src/make_groups.py` — embeds labeled images with frozen DINOv2, clusters
  near-duplicates (cosine threshold, connected components), writes `groups.csv`
  and prints a leak diagnostic (share of test images with a near-duplicate in
  train under random vs. group-aware splits).
- `src/dataset.py` — `load_groups`; `make_splits(..., groups=...)` uses
  `StratifiedGroupKFold` so each group stays in one split; `get_train_transforms(strong=True)`.
- `src/train.py`, `src/predict.py` — `--group_csv`; `src/train.py` — `--aug {basic,strong}`,
  `--label_smoothing`.

### Verified

- Synthetic check: 120 clusters of 6 near-duplicates recovered exactly; random
  split leaked 100 % of test images, group-aware split 0 %. Not yet run on the
  real data.

## [0.6.0] - 2026-09-30

### Added

- `src/predict.py` — inference with flip test-time augmentation and
  checkpoint ensembling. Evaluates on a held-out split (`--split val|test`, to
  measure what TTA/ensembling buy) or predicts an image folder (`--img_dir`) and
  writes `predictions.csv` (one-hot, train.csv schema) and `*_probs.csv`.
- `src/train.py` — `--final` (train on all labeled images for a fixed number of
  epochs, save `checkpoints/<run>_final.pt`), `--img_size` (multiple of 14),
  `--lora_rank`; `compute_metrics` factored out of `evaluate`.
- `src/pretrain.py` — `--extra_img_dirs` (add further unlabeled images such as
  DFUC2020 to Phase 1), `--img_size`, `--lora_rank`.
- README — leaderboard workflow and notes on using DFUC2020/2021 data.

### Changed

- `src/summarize.py` skips `--final` runs.

### Verified

- Synthetic-data smoke test (tiny random DINOv2, old and new peft/transformers):
  extra-directory pre-training, `--img_size 126`, `--final`, 2-checkpoint
  ensemble + TTA, CSV output, `--lora_rank 8`. Not yet run on the real data.

## [0.5.0] - 2026-09-30

### Added

- `src/train.py` — `--train_frac` (class-stratified subset of the training
  split, identical for runs sharing `--split_seed`; val/test unchanged) for
  low-label experiments; last-epoch test metrics (`test_last_epoch`) saved
  alongside best-validation-epoch metrics.
- `src/dataset.py` — `subsample_stratified`.
- `run_lowlabel.sh` — 10/25/50 % label fractions x {scratch, pretrained} for one seed.
- `src/summarize.py` — results grouped by label fraction, `results/summary.csv`,
  `results/label_efficiency.png` (macro-F1 vs. label fraction).
- README — full-label results (3 seeds: no significant benefit from Phase 1),
  low-label protocol, ideas for further study, notes on DFUC challenges.

## [0.4.1] - 2026-09-30

### Fixed

- `src/model.py` — with older peft versions (e.g. 0.13), `set_peft_model_state_dict`
  silently ignored every Phase-1 LoRA tensor, so "pretrained" runs were
  actually identical to scratch (the debug comparison in 0.3.0 is therefore not
  evidence for the method). The checkpoint is now loaded directly with
  `load_state_dict`; `set_peft_model_state_dict` is only a fallback.
- `src/summarize.py` — ignores `--debug` result files (a leftover debug run had
  been counted as a scratch run).

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

- `src/model.py` — loading Phase-1 LoRA weights fails loudly if no LoRA tensors
  are found or the load was silently ignored (all `lora_B` still zero); LoRA
  target names adapt to `transformers` 5.x (`q_proj`/`v_proj`).
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
