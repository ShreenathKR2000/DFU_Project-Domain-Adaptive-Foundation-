# Domain-Adaptive Foundation Models for Fine-Grained DFU Classification

Diabetic Foot Ulcer (DFU) classification on **DFUC2021** with a frozen
**DINOv2 ViT-B/14** backbone, **LoRA** adapters (0.68 % of the parameters) and
**supervised contrastive learning**, plus self-supervised domain adaptation and
pseudo-labelling of the unlabeled images.

- **Task:** 4 classes — `none` (leaderboard: *Control*), `infection`, `ischaemia`, `both`.
- **Data:** 5,955 labeled + 3,994 unlabeled training images; extreme imbalance
  (`ischaemia` is ~4 % of the labels).
- **Ranking metric (leaderboard):** macro-F1 — see [docs/LEADERBOARD.md](docs/LEADERBOARD.md).

## Status at a glance

| Question | Answer so far |
|---|---|
| Does the pipeline run on the cluster GPU? | Yes (L40S, ~20 s / epoch, ~10 min per 30-epoch run) |
| Internal result (random split, 3 seeds) | scratch **0.880 ± 0.017** vs Phase-1 **0.870 ± 0.010** test macro-F1 — no measurable benefit of Phase 1 |
| Is that number comparable to the leaderboard? | **No.** Top-4 leaderboard entries have macro-F1 ≈ 0.65–0.66; the random split leaks near-duplicates and is over-optimistic |
| Leak-free (group-aware) results | *pending — run `src.make_groups`, see [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)* |
| Low-label / pseudo-labelling studies | *implemented, results pending* |
| Leaderboard score of this approach | *to be filled in after submission* |

## Approach

```
unlabeled images ──► Phase 1: SimMIM masked-image modelling (LoRA + light decoder)
(3,994 + optional extra)            │ domain-adapted LoRA weights
                                    ▼
labeled images ─────► Phase 2: CE + SupCon on [CLS], LoRA + linear head
  ▲  group-aware split (no near-duplicate leakage)     │ teachers
  │                                                    ▼
  └──────── Phase 3: pseudo-label the unlabeled images → student (strong aug)
                                                       ▼
                       Inference: ensemble of models + flip TTA → submission CSV
```

**How it differs from the usual DFUC2021 recipe** (supervised CNN ensembles on the
5,955 labeled images, as far as I know): a *foundation model* adapted with very few
trainable parameters (cheap, hard to overfit, so many ensemble members and
pseudo-label rounds are affordable); *self-supervised adaptation* on the unlabeled
images; a *leak-free evaluation protocol*; and all metrics reported like the
leaderboard. Whether this beats CNN ensembles is an open empirical question —
see [docs/LEADERBOARD.md](docs/LEADERBOARD.md) for an honest assessment.

## Quick start

```bash
# 1. Environment (PyTorch build must match your CUDA driver)
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# 2. Data: Data/DFUC2021_train/{images/*.jpg, train.csv}   (not tracked by git)

# 3. Smoke test (20 images, seconds)
python -m src.pretrain --debug && python -m src.train --debug

# 4. Real runs (set --batch_size 32 on a large GPU)
python -m src.pretrain --epochs 100 --batch_size 32 --num_workers 8
python -m src.train --epochs 30 --batch_size 32 --imbalance weights --num_workers 8 \
    --pretrained_lora_path checkpoints/dfu_pretrained_backbone.pt --run_name pretrained_seed0
```

Cluster / Slurm / VS Code instructions: [docs/CLUSTER.md](docs/CLUSTER.md).

## Commands

| Goal | Command |
|---|---|
| Phase 1 pre-training | `python -m src.pretrain [--epochs N --extra_img_dirs DIR ...]` |
| Phase 2 training + held-out evaluation | `python -m src.train [--pretrained_lora_path ...]` |
| Leak-free split (near-duplicate groups) | `python -m src.make_groups` then `--group_csv groups.csv` |
| Regularisation for shift | `--aug strong --label_smoothing 0.1` |
| Low-label study (10/25/50 % labels) | `./scripts/run_lowlabel.sh <seed>` |
| Pseudo-labelling (teachers → student) | `GROUP_CSV=groups.csv ./scripts/run_pseudo.sh <split_seed>` |
| Compare runs (leaderboard-style table, CSV, plot) | `python -m src.summarize` |
| Final models on all labels | `python -m src.train --final --epochs 30 --run_name final1` |
| Ensemble + TTA inference / submission CSV | `python -m src.predict --checkpoints ... (--split test \| --img_dir DIR)` |
| Slurm | `sbatch --partition=<gpu> scripts/pretrain.sbatch` / `scripts/train_array.sbatch` |

## Repository layout

```
src/
  dataset.py         data, transforms, splits (stratified / group-aware), pseudo-labels
  common.py          paths, seeding, DataLoader factory
  metrics.py         accuracy / macro-F1 / per-class F1 / AUC / confusion matrix
  model.py           DFUDinoLoRA (DINOv2 + LoRA + head)
  pretrain_model.py  SimMIM wrapper (Phase 1)
  pretrain.py        Phase 1 entry point
  train.py           Phase 2 entry point (+ --final, --train_frac, --pseudo_csv)
  predict.py         ensemble + TTA inference, submission CSV, teacher predictions
  make_groups.py     near-duplicate clustering and leak diagnostic
  summarize.py       results table / CSV / plot, separated by experiment variant
scripts/             run_lowlabel.sh, run_pseudo.sh, *.sbatch
docs/                EXPERIMENTS.md, LEADERBOARD.md, CLUSTER.md
Notebooks/           01_data_explore.ipynb (EDA)
CHANGELOG.md         full history from the initial state
```

## Documentation

- [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) — evaluation protocol, results, how to run each study
- [docs/LEADERBOARD.md](docs/LEADERBOARD.md) — leaderboard metrics, targets, what to expect, submission workflow
- [docs/CLUSTER.md](docs/CLUSTER.md) — running on an HPC cluster (Open OnDemand, VS Code, Slurm) and troubleshooting
- [CHANGELOG.md](CHANGELOG.md) — what changed and why

## License

For academic and research purposes.
