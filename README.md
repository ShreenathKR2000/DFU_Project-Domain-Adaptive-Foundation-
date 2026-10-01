# Domain-Adaptive Foundation Models for Fine-Grained DFU Classification

Diabetic Foot Ulcer (DFU) classification on **DFUC2021** with a frozen **DINOv2 ViT-B/14**
backbone, **LoRA** adapters (0.68 % of the parameters), **supervised contrastive learning**,
self-supervised **domain adaptation** on unlabeled images, **pseudo-labelling**, and a
**leak-free evaluation protocol**.

- **Task:** 4 classes — `none` (leaderboard name: *Control*), `infection`, `ischaemia`, `both`.
- **Data:** 5,955 labeled + 3,994 unlabeled training images, 5,734 hidden-label test images.
  Strong imbalance (ischaemia ≈ 4 % of the labels).
- **Leaderboard metric:** macro-F1 (plus per-class F1 and AUC) — see [docs/LEADERBOARD.md](docs/LEADERBOARD.md).

![From an optimistic internal score to the leaderboard range](docs/figures/fig_story.png)

## Findings in one page

All numbers are on held-out test splits of the public training data (mean ± std over 3 seeds)
unless stated. Full details, figures and caveats: [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).

| # | Question | Finding |
|---|---|---|
| 1 | Is the usual random split trustworthy? | **No.** 22 % of random-split test images have a near-duplicate (cosine ≥ 0.90) in the training split. Re-evaluating with a group-aware split lowers the same scratch model from **0.880 to 0.773** macro-F1. |
| 2 | Does domain-adaptive pre-training (Phase 1) help? | **Modestly.** On the leak-free split Phase 1 beats scratch in 3/3 seeds (**0.799 vs 0.773**, +0.026; ischaemia and "both" F1 +0.06/+0.05). On the random split it looked useless (0.870 vs 0.880): leakage hid the effect. Caveat: on the last-epoch model the gain is only +0.004. |
| 3 | Do the unlabeled images help as pseudo-labels? | **A little.** Phase 1 + pseudo-labels: **0.821** (+0.022 over Phase 1 alone, 3/3 seeds). Few rare-class pseudo-labels pass the confidence filter, so the gain probably comes mostly from extra common-class data. |
| 4 | Do strong augmentation + label smoothing help? | **No.** Macro-AUC drops (0.936 → 0.905) and ischaemia F1 drops (0.712 → 0.655) for scratch models. |
| 5 | Do ensembles / flip-TTA help? | **Yes — the most reliable gain.** 3-model ensembles add ≈ +0.03 macro-F1, flip-TTA another +0.01–0.02. Best: **0.838** (3 models) and **0.847** (9 models) with TTA. |
| 6 | How do the models behave on the real challenge test images? | Predicted class mix differs from training (ischaemia 8–10 % vs 4 %, infection 29–34 % vs 43 %): a **distribution shift**. No leaderboard score exists yet (not submitted). |

> **What is *not* shown.** No leaderboard score; no low-label study; results come from one
> fixed split (seeds vary initialisation only); ischaemia has only 29 test images, so
> per-class results are noisy. The leaderboard (top-4 ≈ 0.65–0.66 macro-F1) uses a hidden,
> shifted test set, so our internal 0.77–0.85 is **not comparable** with it.

## Approach

![Pipeline](docs/figures/fig_pipeline.png)

1. **Leak-free split** — cluster near-duplicate images (frozen DINOv2 embeddings) and keep each
   cluster in one split ([`src/make_groups.py`](src/make_groups.py)).
2. **Phase 1** — SimMIM masked-image modelling: mask 60 % of patches, reconstruct pixels, train
   only LoRA adapters + a light decoder on the 3,994 unlabeled images ([`src/pretrain.py`](src/pretrain.py)).
3. **Phase 2** — cross-entropy + supervised-contrastive loss on the [CLS] embedding; only LoRA +
   a linear head are trained ([`src/train.py`](src/train.py)).
4. **Pseudo-labelling** — teachers label the unlabeled images; confident ones join training.
5. **Final models + inference** — train on all labels (+ pseudo-labels), then ensemble with
   flip test-time augmentation ([`src/predict.py`](src/predict.py)).

Method details, hyper-parameters and metric definitions: [docs/METHODS.md](docs/METHODS.md).

**How it differs from the usual DFUC2021 recipe** (supervised CNN ensembles on the labeled
images, as far as I know): a *foundation model* adapted with very few trainable parameters
(≈ 5 min per model on an L40S, so ensembles and pseudo-label rounds are cheap); *self-supervised
adaptation* and *pseudo-labelling* of the unlabeled images; a *leak-free evaluation protocol*;
and metrics reported exactly like the leaderboard.

## Quick start

```bash
# 1. Environment (PyTorch build must match your CUDA driver)
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# 2. Data (not tracked): Data/DFUC2021_train/{images/*.jpg, train.csv}

# 3. Smoke test (20 images, seconds)
python -m src.pretrain --debug && python -m src.train --debug

# 4. Everything in one go (~45 min on one L40S): groups, ablations, pseudo-labels, ensembles
nohup bash scripts/run_all_experiments.sh > logs/all.log 2>&1 &
```

Step-by-step commands for each experiment are in [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md);
cluster / Slurm / VS Code instructions in [docs/CLUSTER.md](docs/CLUSTER.md).

| Goal | Command |
|---|---|
| Phase 1 pre-training | `python -m src.pretrain [--epochs N --extra_img_dirs DIR ...]` |
| Near-duplicate groups | `python -m src.make_groups --auto` → `groups.csv` |
| Phase 2 + held-out evaluation | `python -m src.train --group_csv groups.csv [--pretrained_lora_path ...]` |
| Pseudo-labels (teacher step) | `python -m src.predict --checkpoints ... --unlabeled_train --tta --out pseudo.csv` |
| Final models on all labels | `python -m src.train --final --epochs 20 --pseudo_csv pseudo_probs.csv ...` |
| Ensemble + TTA / submission CSV | `python -m src.predict --checkpoints ... (--split test \| --img_dir DIR) --tta` |
| Compare runs | `python -m src.summarize` |
| Rebuild figures and tables | `python scripts/make_figures.py` |

## Repository layout

```
src/            dataset, common, metrics, model, pretrain(+_model), train, predict, make_groups, summarize
scripts/        run_all_experiments.sh (one-shot suite), run_pseudo.sh, run_lowlabel.sh, *.sbatch, make_figures.py
docs/           METHODS.md, EXPERIMENTS.md, LEADERBOARD.md, CLUSTER.md, results_tables.md, figures/
results/        one JSON per run (metrics, confusion matrix, per-epoch history), summary.csv, ensemble_round2.csv
logs/           raw training / evaluation logs (see logs/README.md)
artifacts/      groups.csv, pseudo-label CSVs, challenge-test prediction CSVs (see artifacts/README.md)
Notebooks/      01_data_explore.ipynb (EDA)
CHANGELOG.md    full history from the initial state
```

## Documentation

- [docs/METHODS.md](docs/METHODS.md) — what each component is and why, protocol, metrics, hyper-parameters
- [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) — every experiment: motivation, setup, results, figures, interpretation, limitations
- [docs/LEADERBOARD.md](docs/LEADERBOARD.md) — leaderboard context, our position, what to expect, submission workflow
- [docs/results_tables.md](docs/results_tables.md) — all tables (generated)
- [docs/CLUSTER.md](docs/CLUSTER.md) — running on an HPC cluster, troubleshooting
- [CHANGELOG.md](CHANGELOG.md) — what changed and why

## License

For academic and research purposes.
