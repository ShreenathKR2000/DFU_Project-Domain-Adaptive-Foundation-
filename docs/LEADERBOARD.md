# DFUC2021 leaderboard: metrics, targets and expectations

The DFUC2021 *open* (testing-set) leaderboard ranks submissions by **macro-F1** over the four
classes. This page records the reference numbers, how the approach relates to them, and what
can realistically be expected. Nothing here is a measured result of this project unless stated.

## 1. Reference scores (live leaderboard snapshot)

| Rank / user | Macro-F1 | Control | Infection | Ischaemia | Both | Macro prec. | Macro recall | Macro AUC | Micro F1 | Micro AUC |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 Mahamudul_Hasan | **0.6592** | 0.7594 | 0.7145 | 0.5996 | 0.5634 | 0.6652 | 0.6593 | 0.8945 | 0.7172 | 0.9165 |
| 2 mdzh10 (Team_Sagittarius) | 0.6537 | 0.7570 | 0.6898 | 0.5843 | 0.5836 | 0.6697 | 0.6433 | 0.8902 | 0.7063 | 0.9111 |
| 3 iftekhar.ahmed (Team_Sagittarius) | 0.6534 | 0.7456 | 0.6484 | 0.6278 | 0.5919 | 0.6488 | 0.6874 | 0.8910 | 0.6855 | 0.9075 |
| 4 ffyytt (DFU-MORIS) | 0.6517 | 0.7709 | 0.6424 | 0.6118 | 0.5817 | 0.6520 | 0.6863 | 0.8698 | 0.6926 | 0.8989 |
| 32 sayefshahriar | 0.5923 | 0.6816 | 0.6865 | 0.5067 | 0.4945 | 0.6175 | 0.5795 | 0.8686 | 0.6613 | 0.8967 |
| 39 mohid | 0.5817 | 0.7371 | 0.6031 | 0.5088 | 0.4778 | 0.5821 | 0.6086 | 0.8257 | 0.6478 | 0.8483 |
| 49 d4mz | 0.5507 | 0.7324 | 0.6011 | 0.4397 | 0.4294 | 0.5579 | 0.5983 | 0.8596 | 0.6382 | 0.8783 |
| **this project** | *TBD* | | | | | | | | | |

Top-4 ranges: macro-F1 **0.652–0.659** (spread only 0.008), Control F1 0.746–0.771, Infection
0.642–0.715, Ischaemia 0.584–0.628, Both 0.563–0.592, macro AUC 0.870–0.895, micro F1 0.686–0.717.
Mid-table (ranks ~32–50): macro-F1 0.55–0.59, macro AUC 0.83–0.87.

## 2. What the numbers say

- **Hard task:** even the best entries are at ~0.66 macro-F1 and ~0.72 micro-F1 (accuracy).
- **Rare classes decide the ranking:** Control F1 is similar across the board (0.73–0.77);
  ischaemia and "both" (0.43–0.63) are where entries differ.
- **AUC ≫ F1:** macro AUC ≈ 0.87–0.89 vs macro-F1 ≈ 0.65 — the models rank images reasonably but the
  hard decisions are off, consistent with a shifted class distribution / calibration between training
  and the hidden test set. Per-class decision thresholds are therefore a cheap lever (tune on
  group-validation data, not on the leaderboard).
- **Internal scores are not comparable:** this project's random-split macro-F1 (0.88) is far above
  the leaderboard because of near-duplicate leakage and because the test set is shifted.
  See [EXPERIMENTS.md](EXPERIMENTS.md) §2–3.

## 3. How this approach differs, and why it might help

| Aspect | Typical supervised entry (as far as I know) | This project |
|---|---|---|
| Backbone | ImageNet CNN/ViT fully fine-tuned, often an ensemble | Frozen DINOv2 ViT-B/14 + LoRA (0.68 % trainable), ensemble of cheap models |
| Unlabeled data (3,994 images) | Often unused, or pseudo-labelling | Self-supervised domain adaptation (SimMIM) **and** pseudo-labelling |
| Loss | Cross-entropy (+ class weights / focal) | CE + supervised contrastive on [CLS] + class weights |
| Validation | Random hold-out / k-fold | Group-aware split against near-duplicate leakage |
| Inference | Flip/TTA + ensemble | Flip TTA + ensemble (`src.predict`) |

Potential advantages: strong general-purpose features with little capacity to overfit (important
with ~230 ischaemia images); ~10 minutes per model on an L40S, so many seeds, ensembles and
pseudo-label rounds are affordable; a principled way to use the unlabeled images.

**What the evidence says so far:** on the random split, Phase 1 gave *no* measurable gain (0.870 vs
0.880). That result must be re-checked on the leak-free split and in the low-label setting before
any claim is made. The approach may still help through the other components (ensemble + TTA,
pseudo-labelling, regularisation for shift), which have not yet been measured on the real data.

## 4. Metrics to expect — honest estimate

These are planning guesses, not results:

- **Single model, group-aware validation:** probably in the 0.55–0.65 macro-F1 range if the split
  removes the leakage; a value near 0.88 would mean the split still leaks.
- **Leaderboard, single DINOv2-LoRA model:** plausibly mid-table (≈ 0.55–0.60 macro-F1, i.e. ranks
  ~30–50) — comparable to the entries above.
- **Ensemble + TTA + pseudo-labelling + regularisation:** a further gain of a few points is plausible
  but unproven; ensembles and TTA are the most reliable levers, the rest must be validated.
- **Top-5 (≥ 0.65):** would need ≈ +0.05–0.10 over a mid-table single model. The top-4 are separated by
  < 0.01, so this is ambitious; treat it as a stretch goal and the research comparisons (Phase 1,
  low-label, pseudo-labelling, group-aware evaluation) as the primary deliverable.

Targets to watch on every evaluation: macro-F1 ≥ 0.65, ischaemia F1 ≥ 0.58, both F1 ≥ 0.56,
macro AUC ≥ 0.89.

## 5. Submission workflow

1. Decide settings on the **group-aware** split (epochs, `--aug`, `--label_smoothing`, resolution, rank).
2. Train 3–5 models on all labels: `python -m src.train --final --seed S --epochs E --run_name finalS`.
3. Predict the challenge images with ensemble + TTA:
   `python -m src.predict --checkpoints checkpoints/final*_final.pt --img_dir <images> --tta --out predictions.csv`
   (one-hot CSV in the `train.csv` schema plus `*_probs.csv`; adapt columns if the submission format differs).
4. Submit, then log the leaderboard score next to the group-validation score. If they track each other,
   the evaluation is trustworthy; if not, revisit the grouping threshold.

## 6. Rules and risks

- Check the challenge rules on external data, pre-trained weights (DINOv2 must be permitted),
  pseudo-labelling and use of unlabeled test images before relying on them.
- Do not tune on the leaderboard; use it as a sparse check of shift.
- DFUC2020 (as far as I recall, bounding-box detection data) cannot train the 4-class head; it can only add
  unlabeled images to Phase 1 (`--extra_img_dirs`). DFUC2022 (segmentation, Dice) is a different task
  and would need masks and a segmentation head.
