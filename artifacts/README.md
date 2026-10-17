# artifacts/

Small data products of the experiments. The scripts expect some of them in the project root
(`groups.csv`, `pseudo_*.csv`); copy them there to reuse.

| File | What it is | Produced by |
|---|---|---|
| `groups.csv` | `image,group` — near-duplicate cluster id for each of the 5,955 labeled images (cosine ≥ 0.85, 3,925 groups, largest 114). Needed for `--group_csv` (leak-free split). | `python -m src.make_groups --auto` |
| `pseudo_g.csv`, `pseudo_g_probs.csv` | Teacher predictions on the 3,994 unlabeled images from the three *scratch + strong aug/LS* models (one-hot / probabilities). Used by the student (`--pseudo_thresh 0.7`). | `src.predict --unlabeled_train` |
| `pseudo_pre.csv`, `pseudo_pre_probs.csv` | Same, from the three *Phase-1* models. Used by the final models (`--pseudo_thresh 0.9 --pseudo_max_per_class 400`). | `src.predict --unlabeled_train` |
| `sub_final.csv`, `sub_final_probs.csv` | Challenge-test predictions (5,734 images) of the 3 final models (Phase 1 + pseudo-labels, trained on all labels), flip TTA. One-hot labels / probabilities. **Submitted 17 Oct 2026 — macro-F1 0.588, rank 39.** | `src.predict --img_dir <test> --tta` |
| `sub_priormatch_probs.csv` | `sub_final_probs.csv` re-balanced so arg-max predictions match the training label mix (per-class log-biases; 765 predictions change). Prepared before the validation protocol; **superseded by `results/calib/` (docs/CALIBRATION_PROTOCOL.md) — do not submit on its own.** | `python scripts/prior_match.py artifacts/sub_final_probs.csv` |
| `sub_groupmodels.csv`, `sub_groupmodels_probs.csv` | Challenge-test predictions of the 9 group-split models (trained on 70 % of the labels), flip TTA. | `src.predict --img_dir <test> --tta` |

Columns of the label files: `image,none,infection,ischaemia,both` (`none` = Control). Model checkpoints
(`checkpoints/*.pt`, a few MB each) are git-ignored.
