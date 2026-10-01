# results/

| File | Content |
|---|---|
| `<run>.json` | One file per training run: `config` (all CLI settings), `split_sizes`, `split_class_counts`, `best_epoch`, `test` (metrics at the best-validation epoch: macro-F1, per-class precision/recall/F1/support, macro/micro AUC, accuracy, confusion matrix), `test_last_epoch`, per-epoch `history` (train loss/CE/SupCon/accuracy, validation metrics), `train_seconds`. |
| `*_final.json` | The three final models (trained on all labels; no test metrics, only config and training history). |
| `summary.csv` | Per-run table written by `python -m src.summarize` (macro-F1, last-epoch F1, AUC, per-class F1). |
| `ensemble_round2.csv` | Ensemble + TTA scores transcribed from terminal output (no log file exists). |

Run names: `scratch_seed*` / `pretrained_seed*` are the **random-split** runs (894-image test split);
`g_*` are the **group-aware** runs (813-image test split): `g_scratch_basic`, `g_pre_basic`,
`g_pre_pseudo`, `g_scratch_reg`, `g_pre_reg`, `g_student` (suffix `_s0..2` = seed).
Condition definitions: [docs/EXPERIMENTS.md](../docs/EXPERIMENTS.md) (E4).
