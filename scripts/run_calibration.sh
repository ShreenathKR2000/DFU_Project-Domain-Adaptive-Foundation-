#!/bin/bash
# Validate decision-rule calibration on the leak-free split, then apply it to the
# challenge predictions only if it passes a rule fixed in advance (src/calibrate.py).
#
#   RULE=noninferiority ./scripts/run_calibration.sh
#
# Steps
#   1. ensemble + TTA probabilities (with true labels) on the group-aware VAL split
#   2. the same on the group-aware TEST split (used once, for the comparison)
#   3. src.calibrate: compare none / prior / val_bias, bootstrap, apply the decision rule,
#      and write results/calib/challenge_<method>_probs.csv + calibration.json
#
# Needs the group-split checkpoints from scripts/run_all_experiments.sh (default: the
# Phase 1 + pseudo-label recipe, the same recipe as the final models) and the DFUC2021
# training images. Steps 1–2 take a few minutes on one GPU; step 3 runs on CPU in seconds.
#
# Optional env: CKPTS (checkpoint glob), GROUP_CSV (artifacts/groups.csv),
#               CHALLENGE (artifacts/sub_final_probs.csv), OUT (results/calib)
set -e
cd "$(dirname "$0")/.."   # run from the project root
RULE=${RULE:?set RULE=noninferiority or RULE=superiority — decide BEFORE running}
CKPTS=${CKPTS:-checkpoints/g_pre_pseudo_s*_best.pt}
GROUP_CSV=${GROUP_CSV:-artifacts/groups.csv}
CHALLENGE=${CHALLENGE:-artifacts/sub_final_probs.csv}
OUT=${OUT:-results/calib}
mkdir -p "$OUT"

# shellcheck disable=SC2086
ls $CKPTS >/dev/null || { echo "No checkpoints match $CKPTS"; exit 1; }
echo "Rule: $RULE   checkpoints: $(ls $CKPTS | tr '\n' ' ')" | tee "$OUT/run.log"

for SPLIT in val test; do
  # shellcheck disable=SC2086
  python -m src.predict --checkpoints $CKPTS --split "$SPLIT" --split_seed 0 \
      --group_csv "$GROUP_CSV" --tta --out "$OUT/${SPLIT}.csv" 2>&1 | tee -a "$OUT/run.log"
done

python -m src.calibrate --val "$OUT/val.csv" --test "$OUT/test.csv" \
    --challenge "$CHALLENGE" --rule "$RULE" --out_dir "$OUT" 2>&1 | tee -a "$OUT/run.log"
