#!/bin/bash
# Pseudo-labelling (noisy-student style) for ONE split seed:
#   1. train 3 teachers on the labeled TRAIN split (same split, different seeds)
#   2. predict the 3,994 unlabeled training images with the teacher ensemble + TTA
#   3. train a student on train + confident pseudo-labels, with strong augmentation
#   4. train a baseline with identical settings but WITHOUT pseudo-labels
# Compare student vs. baseline with `python -m src.summarize` (rows differ by the
# "pseudo" tag). Run for several split seeds (0 1 2) to get a paired comparison.
#
#   GROUP_CSV=groups.csv ./run_pseudo.sh 0
# Optional env: ARM=pretrained (init from Phase-1 LoRA), LORA_PATH, THRESH (0.9),
#               CAP (max pseudo-labels per class), EPOCHS (30)
#
# The teachers never see this split's val/test labels, so the test metrics stay clean.
set -e
SPLIT=${1:?usage: GROUP_CSV=groups.csv ./run_pseudo.sh <split_seed>}
ARM=${ARM:-scratch}
EPOCHS=${EPOCHS:-30}
THRESH=${THRESH:-0.9}
LORA=${LORA_PATH:-checkpoints/dfu_pretrained_backbone.pt}

OPTS=(--split_seed "$SPLIT" --epochs "$EPOCHS" --batch_size 32 --imbalance weights --num_workers 8)
[ -n "$GROUP_CSV" ] && OPTS+=(--group_csv "$GROUP_CSV")
[ "$ARM" = "pretrained" ] && OPTS+=(--pretrained_lora_path "$LORA")
PSEUDO=(--pseudo_csv "pseudo_s${SPLIT}_probs.csv" --pseudo_thresh "$THRESH")
[ -n "$CAP" ] && PSEUDO+=(--pseudo_max_per_class "$CAP")

for T in 0 1 2; do
  python -m src.train "${OPTS[@]}" --seed $((100 + T)) --run_name "teacher_s${SPLIT}_$T"
done
python -m src.predict --checkpoints checkpoints/teacher_s${SPLIT}_{0,1,2}_best.pt \
    --unlabeled_train --tta ${GROUP_CSV:+--group_csv "$GROUP_CSV"} --out "pseudo_s${SPLIT}.csv"

python -m src.train "${OPTS[@]}" --seed "$SPLIT" --aug strong "${PSEUDO[@]}" \
    --run_name "student_s${SPLIT}"
python -m src.train "${OPTS[@]}" --seed "$SPLIT" --aug strong \
    --run_name "baseline_strong_s${SPLIT}"
