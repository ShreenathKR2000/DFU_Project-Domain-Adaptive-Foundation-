#!/bin/bash
# Low-label experiment for ONE seed: scratch vs. Phase-1 pretrained at 10/25/50 %
# of the training labels.  Epochs are raised for small fractions so every
# fraction gets a comparable number of gradient steps.
#
#   ./scripts/run_lowlabel.sh 0        # seed 0
#   (run seeds 1 and 2 the same way, e.g. in parallel with nohup)
#
# The full-label (100 %) runs come from the main comparison (run_rest.sh).
set -e
cd "$(dirname "$0")/.."   # run from the project root
SEED=${1:?usage: ./scripts/run_lowlabel.sh <seed>}
LORA=${LORA_PATH:-checkpoints/dfu_pretrained_backbone.pt}

for CFG in "0.1 100" "0.25 60" "0.5 40"; do
  set -- $CFG; FRAC=$1; EPOCHS=$2
  TAG=f$(python3 -c "print(int(round($FRAC*100)))")
  python -m src.train --epochs $EPOCHS --batch_size 32 --seed $SEED \
      --imbalance weights --num_workers 8 --train_frac $FRAC \
      --run_name scratch_${TAG}_seed$SEED
  python -m src.train --epochs $EPOCHS --batch_size 32 --seed $SEED \
      --imbalance weights --num_workers 8 --train_frac $FRAC \
      --run_name pretrained_${TAG}_seed$SEED --pretrained_lora_path "$LORA"
done
