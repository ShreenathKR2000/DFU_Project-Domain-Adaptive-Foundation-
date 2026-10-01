#!/bin/bash
# Phase 2 comparison: 3 seeds x {scratch, pretrained} = 6 runs as a job array.
# Runs with the same seed use the same train/val/test split, so the two arms
# are directly comparable.  Run Phase 1 first (scripts/pretrain.sbatch).
#
#   sbatch --partition=<gpu-partition> scripts/train_array.sbatch
# Afterwards:  python -m src.summarize
#SBATCH --job-name=dfu_train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --array=0-5
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate

ARMS=(scratch pretrained)
ARM=${ARMS[$((SLURM_ARRAY_TASK_ID % 2))]}
SEED=$((SLURM_ARRAY_TASK_ID / 2))

ARGS=(--epochs "${EPOCHS:-30}" --batch_size 32 --seed "$SEED"
      --imbalance "${IMBALANCE:-weights}" --num_workers 8
      --run_name "${ARM}_seed${SEED}")
if [ "$ARM" = "pretrained" ]; then
  ARGS+=(--pretrained_lora_path "${LORA_PATH:-checkpoints/dfu_pretrained_backbone.pt}")
fi

python -m src.train "${ARGS[@]}"
