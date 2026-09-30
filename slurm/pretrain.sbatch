#!/bin/bash
# Phase 1: SimMIM domain-adaptive pre-training on the unlabeled DFU images.
#
# Submit from the project root, naming your GPU partition:
#   sbatch --partition=<gpu-partition> slurm/pretrain.sbatch
# Override epochs:  EPOCHS=150 sbatch --partition=... slurm/pretrain.sbatch
#SBATCH --job-name=dfu_pretrain
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate

python -m src.pretrain --epochs "${EPOCHS:-100}" --batch_size 32 --num_workers 8
