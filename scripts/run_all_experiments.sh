#!/bin/bash
# One-shot experiment suite, sized for a ~90 minute GPU window.
#
#   nohup bash scripts/run_all_experiments.sh > logs/all.log 2>&1 &
#   tail -f logs/all.log
#
# Everything uses ONE fixed, leak-free (near-duplicate-grouped) split (--split_seed 0);
# replicates differ only in --seed. Conditions (all macro-F1 on the held-out test split):
#   A  scratch      basic aug          -> baseline
#   B  pretrained   basic aug          -> effect of Phase 1 (domain-adaptive MIM)
#   C  scratch      strong aug + LS    -> effect of regularisation (also the teachers)
#   D  pretrained   strong aug + LS
#   S  student      C + pseudo-labels  -> effect of pseudo-labelling (vs. C)
# plus ensemble / flip-TTA evaluation of the seed replicates.
#
# Jobs are scheduled in priority order and a job is only started if it is expected to
# finish inside the time budget, so you always get a usable (possibly partial) result.
#
# Environment knobs (defaults in brackets):
#   BUDGET_MIN [80]   total wall-clock budget; leave ~10 min spare in your allocation
#   PAR [2]           jobs run concurrently on the GPU
#   EPOCHS [20]       Phase-2 epochs per run        PRE_EPOCHS [30]  Phase-1 epochs (if needed)
#   SEEDS [3]         max replicates per condition  THRESH [0.7]     pseudo-label confidence
#   CAP [400]         max pseudo-labels per class   LORA_PATH [checkpoints/dfu_pretrained_backbone.pt]
#   DFU_DATA_DIR      data dir if not Data/DFUC2021_train   DFU_MODEL  local DINOv2 dir if offline

cd "$(dirname "$0")/.." || exit 1
mkdir -p logs/exp results checkpoints

BUDGET=${BUDGET_MIN:-80}
PAR=${PAR:-2}
EPOCHS=${EPOCHS:-20}
PRE_EPOCHS=${PRE_EPOCHS:-30}
NSEEDS=${SEEDS:-3}
THRESH=${THRESH:-0.7}     # teachers use label smoothing 0.1 => confidences saturate near 0.9
CAP=${CAP:-400}
LORA=${LORA_PATH:-checkpoints/dfu_pretrained_backbone.pt}
RESERVE=8                 # minutes kept for the final ensemble evaluation + summary
START=$(date +%s)
EST=0                     # expected minutes for one job (measured after the first stage)
PROG=logs/exp/progress.log
rm -f logs/exp/*.dur

elapsed() { echo $(( ($(date +%s) - START) / 60 )); }
log() { echo "[$(printf '%3d' "$(elapsed)")m] $*" | tee -a "$PROG"; }

# ── activate the environment ──────────────────────────────────────────────────
[ -f .venv/bin/activate ] && source .venv/bin/activate
if ! python -c "import torch, transformers, peft, sklearn, scipy, pytorch_metric_learning" 2>/dev/null; then
  log "Python packages missing — creating .venv and installing (needs internet, ~3-5 min)"
  python3 -m venv .venv && source .venv/bin/activate || exit 1
  pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu118 || exit 1
  pip install -q -r requirements.txt || exit 1
fi

# ── pre-flight checks ─────────────────────────────────────────────────────────
DATA=${DFU_DATA_DIR:-Data/DFUC2021_train}
[ -f "$DATA/train.csv" ] || { log "ERROR: $DATA/train.csv not found (set DFU_DATA_DIR)"; exit 1; }
NIMG=$(ls "$DATA/images" 2>/dev/null | wc -l)
log "Data: $DATA  ($NIMG images).  Expect 9949 for the full DFUC2021 training set."
MIN_IMAGES=${MIN_IMAGES:-5955}
[ "$NIMG" -ge "$MIN_IMAGES" ] || { log "ERROR: fewer than $MIN_IMAGES images in $DATA/images"; exit 1; }

if [ -z "$ALLOW_CPU" ]; then      # ALLOW_CPU=1 is for plumbing tests only
  python -c "
import sys, torch
ok = torch.cuda.is_available()
print('CUDA available:', ok, torch.cuda.get_device_name(0) if ok else '')
sys.exit(0 if ok else 1)" || { log "ERROR: no GPU visible — are you inside a GPU job/session?"; exit 1; }
fi
BUSY=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
[ "$BUSY" -gt 0 ] && log "WARNING: $BUSY other process(es) are using this GPU; runs will be slower."

NW=$(( $(nproc) / PAR )); [ "$NW" -gt 8 ] && NW=8; [ "$NW" -lt 2 ] && NW=2
log "Budget ${BUDGET} min, ${PAR} parallel jobs, ${NW} data workers each, ${EPOCHS} epochs per run"

log "Caching the DINOv2 model (needs internet once) ..."
python -c "
import os
from transformers import AutoModel
AutoModel.from_pretrained(os.environ.get('DFU_MODEL', 'facebook/dinov2-base'))" > logs/exp/model_cache.log 2>&1 \
  || { log "ERROR: cannot load the DINOv2 model (see logs/exp/model_cache.log)"; exit 1; }

# bf16 autocast: use it only if a 2-epoch debug run works
AMPF=""
if python -m src.train --debug --amp --run_name preflight --out_dir logs/exp/preflight \
     > logs/exp/preflight.log 2>&1; then
  AMPF="--amp"; log "bf16 autocast: OK (enabled)"
else
  log "bf16 autocast: failed in pre-flight — running in fp32 (TF32)"
fi
rm -f checkpoints/preflight_best.pt

# ── 1. near-duplicate groups (leak-free split) ────────────────────────────────
log "Computing near-duplicate groups ..."
python -m src.make_groups --auto --num_workers "$NW" --out groups.csv > logs/exp/make_groups.log 2>&1 \
  || { log "ERROR: make_groups failed (see logs/exp/make_groups.log)"; exit 1; }
grep -E "Using threshold|--auto|random split|group-aware split" logs/exp/make_groups.log | tee -a "$PROG"

# ── 2. Phase 1 (only if no checkpoint exists) ─────────────────────────────────
if [ -f "$LORA" ]; then
  log "Phase-1 checkpoint found: $LORA (skipping pre-training)"
else
  log "Phase 1: SimMIM pre-training, $PRE_EPOCHS epochs ..."
  python -m src.pretrain --epochs "$PRE_EPOCHS" --batch_size 32 --num_workers "$NW" $AMPF \
      > logs/exp/pretrain.log 2>&1
  [ -f checkpoints/dfu_pretrained_backbone.pt ] \
    || { log "ERROR: pre-training failed (see logs/exp/pretrain.log)"; exit 1; }
  LORA=checkpoints/dfu_pretrained_backbone.pt
  log "Phase 1 done: $(grep 'Epoch' logs/exp/pretrain.log | tail -1)"
fi

# ── job scheduler ─────────────────────────────────────────────────────────────
COMMON=(--group_csv groups.csv --split_seed 0 --imbalance weights
        --epochs "$EPOCHS" --batch_size 32 --num_workers "$NW" $AMPF)
BASIC=()
REG=(--aug strong --label_smoothing 0.1)
PRE=(--pretrained_lora_path "$LORA")
SKIPPED=()

# launch NAME MULT [train args...]  (MULT scales the expected duration, e.g. students are bigger)
launch() {
  local name=$1 mult=$2; shift 2
  if [ -f "results/$name.json" ]; then
    log "KEEP  $name (result exists — delete results/$name.json to rerun)"; return
  fi
  while [ "$(jobs -rp | wc -l)" -ge "$PAR" ]; do wait -n; done
  if [ "$EST" -gt 0 ]; then
    local need=$(( EST * mult / 10 )) left=$(( BUDGET - RESERVE - $(elapsed) ))
    if [ "$need" -gt "$left" ]; then
      log "SKIP  $name (needs ~${need} min, ${left} min left)"; SKIPPED+=("$name"); return
    fi
  fi
  ( t0=$(date +%s)
    python -m src.train "$@" "${COMMON[@]}" --run_name "$name" > "logs/exp/$name.log" 2>&1
    echo "$(( $(date +%s) - t0 ))" > "logs/exp/$name.dur"
    [ -f "results/$name.json" ] && echo "$name finished" >> "$PROG" || echo "$name FAILED" >> "$PROG"
  ) &
  log "START $name"
}

# Wait for the running jobs, then set the per-job time estimate (minutes) from the
# longest job measured so far. launch() multiplies it by MULT/10.
end_stage() {
  wait
  local max=0 f d
  for f in logs/exp/*.dur; do
    [ -f "$f" ] || continue
    d=$(cat "$f"); [ "$d" -gt "$max" ] && max=$d
  done
  EST=$(( (max + 59) / 60 )); [ "$EST" -lt 1 ] && EST=1
  log "stage done — longest job so far: $(( max / 60 )) min"
}

# ── 3. experiment stages (priority order) ─────────────────────────────────────
# Stage 1 — one replicate of every condition (the core comparison)
launch g_scratch_basic_s0 10 --seed 0 "${BASIC[@]}"
launch g_pre_basic_s0     10 --seed 0 "${PRE[@]}" "${BASIC[@]}"
launch g_scratch_reg_s0   10 --seed 0 "${REG[@]}"
launch g_pre_reg_s0       10 --seed 0 "${PRE[@]}" "${REG[@]}"
end_stage

# Stage 2 — more teachers (scratch + regularisation), then pseudo-labels from their ensemble
for s in 1 2; do
  [ "$s" -lt "$NSEEDS" ] && launch "g_scratch_reg_s$s" 10 --seed "$s" "${REG[@]}"
done
end_stage

TEACHERS=( $(ls checkpoints/g_scratch_reg_s*_best.pt 2>/dev/null) )
PSEUDO_OK=0
if [ "${#TEACHERS[@]}" -ge 1 ]; then
  log "Pseudo-labelling the unlabeled images with ${#TEACHERS[@]} teacher(s) + flip TTA ..."
  if python -m src.predict --checkpoints "${TEACHERS[@]}" --unlabeled_train --tta \
        --num_workers "$NW" --out pseudo_g.csv > logs/exp/pseudo_predict.log 2>&1; then
    PSEUDO_OK=1
    grep -E "Predicted class counts" logs/exp/pseudo_predict.log | tee -a "$PROG"
  else
    log "pseudo-label prediction failed (see logs/exp/pseudo_predict.log)"
  fi
fi

# Stage 3 — student + second replicate of the remaining conditions
if [ "$PSEUDO_OK" -eq 1 ]; then
  launch g_student_s0 14 --seed 0 "${REG[@]}" --pseudo_csv pseudo_g_probs.csv \
      --pseudo_thresh "$THRESH" --pseudo_max_per_class "$CAP"
fi
if [ "$NSEEDS" -gt 1 ]; then
  launch g_scratch_basic_s1 10 --seed 1 "${BASIC[@]}"
  launch g_pre_basic_s1     10 --seed 1 "${PRE[@]}" "${BASIC[@]}"
  launch g_pre_reg_s1       10 --seed 1 "${PRE[@]}" "${REG[@]}"
fi
end_stage

# Stage 4 — remaining replicates
if [ "$PSEUDO_OK" -eq 1 ]; then
  for s in 1 2; do
    [ "$s" -lt "$NSEEDS" ] && launch "g_student_s$s" 14 --seed "$s" "${REG[@]}" \
        --pseudo_csv pseudo_g_probs.csv --pseudo_thresh "$THRESH" --pseudo_max_per_class "$CAP"
  done
fi
if [ "$NSEEDS" -gt 2 ]; then
  launch g_scratch_basic_s2 10 --seed 2 "${BASIC[@]}"
  launch g_pre_basic_s2     10 --seed 2 "${PRE[@]}" "${BASIC[@]}"
  launch g_pre_reg_s2       10 --seed 2 "${PRE[@]}" "${REG[@]}"
fi
end_stage

# ── 4. ensemble / TTA evaluation on the shared test split ─────────────────────
ens() {   # ens TAG GLOB
  local tag=$1; shift
  local ck=( $(ls $1 2>/dev/null) )
  [ "${#ck[@]}" -ge 2 ] || return
  for mode in "" "--tta"; do
    local suffix=${mode:+_tta}
    python -m src.predict --checkpoints "${ck[@]}" --split test --split_seed 0 \
        --group_csv groups.csv --num_workers "$NW" $mode > "logs/exp/ens_${tag}${suffix}.txt" 2>&1
    log "ensemble ${tag}${mode:+ + TTA} (${#ck[@]} models): $(grep -E '^(ensemble|single)' "logs/exp/ens_${tag}${suffix}.txt" | head -1)"
  done
}
log "Ensemble / TTA evaluation ..."
ens scratch_reg  "checkpoints/g_scratch_reg_s*_best.pt"
ens pre_reg      "checkpoints/g_pre_reg_s*_best.pt"
ens scratch_basic "checkpoints/g_scratch_basic_s*_best.pt"
ens student      "checkpoints/g_student_s*_best.pt"

# ── 5. summary ────────────────────────────────────────────────────────────────
log "Summary (group-aware split; one row per condition):"
python -m src.summarize --no_plot 2>&1 | grep -v Warning | tee logs/exp/summary.txt

{
  echo "Experiment run finished after $(elapsed) min."
  echo "Skipped jobs: ${SKIPPED[*]:-none}"
  echo "Failed jobs : $(grep -c FAILED "$PROG")  (see logs/exp/<name>.log)"
  echo "Pseudo-labels: $(grep -h 'Pseudo-labels' logs/exp/g_student_s0.log 2>/dev/null | head -1)"
  echo "Groups: $(grep -h 'Using threshold' logs/exp/make_groups.log)"
} | tee -a "$PROG"
log "Done. Key files: results/summary.csv  results/*.json  logs/exp/summary.txt  logs/exp/ens_*.txt"
