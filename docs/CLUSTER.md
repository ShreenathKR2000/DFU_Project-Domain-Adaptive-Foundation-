# Running on an HPC cluster

Tested on a Slurm cluster with Open OnDemand and an NVIDIA L40S (46 GB).
Everything here also works on a single workstation GPU.

## 1. Concepts

- **Login node**: edit files and submit jobs; never train here.
- **Compute node / job**: where the GPU is. An OnDemand *VS Code* session is itself a
  Slurm job — check `nvidia-smi` in its terminal to see whether it has a GPU.
- **Wall time**: a session or job is killed when its time limit ends, together with
  anything running inside it. Long runs should use `sbatch`, or at least `nohup`.

## 2. One-time setup

```bash
cd /scratch/<user>/DFU_Project          # put Data/ and checkpoints on scratch; copy results out later
python3 -m venv .venv && source .venv/bin/activate     # no module/conda needed
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118   # cu118 runs fine on newer drivers
pip install -r requirements.txt
python -c "from transformers import AutoModel; AutoModel.from_pretrained('facebook/dinov2-base')"  # cache the model (needs internet)
python -c "import torch; print(torch.cuda.is_available())"                            # True on a GPU node
```

If `module` is missing in the VS Code terminal, use `bash -l` or
`source /etc/profile.d/modules.sh`; otherwise the venv above is enough. Activate the
venv in **every** new terminal (`source .venv/bin/activate`; the prompt shows `(.venv)`).
Always run from the project root (`python -m src.train`, not `python src/train.py`).

## 3. Running

### Interactive (inside a GPU VS Code session)

```bash
mkdir -p logs
nohup sh -c "python -m src.pretrain --epochs 100 --batch_size 32 --num_workers 8 && \
python -m src.train --epochs 30 --batch_size 32 --imbalance weights --num_workers 8 \
  --pretrained_lora_path checkpoints/dfu_pretrained_backbone.pt" > logs/run.log 2>&1 &
tail -f logs/run.log          # Ctrl+C stops tailing, not the job
watch -n 2 nvidia-smi         # GPU-Util near 100 % = GPU is the bottleneck (good)
```

### Batch (Slurm)

```bash
mkdir -p logs
sbatch --partition=<gpu-partition> scripts/pretrain.sbatch          # EPOCHS=100 env to change
sbatch --partition=<gpu-partition> scripts/train_array.sbatch       # 3 seeds x {scratch, pretrained}
squeue -u $USER                                                     # PD = waiting, R = running
scancel <jobid>
```

Find the partition name with `sinfo -s`. Open OnDemand's *Job Composer* can submit the same
scripts; point its job script at the project directory (`cd` there first — the Data/ folder
lives in the project, not in the job folder).

## 4. Batch size and throughput (L40S)

`--batch_size 32` uses ~5 GB and already drives the GPU to ~98 % utilisation, so a bigger batch
does not speed things up; it only reduces the number of updates. Use spare memory for
parallel runs (e.g. one process per seed) rather than larger batches. Keep the batch size
identical across arms of a comparison.

## 5. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `ModuleNotFoundError: torch` | venv not active (`source .venv/bin/activate`) |
| `No module named 'src'` | run from the project root with `python -m src.<module>` |
| `ImportError ... circular import` or unknown CLI flags | a stale/modified file copied to the cluster — re-copy `src/` from the repo and `diff -q` each file |
| `Phase-1 LoRA weights ... were not applied` | fixed in v0.4.1 (older peft silently ignored the checkpoint); update `src/model.py` |
| Model download fails on a compute node | no internet there — cache `facebook/dinov2-base` on the login node first |
| CUDA out of memory | lower `--batch_size` / `--img_size` |
| Job killed at the time limit | request more time, or `nohup`/`sbatch` instead of an interactive session |
| Scratch files disappear | scratch is often purged — copy `checkpoints/` and `results/` somewhere permanent |
