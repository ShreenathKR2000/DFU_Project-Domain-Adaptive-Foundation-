# Domain-Adaptive Foundation Models for Fine-Grained DFU Classification

Diabetic Foot Ulcer (DFU) classification using a frozen **DINOv2 ViT-B/14**
backbone with **LoRA adapters** and **Supervised Contrastive Learning**.

## Problem

The DFUC2021 dataset exhibits two key challenges:

1. **Extreme class imbalance** — the four classes (*none*, *infection*,
   *ischaemia*, *both*) are heavily skewed.
2. **Partial labeling** — only 5,955 of 9,949 images carry ground-truth
   labels; the remaining 3,994 are unlabeled.

## Approach

A **two-phase** training pipeline, both fitting within a **GTX 1650 (4 GB VRAM)**:

### Phase 1 — Domain-Adaptive Pre-training (unsupervised)

| Component | Detail |
|---|---|
| Data | 3,994 **unlabeled** DFU images |
| Task | Masked Image Modeling (SimMIM-style) |
| Masking | 60 % of 14×14 patches zeroed at the pixel level |
| Decoder | LayerNorm + Linear (768 → 588) — reconstructs RGB patches |
| Loss | L1 between predicted and original pixels (masked patches only) |
| Trainable | LoRA adapters (Q/V) + decoder head |

### Phase 2 — Supervised Fine-tuning

| Component | Detail |
|---|---|
| Data | 5,955 **labeled** DFU images |
| Backbone | `facebook/dinov2-base` (frozen) + pre-trained LoRA adapters |
| Loss | 0.5 × CrossEntropy + 0.5 × SupConLoss (`pytorch-metric-learning`) |
| Head | LayerNorm + Linear (768 → 4) |
| Trainable | LoRA adapters + classification head |

Only **0.68 % of parameters** are ever trainable, keeping GPU memory low.

## Repository Structure

```
DFU_Project/
├── Data/                    # DFUC2021_train/ (not tracked by git)
│   └── DFUC2021_train/
│       ├── images/
│       └── train.csv
├── Notebooks/
│   └── 01_data_explore.ipynb
├── src/
│   ├── __init__.py
│   ├── dataset.py           # DFUDataset + transforms
│   ├── model.py             # DFUDinoLoRA (DINOv2 + LoRA + head)
│   ├── pretrain_model.py    # DFUDinoLoRAForMIM (MIM wrapper + decoder)
│   ├── pretrain.py          # Phase 1: unsupervised MIM pre-training
│   ├── train.py             # Phase 2: supervised classification + evaluation
│   ├── make_groups.py       # near-duplicate clustering for leak-free splits
│   ├── summarize.py         # scratch vs. pretrained summary / CSV / plot
│   └── predict.py           # TTA + ensemble inference, submission CSV
├── slurm/                   # sbatch scripts
├── run_lowlabel.sh          # low-label experiment (one seed)
├── run_pseudo.sh            # pseudo-labelling pipeline (one split seed)
├── .gitignore
├── CHANGELOG.md
└── README.md
```

## Setup

### 1. Create a Conda environment

```bash
conda create -n dfu python=3.10 -y
conda activate dfu
```

### 2. Install PyTorch (CUDA 11.8 — adjust for your driver)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 3. Install project dependencies

```bash
pip install transformers peft pytorch-metric-learning pandas \
            matplotlib Pillow scikit-learn
```

### 4. Place the dataset

Download or symlink the DFUC2021 training data so that the project tree looks
like:

```
DFU_Project/Data/DFUC2021_train/images/*.jpg
DFU_Project/Data/DFUC2021_train/train.csv
```

## Usage

### Phase 1 — Domain-Adaptive Pre-training

Debug run (20 unlabeled images, batch_size=2, 2 epochs):

```bash
cd DFU_Project
python -m src.pretrain --debug
```

Full pre-training run:

```bash
python -m src.pretrain
```

Override defaults:

```bash
python -m src.pretrain --epochs 30 --batch_size 4 --lr 5e-5 --mask_ratio 0.5
```

Outputs:
- `checkpoints/dfu_pretrain_mim.pt` — full model (backbone + decoder), for
  resuming pre-training.
- `checkpoints/dfu_pretrained_backbone.pt` — LoRA weights only, for Phase 2.

### Phase 2 — Supervised Fine-tuning

Debug run (20 labeled images, batch_size=2, 2 epochs):

```bash
python -m src.train --debug
```

Full training run with Phase-1 pre-trained LoRA weights (recommended):

```bash
python -m src.train \
    --pretrained_lora_path checkpoints/dfu_pretrained_backbone.pt
```

Full training run from scratch (no Phase-1 weights):

```bash
python -m src.train
```

Override defaults:

```bash
python -m src.train --epochs 30 --batch_size 4 --lr 5e-5 \
    --pretrained_lora_path checkpoints/dfu_pretrained_backbone.pt
```

### Evaluating the hypothesis (scratch vs. Phase-1)

`src.train` splits the 5,955 labeled images (stratified, 70/15/15) into
train / val / test. Each epoch it reports validation accuracy, **macro-F1** and
per-class recall; the best epoch (by validation macro-F1) is evaluated once on
the held-out test split. Results (metrics, confusion matrix, per-epoch history)
are written to `results/<run_name>.json`; the best LoRA + head weights to
`checkpoints/<run_name>_best.pt`.

Runs that share `--seed` share the same split, so the two arms are paired:

```bash
python -m src.train --seed 0 --imbalance weights --run_name scratch_seed0
python -m src.train --seed 0 --imbalance weights --run_name pretrained_seed0 \
    --pretrained_lora_path checkpoints/dfu_pretrained_backbone.pt
python -m src.summarize        # mean ± std per arm + paired per-seed differences
```

Other flags: `--imbalance {none,weights,sampler}` (inverse-frequency CE weights
or class-balanced sampling), `--train_frac` (low-label experiments, see below),
`--num_workers`, `--val_frac`, `--test_frac`.
Phase 1 also accepts `--seed` and `--num_workers`.

Notes on interpreting results:

- DFUC2021 is heavily imbalanced (`ischaemia` has only a few hundred labeled
  images), so accuracy alone is misleading — use macro-F1 and per-class recall.
- Only a few dozen `ischaemia` images land in each val/test split, so single
  runs are noisy; compare means over several seeds.
- The dataset has no patient IDs, so near-duplicate images of one patient can
  fall on both sides of a random split and inflate scores for *both* arms.
- Phase 1 only uses unlabeled images, so the test split is never seen during
  pre-training.

## Results

### Full-label comparison (DFUC2021 train split, 3 seeds)

Stratified 70/15/15 split (4,167 / 894 / 894 images), batch 32, 30 epochs,
inverse-frequency CE weights, Phase 1 = 100 epochs of SimMIM on the 3,994
unlabeled images (reconstruction loss 0.39 → 0.096, still slowly decreasing).
Test numbers at the epoch with the best validation macro-F1; seeds 0/1/2 share
splits across arms.

| | scratch | Phase-1 pretrained |
|---|---|---|
| Test macro-F1 (per seed) | 0.898, 0.866, 0.877 | 0.859, 0.874, 0.876 |
| **Test macro-F1 (mean ± std)** | **0.880 ± 0.017** | **0.870 ± 0.010** |
| Test accuracy (mean) | 0.877 | 0.854 |
| Recall none / infection | 0.875 / 0.873 | 0.839 / 0.847 |
| Recall ischaemia / both | 0.833 / 0.917 | 0.863 / 0.946 |

Paired difference (pretrained − scratch): −0.040, +0.009, −0.000 (mean −0.011).

**Finding.** With ~4,100 labeled training images, Phase-1 domain-adaptive MIM
pre-training gave **no measurable benefit** for classification: the mean
difference is smaller than the seed-to-seed spread of the baseline (±0.017), so
the two arms are statistically indistinguishable. The pretrained arm was
slightly better on the rare classes (ischaemia, both; +0.03 recall each, but
ischaemia has only 34 test images, i.e. ±1 image ≈ ±0.03) and slightly worse on
the two large classes. The dominant error for *both* arms is `none` ↔
`infection` confusion (~100 of ~130 errors per run). Pretrained runs also
started lower (epoch-1 val macro-F1 ≈ 0.55–0.63 vs 0.65–0.68), suggesting Phase 1
moved the adapters away from DINOv2's already strong semantic features.

Possible reasons (untested): pixel reconstruction emphasises low-level
colour/texture rather than semantics; 3,994 images and rank-16 Q/V LoRA is a
small adaptation; and with thousands of labels a strong frozen backbone leaves
little headroom for pre-training to help.

*Earlier versions of this README reported a "+5 pp" debug-run gain for Phase 1;
that was noise on 20 images (and with older peft versions the Phase-1 weights
were silently not applied — fixed in v0.4.1).*

### Low-label experiment (does Phase 1 help when labels are scarce?)

Domain-adaptive pre-training is usually motivated by label scarcity, so the
natural follow-up is to train both arms on only a fraction of the training
labels (`--train_frac`, class-stratified, identical subset for both arms of a
seed; validation and test splits stay full-size):

```bash
./run_lowlabel.sh 0        # one seed: 10 / 25 / 50 % × {scratch, pretrained}
python -m src.summarize    # table per fraction, results/summary.csv, results/label_efficiency.png
```

Epochs are raised for small fractions (100 / 60 / 40 at 10 / 25 / 50 %) so each
fraction gets a comparable number of gradient steps. `summarize` reports macro-F1
at the best-validation epoch **and** for the last-epoch model; the latter needs no
validation labels and is the fairer number in a low-label claim (the full-size
validation split is otherwise a source of label information).
Results for this experiment: *to be added*.

How to read it: if pretrained beats scratch at 10 % / 25 % labels and the gap
closes at 100 %, Phase 1 is a label-efficiency gain (claim: "reduces labeling
needs by X×"). If the curves overlap everywhere, the honest conclusion is that
pixel-MIM adds nothing over DINOv2 for this task.

### Ideas for further study

- More seeds (5+) or k-fold CV; report paired differences with confidence intervals.
- A different Phase-1 objective: reconstruct frozen-DINOv2 patch *features*
  (feature distillation) instead of pixels; lower mask ratio; more LoRA capacity
  (rank, key/MLP layers, last blocks fully trained); continue Phase 1 beyond 100 epochs.
- Self-training with the 3,994 unlabeled images (pseudo-labels from the trained
  classifier) instead of / in addition to MIM.
- Class-imbalance options already in `train.py` (`--imbalance sampler`), and
  ordinal / hierarchical modelling of (none, infection, ischaemia, both) as two
  binary flags (infection, ischaemia).
- Patient-level splitting if patient IDs can be recovered (DFUC2021 provides none;
  near-duplicate images across splits can inflate every number above).
- Error analysis of the `none` ↔ `infection` confusion (Grad-CAM / attention
  maps, clinician review).

### Relation to the DFUC challenges

The scores above are on a random internal split of the public DFUC2021 *training*
images and are **not comparable** to challenge leaderboards (hidden test sets,
different distribution, possible near-duplicate leakage here). The DFUC2021
task was 4-class classification (official metric: macro-F1); **DFUC2022**
(`dfuc2022.grand-challenge.org`) is, as far as I recall, a *segmentation* task
scored by Dice — please verify the task, metric, data and external-data rules on
the challenge page before investing. If it is segmentation, this classification
pipeline cannot be submitted as is. Suggested adaptations:

1. **Dense head on the same backbone**: DINOv2 patch tokens (16×16 at 224 px)
   → light convolutional/DPT-style decoder with upsampling; Dice + BCE (or
   Lovász/boundary) loss; report Dice/IoU.
2. **Higher resolution** (e.g. 448–518 px, multiples of 14) and multi-scale /
   flip test-time augmentation; ulcers are small structures.
3. **Phase 1 fits dense tasks better**: MIM trains *patch-level* features, which
   segmentation uses directly, unlike global classification; it may show gains
   here that it did not for classification — test with `--train_frac`-style
   label fractions on the segmentation data.
4. **Stronger recipe**: LoRA on all linear layers with higher rank (or unfreeze
   the last blocks), `dinov2-large`, SWA/EMA, heavy colour/scale augmentation,
   5-fold ensembling, and pseudo-labelling of unlabeled images if the rules allow.
5. **Evaluation discipline**: choose hyper-parameters on validation only,
   check the rules on external data/pre-trained weights, and keep submissions
   reproducible (fixed seeds, logged configs — already saved in `results/*.json`).

### What the live leaderboard says about the evaluation

On the DFUC2021 open leaderboard the top entries have macro-F1 ≈ 0.65 and ranks
~50 ≈ 0.55, while this project's internal split gives ≈ 0.88. A gap that large
means the internal split is **optimistic**: DFUC2021 has no patient IDs, so
near-identical images of one ulcer can land in both train and test, and the
hidden test set is also shifted (different patients / sessions). Consequences:

- Epoch selection, hyper-parameters and the scratch-vs-Phase-1 comparison made on
  the random split reward memorisation (train accuracy reaches 99%), so they say
  little about leaderboard performance — and the "no benefit from Phase 1"
  finding should be re-checked on a leak-free split, since domain adaptation is
  precisely about shift.
- The fix is a **group-aware split**: cluster near-duplicates and keep each
  cluster in one split.

```bash
python -m src.make_groups --threshold 0.90        # writes groups.csv + leak diagnostic
python -m src.train --group_csv groups.csv --seed 0 --imbalance weights ...
```

`make_groups` prints how many groups each threshold gives and the share of test
images with a near-duplicate in train under the random vs. group-aware split.
Choose a threshold that merges obvious duplicates without forming one giant
cluster, then use `--group_csv` (also accepted by `src.predict`) for every
validation decision. A good sign that the evaluation is fixed: group-split
macro-F1 drops towards the leaderboard range (~0.55–0.65) and tracks your
leaderboard submissions.

Regularisation aimed at shift (compare on the group split): `--aug strong`
(random resized crops, rotation, stronger colour jitter, blur), `--label_smoothing
0.1`, fewer epochs / earlier stopping chosen on the group-validation split,
then ensemble + TTA (`src.predict`). Further ideas: pseudo-labelling the 3,994
unlabeled training images, a bigger backbone (`dinov2-large`), and — only if the
challenge rules allow it — Phase-1 adaptation on the unlabeled test images.

### Pseudo-labelling the unlabeled images

The 3,994 unlabeled DFUC2021 training images are used so far only for Phase 1.
Pseudo-labelling (noisy student) uses them for *supervised* training too:

1. train teachers on the labeled **train split** (same `--split_seed`, different
   `--seed`s — the teachers never see the val/test labels of that split);
2. predict the unlabeled images with the teacher ensemble + flip TTA
   (`python -m src.predict --unlabeled_train ...` writes `*_probs.csv`);
3. train a student on train + confident pseudo-labels (`--pseudo_csv`,
   `--pseudo_thresh`, `--pseudo_max_per_class`) with strong augmentation.

```bash
GROUP_CSV=groups.csv ./run_pseudo.sh 0     # also 1, 2 for a paired comparison
python -m src.summarize
```

`run_pseudo.sh` trains the student **and** a baseline with identical settings but
no pseudo-labels; `summarize` shows them as separate rows (`[group+strong-aug+pseudo0.9]`
vs `[group+strong-aug]`). The student is useful only if it beats that baseline over
several split seeds.

Notes: confident predictions are dominated by the big classes, so watch the
`Pseudo-labels: +N images {...}` line and use `CAP=<n>` (or a lower `THRESH`) to
keep ischaemia/both from being drowned out; pseudo-labels can reinforce teacher
mistakes (confirmation bias), so try a higher threshold if the student gets worse;
images in the unlabeled pool are not covered by `groups.csv`, so near-duplicates
of held-out images can enter training (with pseudo-labels, never true ones).
`--pseudo_img_dir` accepts other image folders (only if the challenge rules allow).
Pseudo-labelling can be repeated: use the student ensemble as the next teacher.

### Toward a leaderboard submission

**Which data can do what.** DFUC2021 (used so far) is labeled for 4-class
classification. DFUC2020, as I recall, provides ulcer *bounding boxes* for
detection, not class labels, so it cannot train the 4-class head. It can be used
(a) as extra *unlabeled* images for Phase 1, (b) for a detection task if that is
the target, or (c) cropped to the boxes to get ulcer-centred patches similar to
DFUC2021 (needs the annotation format). Check licences and each challenge's rules
on external data and pre-trained weights first; if the target is segmentation
(DFUC2022), masks are required and neither dataset supplies them.

**Recipe (classification).** Decide everything on held-out data, then retrain on
all labels and submit an ensemble:

```bash
# 1. Extra unlabeled images for Phase 1 (optional; any folder of jpg/png)
python -m src.pretrain --epochs 100 --batch_size 32 --num_workers 8 \
    --extra_img_dirs /path/to/DFUC2020/images

# 2. Pick settings on the held-out split (resolution, LoRA rank, epochs, ...).
#    Vary --seed, keep --split_seed fixed so the models share one test split.
for S in 10 11 12; do
  python -m src.train --split_seed 0 --seed $S --epochs 30 --batch_size 32 \
      --imbalance weights --num_workers 8 --img_size 224 --run_name m$S
done

# 3. Measure what TTA and ensembling buy on that test split
python -m src.predict --checkpoints checkpoints/m10_best.pt checkpoints/m11_best.pt \
    checkpoints/m12_best.pt --split test --split_seed 0 --tta

# 4. Final models on ALL labeled images (fixed epochs chosen in step 2)
for S in 1 2 3 4 5; do
  python -m src.train --final --seed $S --epochs 30 --batch_size 32 \
      --imbalance weights --num_workers 8 --run_name final$S
done

# 5. Predict the challenge images (one-hot CSV + probabilities)
python -m src.predict --checkpoints checkpoints/final*_final.pt \
    --img_dir /path/to/challenge_images --tta --out predictions.csv
```

Knobs worth testing in step 2 (one at a time, compare over seeds): `--img_size 448`
(roughly 5x slower; small ulcers may benefit), `--lora_rank 32`, `--imbalance
sampler`, Phase 1 with `--extra_img_dirs`, and longer Phase 1. Expect gains from
these to be modest relative to seed noise (±0.02 macro-F1 here); the ensemble +
TTA step is the most reliable improvement. The largest unknown is distribution
shift between the public training images and the hidden test set, which no
internal split can measure.

## Initial Results (Debug Runs)

### Phase 1 — MIM Pre-training (debug)

```
Device     : cuda
Debug      : True
Epochs     : 2
Batch      : 2
Mask ratio : 0.6
Loss       : l1
Unlabeled imgs : 3,994 (debug subset: 20)

trainable params: 589,824 || all params: 87,170,304 || trainable%: 0.6766
decoder params : 453,708

Epoch 1/2  loss=0.757032  (3.0 s)
Epoch 2/2  loss=0.659234  (1.8 s)

Full pretrain model  → checkpoints/dfu_pretrain_mim.pt
Backbone LoRA weights → checkpoints/dfu_pretrained_backbone.pt
```

**Observations** — Reconstruction loss drops from 0.757 → 0.659 (−13 %) across
2 epochs on just 20 images, confirming the LoRA adapters + decoder learn to
reconstruct masked patches.

### Phase 2 — Supervised Fine-tuning (debug)

**From scratch** (no Phase-1 weights):

```
LoRA   : scratch
Epoch 1/2  loss=0.5419  ce=1.0838  con=0.0000  acc=0.550  (2.3 s)
Epoch 2/2  loss=0.4112  ce=0.8223  con=0.0000  acc=0.700  (1.9 s)
```

**With Phase-1 pre-trained LoRA weights** (`--pretrained_lora_path`):

```
LoRA   : checkpoints/dfu_pretrained_backbone.pt
Loaded Phase-1 LoRA weights from checkpoints/dfu_pretrained_backbone.pt
Epoch 1/2  loss=0.6986  ce=1.3972  con=0.0000  acc=0.400  (6.8 s)
Epoch 2/2  loss=0.4085  ce=0.8170  con=0.0000  acc=0.750  (1.9 s)
```

| Init | Epoch 2 acc | Epoch 2 loss |
|---|---|---|
| Scratch | 70.0 % | 0.4112 |
| Phase-1 LoRA | **75.0 %** | **0.4085** |

**Observations** — *Superseded:* with older peft versions the Phase-1 weights were
silently not applied in these debug runs (fixed in v0.4.1), so this table only
shows run-to-run noise on 20 images. See the evaluation protocol above.

## Cluster / HPC Deployment (Slurm)

Create the environment once on the cluster (venv or conda), cache the model
(`python -c "from transformers import AutoModel; AutoModel.from_pretrained('facebook/dinov2-base')"`),
then submit from the project root (the scripts expect `.venv/` there; edit the
`source` line for conda):

```bash
mkdir -p logs
sbatch --partition=<gpu-partition> slurm/pretrain.sbatch      # Phase 1 (EPOCHS=100 by default)
sbatch --partition=<gpu-partition> slurm/train_array.sbatch   # 3 seeds x {scratch, pretrained}
python -m src.summarize
```

Monitor with `squeue -u $USER` and `tail -f logs/*.out`. On an NVIDIA L40S
(46 GB) batch size 32 uses about 5 GB; raise `--batch_size` if desired.

## License

This project is for academic and research purposes.
