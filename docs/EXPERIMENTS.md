# Experiments

What has been run, what it showed, and how to run the rest. Commands assume the project
root with the venv active (see [CLUSTER.md](CLUSTER.md)); results land in `results/*.json`
and are compared with `python -m src.summarize`.

## 1. Evaluation protocol

- **Split:** the 5,955 labeled images are split 70/15/15 (train/val/test), class-stratified,
  deterministic in `--split_seed` (default: `--seed`). Runs sharing a seed share the split,
  so two arms can be compared *paired*.
- **Group-aware split:** with `--group_csv groups.csv`, near-duplicate images stay in the
  same split (see §3). This is the protocol to trust for any model-selection decision.
- **Model selection:** the epoch with the best **validation macro-F1** is evaluated once on
  the test split. The last-epoch model is also evaluated (needs no validation labels).
- **Reported metrics** (same as the DFUC2021 leaderboard): macro-F1, per-class F1
  (`none` = *Control*, infection, ischaemia, both), macro/micro AUC, accuracy, confusion matrix.
- **Phase 1 uses only unlabeled images**, so the test split is never seen during pre-training.
- **Seeds:** ≥ 3 per condition; judge differences against the seed-to-seed spread, not single runs.

## 2. Full-label comparison — done (random split)

4,167 / 894 / 894 images, batch 32, 30 epochs, inverse-frequency CE weights;
Phase 1 = 100 epochs SimMIM on the 3,994 unlabeled images (loss 0.39 → 0.096, still falling).

| Test metric (mean of 3 seeds) | scratch | Phase-1 pretrained |
|---|---|---|
| **Macro-F1** | **0.880 ± 0.017** | **0.870 ± 0.010** |
| per seed | 0.898, 0.866, 0.877 | 0.859, 0.874, 0.876 |
| F1 none / infection | 0.879 / 0.868 | 0.849 / 0.843 |
| F1 ischaemia / both | 0.860 / 0.914 | 0.875 / 0.912 |
| Accuracy | 0.877 | 0.854 |

Paired difference (pretrained − scratch): −0.040, +0.009, −0.000 (mean −0.011).

**Finding.** No measurable benefit from Phase 1 at full labels: the mean difference is smaller
than the seed spread of the baseline. The dominant error for both arms is `none` ↔
`infection` confusion (~100 of ~130 errors per run). Pretrained runs also *started* lower
(epoch-1 val macro-F1 ≈ 0.55–0.63 vs 0.65–0.68), suggesting pixel-reconstruction MIM moves
the adapters away from DINOv2's already semantic features.

**Caveat — these numbers are optimistic.** The same model family scores ≈ 0.55–0.66 on the
hidden leaderboard test set; a random split puts near-duplicates of one ulcer in train and test
(train accuracy reaches 99 %). The conclusion about Phase 1 must be re-checked on the
group-aware split (§3), because domain adaptation is about distribution shift.

*(Early versions reported a "+5 pp" debug-run gain for Phase 1. That was noise on 20 images —
and with older peft versions the Phase-1 weights were silently not applied; fixed in v0.4.1.)*

## 3. Group-aware (leak-free) evaluation — next

```bash
python -m src.make_groups --threshold 0.90         # writes groups.csv, prints leak diagnostic
for ARM in scratch pretrained; do
  EXTRA=""; [ $ARM = pretrained ] && EXTRA="--pretrained_lora_path checkpoints/dfu_pretrained_backbone.pt"
  for S in 0 1 2; do
    python -m src.train --group_csv groups.csv --seed $S --epochs 30 --batch_size 32 \
        --imbalance weights --num_workers 8 --run_name g_${ARM}_seed$S $EXTRA
  done
done
python -m src.summarize
```

`make_groups` embeds images with frozen DINOv2, links pairs with cosine ≥ threshold and prints
the share of test images that have a near-duplicate in train under the random vs. group-aware
split. Pick a threshold that merges obvious duplicates without one giant cluster.

**What to look for:** group-split macro-F1 well below 0.88 (hopefully in the 0.55–0.70 range);
then repeat with `--aug strong --label_smoothing 0.1` to test regularisation for shift. Only then
re-assess scratch vs. Phase 1. `summarize` keeps every variant in its own row
(`scratch [group+strong-aug+ls0.1]`, …), so nothing is averaged across experiments.

## 4. Low-label study (does Phase 1 help when labels are scarce?) — pending

```bash
for S in 0 1 2; do nohup ./scripts/run_lowlabel.sh $S > logs/lowlabel_$S.log 2>&1 & done
python -m src.summarize            # one table per label fraction + results/label_efficiency.png
```

Trains scratch vs. pretrained on 10 / 25 / 50 % of the training labels (`--train_frac`,
class-stratified, identical subset for both arms; epochs 100 / 60 / 40 so each fraction gets a
comparable number of steps; val/test stay full-size). Pass `--group_csv` via editing the script
if you want it leak-free.

**Reading it:** if pretrained beats scratch at 10–25 % and the gap closes at 100 %, Phase 1 is a
label-efficiency gain ("reduces labelling needs by ×N"). If the curves overlap everywhere,
pixel-MIM adds nothing over DINOv2 for this task. At 10 % ischaemia has ~16 training images, so
expect large spread — judge by paired differences over the three seeds. Also compare the
last-epoch column, which uses no validation labels.

## 5. Pseudo-labelling (noisy student) — pending

```bash
GROUP_CSV=groups.csv CAP=300 ./scripts/run_pseudo.sh 0      # also 1, 2
python -m src.summarize
```

Per split seed: 3 teachers (same split, different seeds) → ensemble + flip-TTA predictions on the
3,994 unlabeled images → student trained on train + confident pseudo-labels (`--pseudo_thresh`,
`--pseudo_max_per_class`) with strong augmentation → a matched baseline without pseudo-labels.
Optional: `ARM=pretrained`, `THRESH=0.9`, `EPOCHS=30`.

**Reading it:** student rows `[group+strong-aug+pseudo0.9]` vs. baseline rows
`[group+strong-aug]`. Look at ischaemia/both F1 and macro AUC; a gain must exceed the seed spread.
If the student is worse, raise `THRESH` or lower `CAP` (confirmation bias). Watch the
`Pseudo-labels: +N images {...}` line — if rare classes are absent the teachers are too unsure
about them. Teachers never see the student's val/test labels (same split). Images in the
unlabeled pool are not covered by `groups.csv`, so near-duplicates of held-out images can enter
training with *pseudo* (never true) labels.

## 6. Further ideas

- More seeds / k-fold on the group split; report paired differences with confidence intervals.
- Different Phase-1 objective: reconstruct frozen-DINOv2 patch *features* instead of pixels; lower
  mask ratio; more LoRA capacity (`--lora_rank 32`, key/MLP layers); longer or larger-data Phase 1
  (`--extra_img_dirs`, e.g. DFUC2020 images if the rules allow).
- Resolution (`--img_size 448`, ~5× slower) and a bigger backbone (`dinov2-large`, needs code changes).
- Decision-threshold / logit-bias adjustment per class tuned on group-validation data (the leaderboard
  shows AUC ≈ 0.87–0.89 but macro-F1 ≈ 0.65: ranking is better than the hard decisions).
- Error analysis of the `none` ↔ `infection` confusion (Grad-CAM / attention maps, clinician review).

## 7. Limitations

No patient IDs (group clustering is a proxy); few ischaemia images (≈34 per split → noisy
per-class metrics); single dataset; internal splits cannot measure shift to the hidden test set.
