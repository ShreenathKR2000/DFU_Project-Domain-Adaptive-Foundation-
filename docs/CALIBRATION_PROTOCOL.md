# Calibration protocol for the second submission

Written **17 Oct 2026, before running it.** This file fixes what will be done and how the
decision will be made, so that the second leaderboard submission is not tuned on the leaderboard.

## Why a protocol

The first submission (macro-F1 0.588, rank 39; [LEADERBOARD.md §3a](LEADERBOARD.md)) showed good
ranking (macro-AUC 0.870, ischaemia AUC 0.952) but misplaced decision boundaries: ischaemia was
predicted for 545 images where the leaderboard metrics imply ≈ 225. Changing the decision rule
*because* of that score and resubmitting would use the leaderboard as a validation set. Repeated,
that overfits the hidden test set and the score stops being an honest estimate.

What is allowed to inform the second submission:

| Source | Used? |
|---|---|
| Training labels (class proportions 42.9 / 42.9 / 3.8 / 10.4 %) | yes |
| Group-aware validation split (labels) | yes — to fit `val_bias` |
| Group-aware test split (labels) | yes — **once**, to compare methods |
| Unlabeled challenge images / their predictions | yes — `prior` uses their predicted class shares, no labels |
| Leaderboard metrics of submission 1 (e.g. the implied ≈ 225 ischaemia images) | **no** — not used as targets or for tuning |

The leaderboard result did motivate *looking* at the decision rule. This is reported openly. The
warning sign was already visible before the first submission (E7: 9.5 % ischaemia predicted on the
challenge images vs 3.8 % in training).

## Methods compared ([`src/calibrate.py`](../src/calibrate.py))

| Method | What it does | Labels used |
|---|---|---|
| `none` | plain arg-max (submission 1) | — |
| `prior` | one log-bias per class so that arg-max predictions on the set being predicted reproduce the training class mix | none from the predicted set |
| `val_bias` | per-class log-biases tuned to maximise macro-F1 on the validation split, then frozen | validation split |

## Decision rule (fixed now)

**Rule: `noninferiority`.**

- Adopt `prior` if, on the internal test split, its mean macro-F1 gain over `none` is ≥ 0 and the
  one-sided 95 % lower bound of the gain (5th percentile of 2,000 paired bootstrap resamples)
  is > −0.02.
- Adopt `val_bias` instead only if it beats `none` in ≥ 90 % of resamples and has a larger mean
  gain than `prior`.
- Otherwise keep `none`, and **do not resubmit**.

*Why non-inferiority for `prior`:* it is meant to correct a shift between training and challenge
images. The internal split has no such shift (internally the models already predict ≈ 3 %
ischaemia), so internally `prior` can be shown not to hurt, not to help. Demanding a clear gain
there would test the wrong thing. The stricter alternative (`--rule superiority`: any method must
beat `none` in ≥ 90 % of resamples) is implemented; if you prefer it, change the rule **here,
before running**, and commit.

## Steps

1. Commit this file (with the rule) **before** running anything.
2. On the GPU machine with the group-split checkpoints and the training images:
   ```bash
   RULE=noninferiority ./scripts/run_calibration.sh
   ```
   This predicts the val and test splits with the 3 `g_pre_pseudo` models + flip TTA, compares the
   three methods, applies the rule and writes `results/calib/challenge_<method>_probs.csv`,
   `results/calib/calibration.json` and `results/calib/run.log`.
3. Commit `results/calib/` whatever the outcome (also if the chosen method is `none`).
4. If the chosen method is not `none`, submit `results/calib/challenge_<method>_probs.csv`
   **once**. Record the score in README, LEADERBOARD.md and EXPERIMENTS.md E7b as a second,
   post-hoc-motivated result next to submission 1.
5. No further leaderboard submissions for calibration.

## Known limitations

- The internal split cannot show the benefit under shift; it only guards against harm.
- `val_bias` is fitted on the group-split models (70 % of labels) but applied to the final models
  (all labels), which may be calibrated slightly differently.
- Rare classes are small internally (29 ischaemia / 61 both test images), so bootstrap intervals
  are wide.
- Training has 10.4 % "both"; matching it may slightly over-call "both" if the challenge set has
  fewer.
