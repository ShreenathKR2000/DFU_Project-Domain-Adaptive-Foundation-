"""
Aggregate ``results/*.json`` (written by ``src.train``) into a comparison of
runs on the held-out test split, grouped by the fraction of training labels
(``--train_frac``) and by experimental *variant*.

A variant is everything that changes the experiment other than the arm
(scratch vs. Phase-1 pretrained): group-aware split, strong augmentation, label
smoothing, pseudo-labels, resolution, LoRA rank. Each (arm, variant) gets its own
row, so runs from different experiments are never averaged together. Columns
mirror the DFUC2021 leaderboard (macro-F1, per-class F1, macro AUC).

Writes ``results/summary.csv`` (one row per run) and, when several label
fractions are present, ``results/label_efficiency.png``.

Usage
-----
    python -m src.summarize            # reads results/
    python -m src.summarize --dir results --no_plot
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os

import numpy as np

from src.dataset import LABEL_COLS

# Colour-blind-safe (Okabe-Ito) colours + distinct marker shapes; identity is
# never colour-only (legend + markers).
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X"]


def load_results(directory: str) -> list[dict]:
    paths = sorted(glob.glob(os.path.join(directory, "*.json")))
    results = [json.load(open(p)) for p in paths]
    # Debug runs (20 images) would pollute the statistics, and --final runs
    # (trained on everything, no test metrics) have nothing to compare.
    return [r for r in results
            if not r["config"].get("debug") and r.get("mode") != "final"]


def frac_of(r: dict) -> float:
    return float(r["config"].get("train_frac", 1.0))


def variant_of(r: dict) -> str:
    """Short tag for every setting that changes the experiment besides the arm."""
    c = r["config"]
    parts = []
    if c.get("group_csv"):
        parts.append("group")
    if c.get("aug", "basic") != "basic":
        parts.append(f"{c['aug']}-aug")
    if c.get("label_smoothing", 0.0):
        parts.append(f"ls{c['label_smoothing']:g}")
    if c.get("pseudo_csv"):
        parts.append(f"pseudo{c.get('pseudo_thresh', 0.9):g}")
    if c.get("img_size", 224) != 224:
        parts.append(f"{c['img_size']}px")
    if c.get("lora_rank", 16) != 16:
        parts.append(f"r{c['lora_rank']}")
    if c.get("imbalance", "weights") not in ("weights",):
        parts.append(f"imb-{c['imbalance']}")
    return "+".join(parts)


def label_of(r: dict) -> str:
    v = variant_of(r)
    return r["arm"] + (f" [{v}]" if v else "")


def fmt(values: list) -> str:
    arr = np.asarray([np.nan if v is None else v for v in values], dtype=float)
    if np.isnan(arr).any():
        return "n/a"
    std = arr.std(ddof=1) if len(arr) > 1 else float("nan")
    return f"{arr.mean():.3f}±{std:.3f}"


def last_f1(r: dict) -> float:
    """Test macro-F1 of the final-epoch model (NaN for runs made before it was logged)."""
    return r["test_last_epoch"]["macro_f1"] if "test_last_epoch" in r else float("nan")


def write_csv(results: list[dict], path: str) -> None:
    cols = ["run_name", "arm", "variant", "train_frac", "seed", "n_train", "best_epoch",
            "macro_f1", "last_epoch_macro_f1", "macro_auc", "micro_auc", "acc"]
    cols += [f"f1_{c}" for c in LABEL_COLS]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in sorted(results, key=lambda r: (frac_of(r), label_of(r), r["config"]["seed"])):
            t = r["test"]
            auc = lambda k: "" if t.get(k) is None else f"{t[k]:.4f}"
            w.writerow([
                r["run_name"], r["arm"], variant_of(r), frac_of(r), r["config"]["seed"],
                r["split_sizes"]["train"], r["best_epoch"],
                f"{t['macro_f1']:.4f}", f"{last_f1(r):.4f}", auc("macro_auc"), auc("micro_auc"),
                f"{t['accuracy']:.4f}",
                *[f"{t['per_class'][c]['f1']:.4f}" for c in LABEL_COLS],
            ])


def plot_label_efficiency(groups: dict, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = sorted({lab for g in groups.values() for lab in g})
    fracs = sorted(groups)
    fig, ax = plt.subplots(figsize=(6.6, 4.2), dpi=150)
    for i, lab in enumerate(labels):
        color, marker = PALETTE[i % len(PALETTE)], MARKERS[i % len(MARKERS)]
        xs, means, stds = [], [], []
        for fr in fracs:
            rs = groups[fr].get(lab, [])
            if not rs:
                continue
            vals = np.array([r["test"]["macro_f1"] for r in rs])
            ax.scatter([fr * 100] * len(vals), vals, s=14, color=color, alpha=0.35,
                       marker=marker, linewidths=0)
            xs.append(fr * 100); means.append(vals.mean())
            stds.append(vals.std(ddof=1) if len(vals) > 1 else 0.0)
        if xs:
            ax.errorbar(xs, means, yerr=stds, color=color, marker=marker, ms=6, lw=1.5,
                        capsize=3, label=lab, mfc=color, mec="white", mew=1)
    ax.set_xscale("log")
    ax.set_xticks([f * 100 for f in fracs])
    ax.set_xticklabels([f"{f * 100:g}%" for f in fracs])
    ax.minorticks_off()
    ax.set_xlabel("Fraction of training labels used")
    ax.set_ylabel("Test macro-F1")
    ax.set_title("Label efficiency", fontsize=10)
    ax.grid(True, axis="y", color="#d0d0d0", lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    fig.text(0.01, 0.005, "Line = mean ± std over seeds; faint points = individual seeds",
             fontsize=7, color="#555555")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results")
    ap.add_argument("--no_plot", action="store_true")
    args = ap.parse_args()

    results = load_results(args.dir)
    if not results:
        raise SystemExit(f"No result files in {args.dir}/")

    groups: dict[float, dict[str, list[dict]]] = {}
    for r in results:
        groups.setdefault(frac_of(r), {}).setdefault(label_of(r), []).append(r)

    for frac in sorted(groups):
        rows = groups[frac]
        n_train = next(iter(rows.values()))[0]["split_sizes"]["train"]
        print(f"\n=== {frac:.0%} of training labels (~{n_train} images) ===")
        print(f"{'run type':<34}{'n':>3}  {'macro-F1':<13}{'F1 last-ep':<13}{'macroAUC':<13}"
              + "".join(f"F1[{c[:4]}]".ljust(13) for c in LABEL_COLS))
        for lab, rs in sorted(rows.items()):
            row = f"{lab:<34}{len(rs):>3}  "
            row += f"{fmt([r['test']['macro_f1'] for r in rs]):<13}"
            row += f"{fmt([last_f1(r) for r in rs]):<13}"
            row += f"{fmt([r['test'].get('macro_auc') for r in rs]):<13}"
            for c in LABEL_COLS:
                row += fmt([r["test"]["per_class"][c]["f1"] for r in rs]).ljust(13)
            print(row)

        # Paired comparison per variant: same seed => same split and labeled subset.
        variants = {variant_of(r) for rs in rows.values() for r in rs}
        for v in sorted(variants):
            tag = f" [{v}]" if v else ""
            sc, pr = rows.get(f"scratch{tag}", []), rows.get(f"pretrained{tag}", [])
            by_s = {r["config"]["seed"]: r for r in sc}
            by_p = {r["config"]["seed"]: r for r in pr}
            seeds = sorted(set(by_s) & set(by_p))
            if not seeds:
                continue
            d = [by_p[s]["test"]["macro_f1"] - by_s[s]["test"]["macro_f1"] for s in seeds]
            dl = [last_f1(by_p[s]) - last_f1(by_s[s]) for s in seeds]
            print(f"  paired Δ macro-F1 pretrained−scratch{tag}: "
                  + "  ".join(f"s{s}:{x:+.3f}" for s, x in zip(seeds, d))
                  + f"  | mean {np.mean(d):+.4f} (better in {sum(x > 0 for x in d)}/{len(d)})"
                  + "  | last-epoch mean "
                  + ("n/a" if np.isnan(dl).any() else f"{np.mean(dl):+.4f}"))

    csv_path = os.path.join(args.dir, "summary.csv")
    write_csv(results, csv_path)
    print(f"\nPer-run table → {csv_path}")
    if not args.no_plot and len(groups) > 1:
        png_path = os.path.join(args.dir, "label_efficiency.png")
        plot_label_efficiency(groups, png_path)
        print(f"Plot          → {png_path}")


if __name__ == "__main__":
    main()
