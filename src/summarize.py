"""
Aggregate ``results/*.json`` (written by ``src.train``) into a comparison of
the scratch vs. Phase-1-pretrained arms on the held-out test split, grouped by
the fraction of training labels used (``--train_frac``).

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

ARM_STYLE = {  # colour-blind-safe pair (Okabe-Ito) + distinct marker shapes
    "scratch": ("#0072B2", "o"),
    "pretrained": ("#D55E00", "s"),
}


def load_results(directory: str) -> list[dict]:
    paths = sorted(glob.glob(os.path.join(directory, "*.json")))
    results = [json.load(open(p)) for p in paths]
    # Debug runs (20 images) would pollute the statistics.
    return [r for r in results if not r["config"].get("debug")]


def frac_of(r: dict) -> float:
    return float(r["config"].get("train_frac", 1.0))


def mean_std(values: list[float]) -> str:
    arr = np.asarray(values, dtype=float)
    if np.isnan(arr).any():
        return "n/a"
    std = arr.std(ddof=1) if len(arr) > 1 else float("nan")
    return f"{arr.mean():.3f} ± {std:.3f}"


def last_f1(r: dict) -> float:
    """Test macro-F1 of the final-epoch model (NaN for runs made before it was logged)."""
    return r["test_last_epoch"]["macro_f1"] if "test_last_epoch" in r else float("nan")


def write_csv(results: list[dict], path: str) -> None:
    cols = ["run_name", "arm", "train_frac", "seed", "n_train", "best_epoch",
            "test_macro_f1", "test_last_macro_f1", "test_bal_acc", "test_acc"]
    cols += [f"recall_{c}" for c in LABEL_COLS]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in sorted(results, key=lambda r: (frac_of(r), r["arm"], r["config"]["seed"])):
            t = r["test"]
            w.writerow([
                r["run_name"], r["arm"], frac_of(r), r["config"]["seed"],
                r["split_sizes"]["train"], r["best_epoch"],
                f"{t['macro_f1']:.4f}", f"{last_f1(r):.4f}",
                f"{t['balanced_accuracy']:.4f}", f"{t['accuracy']:.4f}",
                *[f"{t['per_class'][c]['recall']:.4f}" for c in LABEL_COLS],
            ])


def plot_label_efficiency(groups: dict, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=150)
    fracs = sorted(groups)
    for arm in ("scratch", "pretrained"):
        color, marker = ARM_STYLE[arm]
        xs, means, stds = [], [], []
        for fr in fracs:
            rs = groups[fr].get(arm, [])
            if not rs:
                continue
            vals = np.array([r["test"]["macro_f1"] for r in rs])
            ax.scatter([fr * 100] * len(vals), vals, s=14, color=color, alpha=0.35,
                       marker=marker, linewidths=0)
            xs.append(fr * 100); means.append(vals.mean())
            stds.append(vals.std(ddof=1) if len(vals) > 1 else 0.0)
        if xs:
            ax.errorbar(xs, means, yerr=stds, color=color, marker=marker, ms=6,
                        lw=1.5, capsize=3, label=arm, mfc=color, mec="white", mew=1)
    ax.set_xscale("log")
    ax.set_xticks([f * 100 for f in fracs])
    ax.set_xticklabels([f"{f * 100:g}%" for f in fracs])
    ax.minorticks_off()
    ax.set_xlabel("Fraction of training labels used")
    ax.set_ylabel("Test macro-F1")
    ax.set_title("Label efficiency: scratch vs. Phase-1 pre-trained LoRA", fontsize=10)
    ax.grid(True, axis="y", color="#d0d0d0", lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, loc="lower right")
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
        groups.setdefault(frac_of(r), {}).setdefault(r["arm"], []).append(r)

    for frac in sorted(groups):
        arms = groups[frac]
        n_train = next(iter(arms.values()))[0]["split_sizes"]["train"]
        print(f"\n=== {frac:.0%} of training labels (~{n_train} images) ===")
        print(f"{'arm':<12}{'n':>3}  {'macro-F1':<16}{'F1 last-ep':<16}{'bal-acc':<16}"
              + "".join(f"R[{c[:4]}]".ljust(14) for c in LABEL_COLS))
        for arm, rs in sorted(arms.items()):
            row = f"{arm:<12}{len(rs):>3}  "
            row += f"{mean_std([r['test']['macro_f1'] for r in rs]):<16}"
            row += f"{mean_std([last_f1(r) for r in rs]):<16}"
            row += f"{mean_std([r['test']['balanced_accuracy'] for r in rs]):<16}"
            for c in LABEL_COLS:
                row += mean_std([r["test"]["per_class"][c]["recall"] for r in rs]).ljust(14)
            print(row)

        # Paired comparison: same seed => same split and same labeled subset.
        by_seed = {
            arm: {r["config"]["seed"]: r for r in rs} for arm, rs in arms.items()
        }
        if "scratch" in by_seed and "pretrained" in by_seed:
            seeds = sorted(set(by_seed["scratch"]) & set(by_seed["pretrained"]))
            d_best = [by_seed["pretrained"][s]["test"]["macro_f1"]
                      - by_seed["scratch"][s]["test"]["macro_f1"] for s in seeds]
            d_last = [last_f1(by_seed["pretrained"][s]) - last_f1(by_seed["scratch"][s])
                      for s in seeds]
            if d_best:
                print("  paired Δ macro-F1 (pretrained − scratch), best-val epoch: "
                      + "  ".join(f"s{s}:{d:+.3f}" for s, d in zip(seeds, d_best)))
                print(f"  mean Δ = {np.mean(d_best):+.4f} "
                      f"(better in {sum(d > 0 for d in d_best)}/{len(d_best)} seeds);  "
                      f"last-epoch mean Δ = "
                      + ("n/a" if np.isnan(d_last).any() else f"{np.mean(d_last):+.4f}"))

    csv_path = os.path.join(args.dir, "summary.csv")
    write_csv(results, csv_path)
    print(f"\nPer-run table → {csv_path}")
    if not args.no_plot and len(groups) > 1:
        png_path = os.path.join(args.dir, "label_efficiency.png")
        plot_label_efficiency(groups, png_path)
        print(f"Plot          → {png_path}")


if __name__ == "__main__":
    main()
