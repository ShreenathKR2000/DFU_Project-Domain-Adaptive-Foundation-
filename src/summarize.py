"""
Aggregate ``results/*.json`` (written by ``src.train``) into a comparison of
the scratch vs. Phase-1-pretrained arms on the held-out test split.

Usage
-----
    python -m src.summarize            # reads results/
    python -m src.summarize --dir results
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

from src.dataset import LABEL_COLS


def load_results(directory: str) -> list[dict]:
    paths = sorted(glob.glob(os.path.join(directory, "*.json")))
    results = [json.load(open(p)) for p in paths]
    # Debug runs (20 images) would pollute the statistics.
    return [r for r in results if not r["config"].get("debug")]


def mean_std(values: list[float]) -> str:
    arr = np.asarray(values, dtype=float)
    std = arr.std(ddof=1) if len(arr) > 1 else float("nan")
    return f"{arr.mean():.3f} ± {std:.3f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results")
    args = ap.parse_args()

    results = load_results(args.dir)
    if not results:
        raise SystemExit(f"No result files in {args.dir}/")

    arms: dict[str, list[dict]] = {}
    for r in results:
        arms.setdefault(r["arm"], []).append(r)

    print(f"{'arm':<12}{'n':>3}  {'macro-F1':<16}{'bal-acc':<16}{'acc':<16}"
          + "".join(f"R[{c[:4]}]".ljust(14) for c in LABEL_COLS))
    for arm, rs in sorted(arms.items()):
        row = f"{arm:<12}{len(rs):>3}  "
        row += f"{mean_std([r['test']['macro_f1'] for r in rs]):<16}"
        row += f"{mean_std([r['test']['balanced_accuracy'] for r in rs]):<16}"
        row += f"{mean_std([r['test']['accuracy'] for r in rs]):<16}"
        for c in LABEL_COLS:
            row += mean_std([r["test"]["per_class"][c]["recall"] for r in rs]).ljust(14)
        print(row)

    # Paired comparison: same seed => same split.
    by_seed = {
        arm: {r["config"]["seed"]: r["test"]["macro_f1"] for r in rs}
        for arm, rs in arms.items()
    }
    if "scratch" in by_seed and "pretrained" in by_seed:
        seeds = sorted(set(by_seed["scratch"]) & set(by_seed["pretrained"]))
        deltas = [by_seed["pretrained"][s] - by_seed["scratch"][s] for s in seeds]
        print("\nPaired test macro-F1 (pretrained − scratch), per seed:")
        for s, d in zip(seeds, deltas):
            print(f"  seed {s}: {d:+.4f}")
        if deltas:
            print(f"  mean Δ = {np.mean(deltas):+.4f}  "
                  f"(pretrained better in {sum(d > 0 for d in deltas)}/{len(deltas)} seeds)")


if __name__ == "__main__":
    main()
