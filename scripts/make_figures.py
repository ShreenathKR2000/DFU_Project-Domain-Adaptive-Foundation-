#!/usr/bin/env python
"""
Regenerate every figure in docs/figures/ and the tables in docs/results_tables.md
from the experiment outputs (results/*.json, results/ensemble_round2.csv,
logs/, artifacts/). Needs only numpy and matplotlib.

    python scripts/make_figures.py

Colour encodes the main factor (blue = scratch, vermillion = Phase-1 pre-training,
green = pseudo-labels); hatching marks the "strong augmentation + label smoothing"
recipe. Colours are the colour-blind-safe Okabe-Ito set and identity is never
colour-only (hatching, markers and value labels are always present).
"""

from __future__ import annotations

import csv
import glob
import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch

RES, LOGS, ART, OUT = "results", "logs", "artifacts", "docs/figures"
CLASSES = ["none", "infection", "ischaemia", "both"]
CLASS_LABELS = ["Control", "Infection", "Ischaemia", "Both"]

BLUE, VERM, GREEN, PINK, GREY = "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#6b6b6b"
INK, MUTED, GRID = "#1a1a1a", "#555555", "#dcdcdc"

# (label, run-name prefix, colour, hatch)
CONDITIONS = [
    ("Scratch", "g_scratch_basic", BLUE, ""),
    ("Phase 1", "g_pre_basic", VERM, ""),
    ("Phase 1 + pseudo-labels", "g_pre_pseudo", GREEN, ""),
    ("Scratch + strong aug/LS", "g_scratch_reg", BLUE, "///"),
    ("Phase 1 + strong aug/LS", "g_pre_reg", VERM, "///"),
    ("Student (scratch + strong aug/LS\n+ pseudo-labels)", "g_student", GREEN, "///"),
]

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "figure.dpi": 100, "savefig.dpi": 170,
    "savefig.bbox": "tight", "figure.facecolor": "white", "axes.facecolor": "white",
    "hatch.linewidth": 0.8,
})


# ── loading ───────────────────────────────────────────────────────────────────

def load_runs(prefix: str) -> list[dict]:
    files = sorted(glob.glob(f"{RES}/{prefix}_s*.json")) or sorted(glob.glob(f"{RES}/{prefix}_seed*.json"))
    return [json.load(open(f)) for f in files]


def metric(runs, key, cls=None, last=False):
    out = []
    for r in runs:
        t = r["test_last_epoch"] if last and "test_last_epoch" in r else r["test"]
        v = t["per_class"][cls]["f1"] if cls else t.get(key)
        out.append(np.nan if v is None else v)
    return np.array(out, dtype=float)


def ms(a):
    return float(np.mean(a)), float(np.std(a, ddof=1)) if len(a) > 1 else 0.0


def read_ens(tag: str, tta: bool):
    p = f"{LOGS}/exp/ens_{tag}{'_tta' if tta else ''}.txt"
    if not os.path.exists(p):
        return None
    m = re.search(r"macro-F1=([\d.]+)\s+bal_acc=([\d.]+)\s+acc=([\d.]+)\s+macro_auc=([\d.]+)", open(p).read())
    return dict(f1=float(m[1]), bal=float(m[2]), acc=float(m[3]), auc=float(m[4])) if m else None


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(f"{OUT}/{name}.png")
    plt.close(fig)
    print("wrote", f"{OUT}/{name}.png")


def legend_conditions(ax, loc="lower right"):
    handles = [
        Patch(facecolor=BLUE, label="scratch"), Patch(facecolor=VERM, label="Phase-1 pre-trained"),
        Patch(facecolor=GREEN, label="+ pseudo-labels"),
        Patch(facecolor="white", edgecolor=MUTED, hatch="///", label="strong aug + label smoothing"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=7.5, loc=loc)


# ── 1. pipeline flowchart ─────────────────────────────────────────────────────

def fig_pipeline():
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    ax.set_xlim(0, 17.0); ax.set_ylim(0, 7.4); ax.axis("off")

    def box(x, y, w, h, title, body, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.15",
                                    fc=color, ec=MUTED, lw=1.0, alpha=0.16))
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.15",
                                    fc="none", ec=color, lw=1.4))
        ax.text(x + w / 2, y + h - 0.32, title, ha="center", va="top", fontsize=9, fontweight="bold", color=INK)
        ax.text(x + w / 2, y + h - 0.85, body, ha="center", va="top", fontsize=7.6, color=INK, linespacing=1.35)

    def arrow(p, q, label=None, dashed=False, rad=0.0, lx=0, ly=0.25):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=11, color=MUTED, lw=1.2,
                                     linestyle="--" if dashed else "-", connectionstyle=f"arc3,rad={rad}"))
        if label:
            ax.text((p[0] + q[0]) / 2 + lx, (p[1] + q[1]) / 2 + ly, label, ha="center", fontsize=7, color=MUTED)

    w, h, top, bot = 3.0, 2.3, 4.5, 1.2
    xs = [0.2, 3.55, 6.9, 10.25, 13.6]
    box(xs[0], top, w, h, "DFUC2021 train", "5,955 labeled images\n(4 classes, imbalanced)\n3,994 unlabeled images", GREY)
    box(xs[1], top, w, h, "0  Leak-free split", "DINOv2 embeddings,\ncosine ≥ 0.85 → 3,925 groups\ngroup-aware 70/15/15", BLUE)
    box(xs[2], top, w, h, "2  Phase 2: classifier", "LoRA + linear head\nCE + SupCon loss\nbest epoch by val macro-F1", VERM)
    box(xs[3], top, w, h, "3  Teachers", "3 seeds, same split\nensemble + flip TTA\npredict unlabeled images", VERM)
    box(xs[4], top, w, h, "4  Pseudo-labels", "confidence ≥ 0.9\n≤ 400 per class\n→ +887 images", GREEN)
    box(xs[1], bot, w, h, "1  Phase 1: SimMIM", "mask 60 % of patches\nLoRA + light decoder, L1\n100 epochs, unlabeled only", VERM)
    box(xs[2] + 0.0, bot, w, h, "Evaluation", "held-out test split\nmacro-F1, per-class F1,\nAUC, confusion matrix", GREY)
    box(xs[4], bot, w, h, "5  Final models", "train on ALL labels\n+ pseudo-labels (6,842)\n3 seeds, 20 epochs", GREEN)
    box(xs[3], bot, w, h, "6  Inference", "ensemble + flip TTA\n→ submission CSV\n(labels + probabilities)", GREEN)

    arrow((xs[0] + w, top + h / 2), (xs[1], top + h / 2))
    arrow((xs[1] + w, top + h / 2), (xs[2], top + h / 2), "splits")
    arrow((xs[2] + w, top + h / 2), (xs[3], top + h / 2))
    arrow((xs[3] + w, top + h / 2), (xs[4], top + h / 2))
    arrow((xs[0] + w * 0.6, top), (xs[1], bot + h * 0.7), "unlabeled", rad=0.15, lx=-0.5, ly=0.0)
    arrow((xs[1] + w, bot + h * 0.7), (xs[2] + 0.4, top), None, rad=-0.1)
    ax.text(6.0, 3.95, "LoRA init", fontsize=7, color=MUTED, ha="right")
    arrow((xs[2] + w / 2, top), (xs[2] + w / 2, bot + h), "test", rad=0.0, lx=0.3, ly=0.0)
    arrow((xs[4] + w / 2, top), (xs[4] + w / 2, bot + h))
    arrow((xs[4], bot + h / 2), (xs[3] + w, bot + h / 2))
    ax.plot([xs[1] + w / 2, xs[1] + w / 2, xs[4] + w / 2], [bot, bot - 0.5, bot - 0.5], color=MUTED, lw=1.2, ls="--")
    ax.add_patch(FancyArrowPatch((xs[4] + w / 2, bot - 0.5), (xs[4] + w / 2, bot), arrowstyle="-|>",
                                 mutation_scale=11, color=MUTED, lw=1.2, linestyle="--"))
    ax.text(8.9, 0.18, "Phase-1 LoRA weights also initialise the final models", ha="center", fontsize=7, color=MUTED)
    ax.set_title("Approach: leak-free evaluation, domain-adaptive pre-training, pseudo-labelling, ensembling",
                 fontsize=10.5, loc="left", pad=6)
    save(fig, "fig_pipeline")


# ── 2. leakage ────────────────────────────────────────────────────────────────

def parse_groups_log():
    txt = open(f"{LOGS}/exp/make_groups.log").read()
    table = [(float(a), int(b), int(c), int(d)) for a, b, c, d in
             re.findall(r"^\s+(0\.\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*$", txt, flags=re.M)]
    rnd = [float(x) for x in re.search(r"random split\s+([\d.]+)%\s+([\d.]+)%\s+([\d.]+)%", txt).groups()]
    grp = [float(x) for x in re.search(r"group-aware split\s+([\d.]+)%\s+([\d.]+)%\s+([\d.]+)%", txt).groups()]
    return table, rnd, grp


def fig_leakage():
    table, rnd, grp = parse_groups_log()
    fig, axs = plt.subplots(1, 3, figsize=(12.2, 3.9), gridspec_kw={"width_ratios": [1.15, 1, 1.15]})

    # (a) random vs group-aware split, same model family
    ax = axs[0]
    rows = [("Scratch", BLUE, "scratch", "g_scratch_basic"), ("Phase 1", VERM, "pretrained", "g_pre_basic")]
    for i, (lab, col, pr, pg) in enumerate(rows):
        for j, (pref, mk) in enumerate(((pr, "o"), (pg, "s"))):
            v = metric(load_runs(pref), "macro_f1")
            x = i + (j - 0.5) * 0.34
            ax.bar(x, v.mean(), 0.3, color=col, alpha=0.9 if j else 0.45, hatch="" if j else "..", ec=col)
            ax.scatter([x] * len(v), v, s=14, color=INK, zorder=3)
            ax.text(x, max(v.max(), v.mean()) + 0.014, f"{v.mean():.3f}", ha="center", fontsize=8)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Scratch", "Phase 1"])
    ax.set_ylim(0.6, 0.95); ax.set_ylabel("Test macro-F1 (3 seeds)")
    ax.legend(handles=[Patch(facecolor="white", edgecolor=MUTED, hatch="..", label="random split"),
                       Patch(facecolor=MUTED, label="group-aware split")], frameon=False, fontsize=8, loc="upper right")
    ax.set_title("(a) Random split is optimistic", loc="left", fontsize=9.5)
    ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)

    # (b) leak diagnostic
    ax = axs[1]
    x = np.arange(3); ths = ["≥ 0.90", "≥ 0.95", "≥ 0.98"]
    ax.bar(x - 0.18, rnd, 0.34, color=GREY, label="random split")
    ax.bar(x + 0.18, grp, 0.34, color=BLUE, label="group-aware split")
    for xi, v in zip(x - 0.18, rnd):
        ax.text(xi, v + 0.5, f"{v:.1f}%", ha="center", fontsize=8)
    for xi, v in zip(x + 0.18, grp):
        ax.text(xi, v + 0.5, f"{v:.1f}%", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(ths)
    ax.set_xlabel("cosine similarity to a training image"); ax.set_ylabel("share of test images with such a neighbour (%)")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("(b) Near-duplicate leakage", loc="left", fontsize=9.5)

    # (c) choosing the clustering threshold
    ax = axs[2]
    thr = [t[0] for t in table]; ngr = [t[1] for t in table]; big = [t[2] for t in table]
    bars = ax.bar(range(len(thr)), big, 0.55, color=PINK, hatch="//", ec="white", label="largest cluster (images)")
    ax.set_yscale("log"); ax.set_ylabel("largest cluster (images, log)")
    ax.set_xticks(range(len(thr))); ax.set_xticklabels([f"{t:.2f}" for t in thr]); ax.set_xlabel("cosine threshold")
    for i, (b, g) in enumerate(zip(big, ngr)):
        ax.text(i, b * 1.15, f"{b}\n{g} groups", ha="center", fontsize=7.5)
    ax.axhline(119, color=MUTED, ls="--", lw=1); ax.text(len(thr) - 0.5, 130, "limit 119 (2 %)", ha="right", fontsize=7.5, color=MUTED)
    i85 = thr.index(0.85)
    bars[i85].set_edgecolor(INK); bars[i85].set_linewidth(2)
    ax.text(i85, 1.6, "chosen", ha="center", color="white", fontsize=8, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.15", fc=INK, ec="none"))
    ax.set_ylim(1, 8000)
    ax.set_title("(c) Threshold selection (--auto)", loc="left", fontsize=9.5)
    fig.suptitle("Why the evaluation protocol was changed", fontsize=10.5, x=0.01, ha="left", y=1.02)
    save(fig, "fig_leakage")


# ── 3. ablation (forest-style dot plots) ──────────────────────────────────────

def forest(ax, key, cls=None, title="", xlim=None, show_labels=True):
    ys = np.arange(len(CONDITIONS))[::-1]
    for y, (lab, pref, col, hatch) in zip(ys, CONDITIONS):
        v = metric(load_runs(pref), key, cls)
        m, s = ms(v)
        ax.hlines(y, m - s, m + s, color=col, lw=3.2, alpha=0.5)
        ax.scatter(v, [y] * len(v), s=16, color=INK, zorder=3)
        ax.scatter([m], [y], s=70, marker="D" if hatch else "o", color=col, zorder=4, edgecolor="white", lw=0.8)
        ax.text(m, y + 0.34, f"{m:.3f}", ha="center", fontsize=7.5)
    ax.set_yticks(ys)
    ax.set_yticklabels([c[0] for c in CONDITIONS], fontsize=8)
    if not show_labels:
        ax.tick_params(labelleft=False)
    ax.set_ylim(-0.6, len(CONDITIONS) - 0.1)
    ax.set_title(title, loc="left", fontsize=9.5, pad=10)
    ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
    if xlim:
        ax.set_xlim(*xlim)


def fig_ablation():
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.2), sharey=True)
    forest(axs[0], "macro_f1", title="Test macro-F1  (leaderboard ranking metric)", xlim=(0.72, 0.88))
    forest(axs[1], "macro_auc", title="Test macro AUC", xlim=(0.87, 0.96), show_labels=False)
    axs[0].set_xlabel("mean ± std over 3 seeds (black dots = individual seeds; diamond = strong aug/LS recipe)")
    axs[1].set_xlabel("mean ± std over 3 seeds")
    fig.suptitle("Group-aware split: what each ingredient adds", fontsize=10.5, x=0.01, ha="left", y=1.02)
    save(fig, "fig_ablation")


def fig_perclass():
    fig, axs = plt.subplots(1, 4, figsize=(13.5, 3.8), sharey=True)
    for i, (c, lab) in enumerate(zip(CLASSES, CLASS_LABELS)):
        forest(axs[i], "f1", cls=c, title=f"F1 — {lab}", xlim=(0.55, 0.95), show_labels=(i == 0))
        axs[i].set_xlabel("test F1")
    n = load_runs("g_pre_basic")[0]["split_class_counts"]["test"]
    fig.suptitle("Per-class F1 on the group-aware test split (n = "
                 + ", ".join(f"{k} {v}" for k, v in zip(CLASS_LABELS, n.values()))
                 + ")  —  ischaemia has only 29 images, so its F1 is noisy", fontsize=10, x=0.01, ha="left", y=1.03)
    save(fig, "fig_perclass")


# ── 4. learning curves ────────────────────────────────────────────────────────

def fig_curves():
    sel = [("Scratch", "g_scratch_basic", BLUE, "o"), ("Phase 1", "g_pre_basic", VERM, "s"),
           ("Phase 1 + pseudo-labels", "g_pre_pseudo", GREEN, "^")]
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
    for lab, pref, col, mk in sel:
        runs = load_runs(pref)
        ep = np.array([h["epoch"] for h in runs[0]["history"]])
        val = np.array([[h["val"]["macro_f1"] for h in r["history"]] for r in runs])
        tr = np.array([[h["train"]["ce_loss"] for h in r["history"]] for r in runs])
        for ax, arr in ((axs[0], val), (axs[1], tr)):
            m, s = arr.mean(0), arr.std(0, ddof=1)
            ax.plot(ep, m, color=col, marker=mk, ms=3.5, lw=1.4, label=lab)
            ax.fill_between(ep, m - s, m + s, color=col, alpha=0.15)
        be = np.mean([r["best_epoch"] for r in runs])
        axs[0].axvline(be, color=col, ls=":", lw=1)
        axs[0].text(be + 0.15, 0.51, f"mean best epoch {be:.0f}", color=col, fontsize=7.5, ha="left", rotation=90, va="bottom")
    axs[0].set_xlabel("epoch"); axs[0].set_ylabel("validation macro-F1"); axs[0].set_ylim(0.5, 0.88)
    axs[0].set_title("(a) Validation macro-F1 (group split)", loc="left", fontsize=9.5)
    axs[1].set_xlabel("epoch"); axs[1].set_ylabel("training cross-entropy"); axs[1].set_yscale("log")
    axs[1].set_title("(b) Training loss", loc="left", fontsize=9.5)
    axs[0].legend(frameon=False, fontsize=8, loc="upper left")
    for ax in axs:
        ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
    save(fig, "fig_curves")


def fig_phase1():
    txt = open(f"{LOGS}/run_all.log").read()
    pts = [(int(e), float(l)) for e, l in re.findall(r"Epoch\s+(\d+)/100\s+loss=([\d.]+)\s+\(", txt)]
    ep, loss = zip(*pts)
    fig, ax = plt.subplots(figsize=(5.6, 3.3))
    ax.plot(ep, loss, color=VERM, lw=1.6)
    ax.set_xlabel("epoch"); ax.set_ylabel("masked-patch L1 reconstruction loss"); ax.set_ylim(0.08, 0.45)
    ax.annotate(f"{loss[0]:.3f}", (ep[0], loss[0]), xytext=(8, -4), textcoords="offset points", fontsize=8)
    ax.annotate(f"{loss[-1]:.3f}", (ep[-1], loss[-1]), xytext=(-30, 12), textcoords="offset points", fontsize=8)
    ax.set_title("Phase 1 (SimMIM) pre-training on 3,994 unlabeled images", loc="left", fontsize=9.5)
    ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
    save(fig, "fig_phase1")


# ── 5. confusion matrices ─────────────────────────────────────────────────────

def fig_confusion():
    sel = [("Scratch", "g_scratch_basic"), ("Phase 1", "g_pre_basic"), ("Phase 1 + pseudo-labels", "g_pre_pseudo")]
    fig, axs = plt.subplots(1, 3, figsize=(12.2, 4.0))
    for ax, (lab, pref) in zip(axs, sel):
        cm = sum(np.array(r["test"]["confusion_matrix"]) for r in load_runs(pref))
        norm = cm / cm.sum(1, keepdims=True)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{100 * norm[i, j]:.0f}%\n({cm[i, j]})", ha="center", va="center", fontsize=8,
                        color="white" if norm[i, j] > 0.55 else INK)
        ax.set_xticks(range(4)); ax.set_xticklabels(CLASS_LABELS, rotation=30, ha="right")
        ax.set_yticks(range(4)); ax.set_yticklabels(CLASS_LABELS)
        ax.set_xlabel("predicted")
        if ax is axs[0]:
            ax.set_ylabel("true")
        ax.set_title(lab, loc="left", fontsize=9.5)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.subplots_adjust(wspace=0.5)
    fig.suptitle("Confusion matrices on the group-aware test split (3 seeds summed, row-normalised; counts in brackets)",
                 fontsize=10, x=0.01, ha="left", y=1.02)
    save(fig, "fig_confusion")


# ── 6. ensembles / TTA ────────────────────────────────────────────────────────

def fig_ensemble():
    rows = [("Scratch", "g_scratch_basic", "scratch_basic", BLUE, ""), ("Phase 1", "g_pre_basic", None, VERM, ""),
            ("Phase 1 + pseudo-labels", "g_pre_pseudo", None, GREEN, ""),
            ("Scratch + strong aug/LS", "g_scratch_reg", "scratch_reg", BLUE, "///"),
            ("Phase 1 + strong aug/LS", "g_pre_reg", "pre_reg", VERM, "///"),
            ("Student", "g_student", "student", GREEN, "///"),
            ("All 9 models\n(Phase 1 + pseudo + student)", None, None, GREY, "")]
    r2 = {}
    for line in open(f"{RES}/ensemble_round2.csv"):
        if line.startswith("#") or line.startswith("set,"):
            continue
        p = next(csv.reader([line]))
        r2[p[0]] = float(p[3])
    r2map = {"g_pre_basic": r2["pre_basic"], "g_pre_pseudo": r2["pre_pseudo"], None: r2["all9"]}
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    ys = np.arange(len(rows))[::-1]
    for y, (lab, pref, tag, col, hatch) in zip(ys, rows):
        single = ms(metric(load_runs(pref), "macro_f1"))[0] if pref else None
        ens = read_ens(tag, False)["f1"] if tag else None
        tta = read_ens(tag, True)["f1"] if tag else r2map.get(pref)
        pts = [(single, "o", "single model (mean of 3)"), (ens, "s", "3-model ensemble"), (tta, "D", "ensemble + flip TTA")]
        xs = [p[0] for p in pts if p[0] is not None]
        if len(xs) > 1:
            ax.hlines(y, min(xs), max(xs), color=col, lw=2, alpha=0.5)
        for v, mk, _ in pts:
            if v is not None:
                ax.scatter([v], [y], marker=mk, s=60, color=col, edgecolor="white", lw=0.8, zorder=3)
                ax.text(v, y + 0.3, f"{v:.3f}", ha="center", fontsize=7.5)
    ax.set_yticks(ys); ax.set_yticklabels([r[0] for r in rows], fontsize=8)
    ax.set_xlim(0.75, 0.87); ax.set_ylim(-0.7, len(rows) - 0.2); ax.set_xlabel("test macro-F1 (group-aware split)")
    ax.legend(handles=[plt.Line2D([], [], marker=m, ls="", color=MUTED, label=l)
                       for m, l in (("o", "single model (mean of 3 seeds)"), ("s", "3-model ensemble"), ("D", "ensemble + flip TTA"))],
              frameon=False, fontsize=8, loc="lower left")
    ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.set_title("Ensembling and flip test-time augmentation", loc="left", fontsize=10, pad=10)
    fig.text(0.01, -0.02, "Rows without a square have no non-TTA ensemble result (round-2 numbers were transcribed from terminal output).",
             fontsize=7, color=MUTED)
    save(fig, "fig_ensemble")


# ── 7. challenge test shift ───────────────────────────────────────────────────

def read_onehot(path):
    rows = list(csv.reader(open(path)))[1:]
    return np.array([[int(x) for x in r[1:]] for r in rows]).sum(0)


def read_probs(path):
    return np.array([[float(x) for x in r[1:]] for r in list(csv.reader(open(path)))[1:]])


def fig_shift():
    c = load_runs("g_pre_basic")[0]["split_class_counts"]
    tot = np.array([sum(c[s][k] for s in c) for k in CLASSES], dtype=float)
    test = np.array([c["test"][k] for k in CLASSES], dtype=float)
    p9, p3 = read_onehot(f"{ART}/sub_groupmodels.csv"), read_onehot(f"{ART}/sub_final.csv")
    series = [("Training labels (5,955)", tot, GREY, ""), ("Group-split test, true (813)", test, BLUE, ""),
              ("Challenge test, predicted by 9 models (5,734)", p9, VERM, "///"),
              ("Challenge test, predicted by 3 final models (5,734)", p3, GREEN, "///")]
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.0), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axs[0]
    w = 0.2
    for i, (lab, v, col, hatch) in enumerate(series):
        share = 100 * v / v.sum()
        xs = np.arange(4) + (i - 1.5) * w
        ax.bar(xs, share, w * 0.92, color=col, hatch=hatch, ec="white" if not hatch else col, label=lab)
        for x, s in zip(xs, share):
            ax.text(x, s + 0.8, f"{s:.0f}", ha="center", fontsize=7)
    ax.set_xticks(range(4)); ax.set_xticklabels(CLASS_LABELS); ax.set_ylabel("share of images (%)")
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    ax.set_title("(a) Class mix: training data vs predictions on the challenge test set", loc="left", fontsize=9.5)
    ax = axs[1]
    for lab, path, col, ls in (("3 final models", f"{ART}/sub_final_probs.csv", GREEN, "-"),
                               ("9 group-split models", f"{ART}/sub_groupmodels_probs.csv", VERM, "--")):
        conf = read_probs(path).max(1)
        h, e = np.histogram(conf, bins=np.linspace(0.25, 1, 31))
        ax.plot((e[:-1] + e[1:]) / 2, 100 * h / h.sum(), color=col, ls=ls, lw=1.8, label=f"{lab}: mean {conf.mean():.2f}")
    ax.set_xlabel("top-class probability"); ax.set_ylabel("share of test images per bin (%)")
    ax.legend(frameon=False, fontsize=7.5, loc="center left", bbox_to_anchor=(0.0, 0.62))
    ax.text(0.27, 12, "3 of the 9 group-split models use\nlabel smoothing, which lowers confidence", fontsize=7, color=MUTED)
    ax.set_title("(b) Prediction confidence on the challenge test images", loc="left", fontsize=9.5)
    save(fig, "fig_shift")


# ── 8. pseudo-labels ──────────────────────────────────────────────────────────

def select_pseudo(path, thresh, cap):
    rows = list(csv.reader(open(path)))[1:]
    p = np.array([[float(x) for x in r[1:]] for r in rows])
    pred, conf = p.argmax(1), p.max(1)
    keep = conf >= thresh
    kept = np.zeros(4, int)
    allc = np.bincount(pred, minlength=4)
    for c in range(4):
        idx = np.where(keep & (pred == c))[0]
        kept[c] = min(len(idx), cap) if cap else len(idx)
    return allc, kept


def fig_pseudo():
    fig, axs = plt.subplots(1, 2, figsize=(10.8, 3.8), sharey=True)
    for ax, (title, path, th, cap) in zip(axs, (
            ("Teachers: Phase 1 models, threshold 0.9 (used for the final models)", f"{ART}/pseudo_pre_probs.csv", 0.9, 400),
            ("Teachers: scratch + strong aug/LS, threshold 0.7 (student)", f"{ART}/pseudo_g_probs.csv", 0.7, 400))):
        allc, kept = select_pseudo(path, th, cap)
        x = np.arange(4)
        ax.bar(x - 0.2, allc, 0.38, color=GREY, label="teacher predictions (3,994 unlabeled images)")
        ax.bar(x + 0.2, kept, 0.38, color=GREEN, hatch="///", ec="white", label=f"kept: confidence ≥ {th}, ≤ {cap} per class")
        for xi, a, k in zip(x, allc, kept):
            ax.text(xi - 0.2, a * 1.08, str(a), ha="center", fontsize=7.5)
            ax.text(xi + 0.2, max(k, 1) * 1.08, str(k), ha="center", fontsize=7.5)
        ax.set_yscale("log"); ax.set_ylim(10, 6000)
        ax.set_xticks(x); ax.set_xticklabels(CLASS_LABELS)
        ax.set_title(title, loc="left", fontsize=8.5)
        ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    axs[0].set_ylabel("images (log scale)")
    fig.suptitle("Pseudo-labels: the rare classes contribute few confident images", fontsize=10.5, x=0.01, ha="left", y=1.02)
    save(fig, "fig_pseudo")


# ── 9. the story ──────────────────────────────────────────────────────────────

def fig_story():
    rs, rp = metric(load_runs("scratch"), "macro_f1"), None
    gs = metric(load_runs("g_scratch_basic"), "macro_f1")
    gp = metric(load_runs("g_pre_basic"), "macro_f1")
    gpp = metric(load_runs("g_pre_pseudo"), "macro_f1")
    ens3 = read_ens("pre_reg", True)["f1"]
    r2 = {}
    for line in open(f"{RES}/ensemble_round2.csv"):
        if line.startswith("#") or line.startswith("set,"):
            continue
        p = next(csv.reader([line])); r2[p[0]] = float(p[3])
    steps = [("Scratch,\nrandom split", ms(rs), GREY), ("Scratch,\ngroup split", ms(gs), BLUE),
             ("+ Phase 1", ms(gp), VERM), ("+ pseudo-\nlabels", ms(gpp), GREEN),
             ("3 models\n+ TTA", (r2["pre_pseudo"], 0), GREEN), ("9 models\n+ TTA", (r2["all9"], 0), GREEN)]
    fig, ax = plt.subplots(figsize=(9.6, 4.6))
    ax.axhspan(0.652, 0.659, color=PINK, alpha=0.35)
    ax.text(5.55, 0.667, "leaderboard top-4: 0.652–0.659", ha="right", fontsize=8, color=MUTED)
    ax.axhspan(0.55, 0.592, color=GREY, alpha=0.18)
    ax.text(5.55, 0.5365, "leaderboard ranks 32–50: 0.55–0.59", ha="right", fontsize=8, color=MUTED)
    for i, (lab, (m, s), col) in enumerate(steps):
        ax.errorbar(i, m, yerr=s if s else None, fmt="o" if i < 4 else "D", ms=9, color=col, mec="white", capsize=4, lw=2)
        ax.text(i, m + 0.016, f"{m:.3f}", ha="center", fontsize=8.5)
    ax.plot(range(len(steps)), [s[1][0] for s in steps], color=MUTED, lw=0.8, ls=":", zorder=0)
    ax.set_xticks(range(len(steps))); ax.set_xticklabels([s[0] for s in steps], fontsize=8.5)
    ax.set_ylim(0.5, 0.93); ax.set_ylabel("macro-F1")
    ax.text(0.55, 0.862, "leakage: −0.11", fontsize=8, ha="left", color=MUTED)
    ax.set_title("From an optimistic internal score to the leaderboard range", loc="left", fontsize=10.5)
    fig.text(0.01, -0.04, "Points are our held-out test splits (mean ± std over 3 seeds; ensembles are single numbers). "
             "The shaded leaderboard bands use a different, hidden test set: shown for context, NOT comparable.",
             fontsize=7.2, color=MUTED, wrap=True)
    ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    save(fig, "fig_story")


# ── markdown tables ───────────────────────────────────────────────────────────

def write_tables():
    lines = ["# Result tables (generated by `scripts/make_figures.py`)", "",
             "Test split of the group-aware protocol (813 images; 376 none / 347 infection / 29 ischaemia / 61 both). "
             "Mean ± std over seeds 0–2.", "",
             "| Condition | n | Macro-F1 | Last-epoch macro-F1 | Macro AUC | F1 Control | F1 Infection | F1 Ischaemia | F1 Both | Accuracy |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    fmt = lambda a: "n/a" if np.isnan(a).any() else f"{a.mean():.3f} ± {a.std(ddof=1):.3f}"
    for lab, pref, _, _ in CONDITIONS:
        r = load_runs(pref)
        lines.append(f"| {lab.replace(chr(10), ' ')} | {len(r)} | {fmt(metric(r, 'macro_f1'))} | {fmt(metric(r, 'macro_f1', last=True))} | "
                     f"{fmt(metric(r, 'macro_auc'))} | " + " | ".join(fmt(metric(r, 'f1', c)) for c in CLASSES)
                     + f" | {fmt(np.array([x['test']['accuracy'] for x in r]))} |")
    lines += ["", "Earlier random-split runs (894-image test split, 30 epochs, fp32; not comparable to the table above):", "",
              "| Condition | n | Macro-F1 | F1 Control | F1 Infection | F1 Ischaemia | F1 Both |", "|---|---|---|---|---|---|---|"]
    for lab, pref in (("Scratch (random split)", "scratch"), ("Phase 1 (random split)", "pretrained")):
        r = load_runs(pref)
        lines.append(f"| {lab} | {len(r)} | {fmt(metric(r, 'macro_f1'))} | " + " | ".join(fmt(metric(r, 'f1', c)) for c in CLASSES) + " |")
    lines += ["", "## Ensembles on the same test split", "", "| Set | Ensemble macro-F1 | + flip TTA | Macro AUC (TTA) | Accuracy (TTA) |", "|---|---|---|---|---|"]
    for lab, tag in (("Scratch", "scratch_basic"), ("Scratch + strong aug/LS", "scratch_reg"),
                     ("Phase 1 + strong aug/LS", "pre_reg"), ("Student", "student")):
        a, b = read_ens(tag, False), read_ens(tag, True)
        lines.append(f"| {lab} | {a['f1']:.4f} | {b['f1']:.4f} | {b['auc']:.4f} | {b['acc']:.3f} |")
    for line in open(f"{RES}/ensemble_round2.csv"):
        if line.startswith("#") or line.startswith("set,"):
            continue
        p = next(csv.reader([line]))
        lines.append(f"| {p[0]} ({p[1]}) | n/a | {p[3]} | {p[4]} | {p[6]} |")
    lines += ["", "## Every run", "", "| Run | Condition | Best epoch | Macro-F1 | Last-epoch F1 | Macro AUC | Acc | F1 C / I / Is / B |", "|---|---|---|---|---|---|---|---|"]
    for f in sorted(glob.glob(f"{RES}/*.json")):
        r = json.load(open(f))
        if r.get("mode") == "final":
            continue
        t, la = r["test"], r.get("test_last_epoch", {})
        auc = "" if t.get("macro_auc") is None else f"{t['macro_auc']:.3f}"
        lf = f"{la['macro_f1']:.3f}" if la else "n/a"
        lines.append(f"| {r['run_name']} | {r['arm']} | {r['best_epoch']} | {t['macro_f1']:.4f} | {lf} | {auc} | {t['accuracy']:.3f} | "
                     + " / ".join(f"{t['per_class'][c]['f1']:.2f}" for c in CLASSES) + " |")
    open("docs/results_tables.md", "w").write("\n".join(lines) + "\n")
    print("wrote docs/results_tables.md")


if __name__ == "__main__":
    for fn in (fig_pipeline, fig_leakage, fig_ablation, fig_perclass, fig_curves, fig_phase1, fig_confusion,
               fig_ensemble, fig_shift, fig_pseudo, fig_story, write_tables):
        fn()
