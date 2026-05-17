"""
Generate 4 thesis-grade figures (PDF, vector) for Chapter 5 SQA Performance.

Output:
  backend/figures/
    fig_label_distribution_per_dataset.pdf
    fig_quality_heatmap_subject.pdf
    fig_sqi_box_plot_per_label.pdf
    fig_dataset_kSQI_distribution.pdf

Usage:
  python "backend/figures_pseudo_labels.py"
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PSEUDO_CSV = ROOT / "pseudo_labels.csv"
FIG_DIR = ROOT / "figures"

DATASETS = ["BP-protocol", "S09-SQA", "BIDMC", "PPG-DaLiA"]
LABELS = ["excellent", "acceptable", "unfit"]
LABEL_COLORS = {
    "excellent": "#2ecc71",
    "acceptable": "#f1c40f",
    "unfit": "#e74c3c",
}


# ============================================================
# Figure 1: stacked bar of label distribution per dataset
# ============================================================
def fig_label_distribution(df: pd.DataFrame) -> Path:
    grouped = pd.crosstab(df["dataset"], df["label"], normalize="index") * 100
    # Reindex rows in DATASETS order, columns in LABELS order
    grouped = grouped.reindex(index=DATASETS, columns=LABELS, fill_value=0.0)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    grouped.plot(
        kind="barh",
        stacked=True,
        ax=ax,
        color=[LABEL_COLORS[lab] for lab in LABELS],
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_xlabel("% of windows")
    ax.set_ylabel("")
    ax.set_title("Pseudo-label distribution per dataset (Orphanidou 2015 rules)")
    ax.legend(title="Label", loc="lower right", framealpha=0.9)
    ax.set_xlim(0, 110)

    # Annotate total N at right margin
    counts = df.groupby("dataset").size()
    for i, ds in enumerate(DATASETS):
        n_total = int(counts.get(ds, 0))
        ax.text(101, i, f"N={n_total}", va="center", fontsize=9)

    plt.tight_layout()
    out = FIG_DIR / "fig_label_distribution_per_dataset.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


# ============================================================
# Figure 2: per-subject % excellent (4 subplots, one per dataset)
# ============================================================
def fig_quality_heatmap(df: pd.DataFrame) -> Path:
    pct = (
        df.groupby(["dataset", "subject_id"])
        .apply(lambda g: (g["label"] == "excellent").mean() * 100)
        .reset_index(name="pct_excellent")
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, dataset in zip(axes.flat, DATASETS):
        sub = pct[pct["dataset"] == dataset].sort_values("pct_excellent", ascending=True)
        if sub.empty:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(dataset)
            continue
        ax.barh(
            sub["subject_id"].astype(str),
            sub["pct_excellent"],
            color="#2ecc71",
            edgecolor="black",
            linewidth=0.3,
        )
        ax.set_xlabel("% excellent windows")
        ax.set_title(f"{dataset}  (n_subjects={len(sub)})")
        ax.set_xlim(0, 100)
        ax.tick_params(axis="y", labelsize=7)

    fig.suptitle("Per-subject signal quality (Orphanidou excellent fraction)", fontsize=12)
    plt.tight_layout()
    out = FIG_DIR / "fig_quality_heatmap_subject.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


# ============================================================
# Figure 3: SQI feature box plots per label class
# ============================================================
def fig_sqi_box_plot(df: pd.DataFrame) -> Path:
    features = [
        ("template_r_median", "tSQI median"),
        ("hr_mean", "HR mean (BPM)"),
        ("rr_ratio", "RR ratio max/min"),
        ("rr_gap_s", "RR max gap (s)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (col, title) in zip(axes.flat, features):
        data = [df[df["label"] == lab][col].dropna().to_numpy() for lab in LABELS]
        bp = ax.boxplot(
            data,
            tick_labels=LABELS,
            patch_artist=True,
            showfliers=False,
        )
        for patch, lab in zip(bp["boxes"], LABELS):
            patch.set_facecolor(LABEL_COLORS[lab])
            patch.set_alpha(0.7)
        ax.set_title(title)
        ax.set_xlabel("")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("SQI features by Orphanidou label class", fontsize=12)
    plt.tight_layout()
    out = FIG_DIR / "fig_sqi_box_plot_per_label.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


# ============================================================
# Figure 4: cross-dataset kSQI distribution ( finding)
# ============================================================
def fig_ksqi_cross_dataset() -> Path | None:
    sources = [
        (ROOT / "eval_sqa_preliminary_results.csv", "self_collect (BP-proto)"),
        (ROOT / "eval_sqa_bidmc_results.csv", "BIDMC"),
        (ROOT / "eval_sqa_ppg_dalia_results.csv", "PPG-DaLiA"),
    ]
    dfs: list[pd.DataFrame] = []
    for path, label in sources:
        if not path.exists():
            print(f"  [skip] {path.name} not found")
            continue
        try:
            d = pd.read_csv(path)
        except Exception as e:
            print(f"  [skip] {path.name}: {type(e).__name__}: {e}")
            continue
        if "ksqi" not in d.columns:
            print(f"  [skip] {path.name} missing 'ksqi' column. Cols={list(d.columns)[:8]}")
            continue
        d = d[["ksqi"]].copy()
        d["source"] = label
        dfs.append(d)

    if not dfs:
        print("  [warn] no kSQI source files available; skipping figure 4")
        return None

    df = pd.concat(dfs, ignore_index=True)
    df = df[np.isfinite(df["ksqi"])]

    fig, ax = plt.subplots(figsize=(10, 6))
    sources_order = sorted(df["source"].unique())
    box_data = [df[df["source"] == s]["ksqi"].to_numpy() for s in sources_order]
    bp = ax.boxplot(
        box_data,
        tick_labels=sources_order,
        patch_artist=True,
        showfliers=True,
        flierprops=dict(marker=".", markersize=2, alpha=0.4),
    )
    palette = ["#3498db", "#9b59b6", "#e67e22", "#1abc9c"]
    for patch, color in zip(bp["boxes"], palette[: len(sources_order)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_yscale("symlog")
    ax.set_ylabel("kSQI (raw kurtosis, symlog scale)")
    ax.set_title("Cross-dataset kSQI distribution ( hardware finding)")
    ax.grid(axis="y", alpha=0.3, which="both")

    plt.tight_layout()
    out = FIG_DIR / "fig_dataset_kSQI_distribution.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


# ============================================================
# Main
# ============================================================
def main() -> int:
    if not PSEUDO_CSV.exists():
        print(f"[error] {PSEUDO_CSV} not found")
        return 1

    df = pd.read_csv(PSEUDO_CSV)
    print(f"Loaded {len(df)} rows from {PSEUDO_CSV.name}")

    FIG_DIR.mkdir(exist_ok=True)
    print(f"Output dir: {FIG_DIR}")

    print()
    print("[1/4] label distribution per dataset...")
    p1 = fig_label_distribution(df)
    print(f"   -> {p1.name}")

    print("[2/4] per-subject quality bars...")
    p2 = fig_quality_heatmap(df)
    print(f"   -> {p2.name}")

    print("[3/4] SQI box plots per label...")
    p3 = fig_sqi_box_plot(df)
    print(f"   -> {p3.name}")

    print("[4/4] cross-dataset kSQI distribution...")
    p4 = fig_ksqi_cross_dataset()
    if p4 is not None:
        print(f"   -> {p4.name}")

    print()
    print("[done] figures generated in", FIG_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
