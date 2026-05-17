"""Generate thesis-defense-grade summary figure for HRV ablation.

Reads ``eval_hrv_ablation_summary.csv`` and renders a 2-panel headline
figure (5-method MAE comparison + component decomposition) for thesis
Chapter 5 / defense slide.

Standalone script — does NOT modify ``eval_hrv_ablation.py`` or any
production code.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
SUMMARY_CSV = HERE / "eval_hrv_ablation_summary.csv"
OUT_PDF = HERE / "figures" / "fig_hrv_ablation_thesis_summary.pdf"

DATASET = "PPG-DaLiA"
TIER = "unfit"
LABEL_SOURCE = "pseudo"
COMPARISON_MODE = "matched"
METRIC = "sdnn"

METHOD_ORDER = [
    "A_no_filter",
    "B_tsqi_naive",
    "C_tsqi_literature",
    "D_heartpy_2stage",
    "E_combined",
]

METHOD_LABELS = {
    "A_no_filter": "A\n(no filter)",
    "B_tsqi_naive": "B\n(naive)",
    "C_tsqi_literature": "C\n(tSQI)",
    "D_heartpy_2stage": "D\n(production)",
    "E_combined": "E\n(SQA stack)",
}

METHOD_COLORS = {
    "A_no_filter": "#888888",
    "B_tsqi_naive": "#d62728",
    "C_tsqi_literature": "#ffbf00",
    "D_heartpy_2stage": "#2ca02c",
    "E_combined": "#1f77b4",
}


def load_mae() -> dict[str, tuple[float, int]]:
    """Return {method: (mae_mean, n_windows)} for the filtered slice."""
    found: dict[str, tuple[float, int]] = {}
    with SUMMARY_CSV.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (
                row["label_source"] == LABEL_SOURCE
                and row["dataset"] == DATASET
                and row["tier"] == TIER
                and row["metric"] == METRIC
                and row["comparison_mode"] == COMPARISON_MODE
                and row["method"] in METHOD_ORDER
            ):
                found[row["method"]] = (
                    float(row["mae_mean"]),
                    int(row["n_windows"]),
                )

    missing = [m for m in METHOD_ORDER if m not in found]
    if missing:
        raise RuntimeError(
            f"Missing rows for methods {missing} with filter "
            f"label_source={LABEL_SOURCE} dataset={DATASET} tier={TIER} "
            f"metric={METRIC} comparison_mode={COMPARISON_MODE}"
        )
    return found


def main() -> int:
    if not SUMMARY_CSV.exists():
        print(f"ERROR: missing {SUMMARY_CSV}", file=sys.stderr)
        return 1

    mae = load_mae()

    print("=" * 60)
    print(f"HRV ablation 5-method summary ({DATASET}, {TIER}, {COMPARISON_MODE})")
    print("=" * 60)
    n_set = {n for _, n in mae.values()}
    print(f"N windows (matched): {sorted(n_set)}")
    for m in METHOD_ORDER:
        v, n = mae[m]
        print(f"  {m:22s}  MAE_SDNN = {v:7.2f} ms  (N={n})")
    print("=" * 60)

    a = mae["A_no_filter"][0]
    b = mae["B_tsqi_naive"][0]
    c = mae["C_tsqi_literature"][0]
    d = mae["D_heartpy_2stage"][0]
    e = mae["E_combined"][0]
    n_matched = mae["A_no_filter"][1]

    def improvement_pct(method_val: float) -> float:
        return (a - method_val) / a * 100.0

    annotations = {
        "A_no_filter": "(baseline)",
        "B_tsqi_naive": f"{improvement_pct(b):+.0f}%",
        "C_tsqi_literature": f"{improvement_pct(c):+.0f}%",
        "D_heartpy_2stage": f"{improvement_pct(d):+.0f}%",
        "E_combined": f"{improvement_pct(e):+.0f}%",
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Panel 1: MAE bars
    xs = list(range(len(METHOD_ORDER)))
    ys = [mae[m][0] for m in METHOD_ORDER]
    colors = [METHOD_COLORS[m] for m in METHOD_ORDER]
    edge_widths = [2.5 if m == "E_combined" else 0.8 for m in METHOD_ORDER]
    bars = ax1.bar(
        xs, ys, color=colors, edgecolor="black", linewidth=edge_widths
    )
    ax1.set_xticks(xs)
    ax1.set_xticklabels([METHOD_LABELS[m] for m in METHOD_ORDER], fontsize=9)
    ax1.set_ylabel("MAE_SDNN (ms)", fontsize=10)
    ax1.set_title(
        f"HRV SDNN MAE - 5-Method Comparison\n"
        f"({DATASET} {TIER}, matched N={n_matched})",
        fontsize=11,
    )
    ax1.axhline(a, color="grey", linestyle=":", linewidth=1.0, alpha=0.7)
    ax1.text(
        len(xs) - 0.4,
        a + 10,
        "A baseline",
        fontsize=8,
        color="grey",
        style="italic",
    )

    y_max = max(ys) * 1.15
    ax1.set_ylim(0, y_max)
    for bar, m in zip(bars, METHOD_ORDER):
        h = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            h + y_max * 0.015,
            f"{h:.1f}\n{annotations[m]}",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold" if m == "E_combined" else "normal",
        )

    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.grid(axis="y", linestyle="--", alpha=0.3)

    # Panel 2: decomposition
    decomp_labels = [
        "tSQI alone\n(C - A)",
        "RR-domain alone\n(D - A)",
        "tSQI + RR-domain\n(E - A)",
        "Marginal tSQI\n(E - D)",
    ]
    decomp_values = [
        improvement_pct(c),
        improvement_pct(d),
        improvement_pct(e),
        (d - e) / a * 100.0,
    ]
    decomp_colors = ["#ffbf00", "#2ca02c", "#1f77b4", "#6a3d9a"]
    decomp_edges = [0.8, 0.8, 2.5, 0.8]

    xs2 = list(range(len(decomp_labels)))
    bars2 = ax2.bar(
        xs2,
        decomp_values,
        color=decomp_colors,
        edgecolor="black",
        linewidth=decomp_edges,
    )
    ax2.set_xticks(xs2)
    ax2.set_xticklabels(decomp_labels, fontsize=9)
    ax2.set_ylabel("Improvement vs Method A (%)", fontsize=10)
    ax2.set_title("Component Contribution Decomposition", fontsize=11)
    ax2.set_ylim(0, max(decomp_values) * 1.18)

    for bar, v in zip(bars2, decomp_values):
        h = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            h + max(decomp_values) * 0.02,
            f"{v:.0f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.grid(axis="y", linestyle="--", alpha=0.3)

    caption = (
        "Method A: no HRV filter (baseline). "
        "Method B: naive np.diff(kept_peaks) after tSQI hard threshold "
        "( strawman, gap-spanning RR bug). "
        "Method C: HeartPy van Gent 2019 vectorized mask "
        "(accepted[:-1] & accepted[1:]) per heartpy/analysis.py:293. "
        "Method D: production HeartPy 2-stage RR rejection "
        "(Task Force 1996 + +/-20% median quotient), mirrors "
        "backend/main.py:_reject_rr_outliers. "
        "Method E: combined C + Stage 2 quotient. "
        f"Matched comparison N={n_matched} (windows where all 5 methods "
        "produced valid output). "
        "Per-method-success N=7061 gives MAE_A 146.43 ms for full-population reference."
    )
    fig.text(
        0.5,
        -0.02,
        caption,
        ha="center",
        va="top",
        fontsize=8.5,
        style="italic",
        wrap=True,
    )

    plt.tight_layout()
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUT_PDF}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
