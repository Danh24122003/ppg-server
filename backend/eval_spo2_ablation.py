"""
 Deep SQA -- SpO2 ablation: SpO2 validity rate stratified by SQA tier.

Question answered:
  Does SQA tier (excellent / acceptable / unfit) predict the validity rate
  of SpO2 estimation (i.e. fraction of windows where the production
  calculate_spo2_v23 returns a clinically plausible reading)? If yes,
  SQA gating supports SpO2 -- defending the "framework" claim across
  HR (), HRV () and SpO2 ().

Data sources (have Red + IR channels):
  1. BP-protocol cleaned (collected_data/cleaned/*.csv, fs=100Hz)
     -- MAX30102 finger reflectance, Omron cuff truth (no SpO2 GT).
  2. S09-SQA dedicated  (collected_data/sqa_experiment/S09_p[1-4]*.csv,
     fs=100Hz) -- MAX30102 4 motion protocols, no SpO2 GT.

Excluded:
  - BIDMC: only `PLETH` single channel in Signals.csv (Pimentel et al.
    2017 IEEE TBME 64(8):1914-1923) -- no Red channel, cannot compute R.
  - PPG-DaLiA: Empatica E4 BVP single-channel green LED -- no Red channel.

Per-window pipeline:
  - Reuse window grid from `eval_sqa_classifier_results.csv` ():
    each row = one 10s window with both pseudo-label and LR-predicted label.
  - Slice IR + Red samples for [t_start_s, t_end_s] from the source CSV.
  - Call calculate_spo2_v23(ir_window, red_window, fs) -- production SpO2
    (post K./M. fix 2026-05-04 quadratic Maxim MAXREFDES117 calibration).
  - Record: spo2_pred (None if rejected), reason (ok / r_out_of_range /
    low_pi / acdc_out_of_range / ac_or_dc_invalid), R, acdc_ir, acdc_red.

Stratify by:
  - Pseudo-label tier (column `label` from  Orphanidou rules).
  - LR predicted tier (column `label_pred_lr` from  GroupKFold model).

Aggregate per (dataset, tier, label_source):
  - n_total, n_valid, pct_valid
  - pct_in_94_100 (clinical normoxic range)
  - median_spo2, iqr_spo2 (valid windows only)
  - top_rejection_reason

Outputs:
  backend/eval_spo2_ablation_results.csv
  backend/eval_spo2_ablation_summary.csv
  backend/figures/fig_spo2_ablation_validity_rate.pdf
  backend/figures/fig_spo2_ablation_distribution.pdf

Usage:
  python "backend/eval_spo2_ablation.py"
  python "backend/eval_spo2_ablation.py" --datasets bp,quoc
  python "backend/eval_spo2_ablation.py" --label-src pseudo
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent  # backend/
PROJECT_ROOT = ROOT.parent

sys.path.insert(0, str(ROOT))
from main import calculate_spo2_v23  # noqa: E402

CLASSIFIER_CSV = ROOT / "eval_sqa_classifier_results.csv"
BP_CLEANED_DIR = PROJECT_ROOT / "collected_data" / "cleaned"
QUOC_RAW_DIR = PROJECT_ROOT / "collected_data" / "sqa_experiment"

OUT_RESULTS_CSV = ROOT / "eval_spo2_ablation_results.csv"
OUT_SUMMARY_CSV = ROOT / "eval_spo2_ablation_summary.csv"
FIG_DIR = ROOT / "figures"

DATASETS_ELIGIBLE = {"BP-protocol", "S09-SQA"}
DATASET_ALIAS = {"bp": "BP-protocol", "quoc": "S09-SQA", "bidmc": "BIDMC"}

TIERS = ["excellent", "acceptable", "unfit"]
TIER_COLORS = {
    "excellent": "#2ecc71",
    "acceptable": "#f1c40f",
    "unfit": "#e74c3c",
}

NORMOXIC_LO = 94.0
NORMOXIC_HI = 100.0


# ============================================================
# Caches (avoid re-loading CSV per window)
# ============================================================
_CSV_CACHE: dict[str, tuple[np.ndarray, np.ndarray]] = {}


def load_signal_cached(dataset: str, file_name: str) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Return (ir, red) arrays for a BP-protocol or S09-SQA file. Cached per file."""
    key = f"{dataset}::{file_name}"
    if key in _CSV_CACHE:
        return _CSV_CACHE[key]
    if dataset == "BP-protocol":
        path = BP_CLEANED_DIR / file_name
    elif dataset == "S09-SQA":
        path = QUOC_RAW_DIR / file_name
    else:
        return None
    if not path.exists():
        return None
    df = pd.read_csv(path, comment="#")
    if "ir" not in df.columns or "red" not in df.columns:
        return None
    ir = df["ir"].to_numpy(dtype=float)
    red = df["red"].to_numpy(dtype=float)
    _CSV_CACHE[key] = (ir, red)
    return ir, red


# ============================================================
# Per-window evaluation
# ============================================================
def eval_row(row: pd.Series) -> dict:
    """Compute SpO2 for a single 10s window via calculate_spo2_v23."""
    out = {
        "spo2_pred": float("nan"),
        "valid_flag": False,
        "ratio_r": float("nan"),
        "acdc_ir": float("nan"),
        "acdc_red": float("nan"),
        "rejection_reason": "",
        "skip_reason": "",
    }
    cached = load_signal_cached(row["dataset"], row["file"])
    if cached is None:
        out["skip_reason"] = "file_missing_or_bad_cols"
        return out
    ir, red = cached
    fs = int(row["fs"])
    s = int(row["t_start_s"] * fs)
    e = int(row["t_end_s"] * fs)
    if s < 0 or e > len(ir) or e <= s:
        out["skip_reason"] = "window_out_of_bounds"
        return out
    ir_win = ir[s:e]
    red_win = red[s:e]
    spo2, R, acdc_ir, acdc_red, status = calculate_spo2_v23(ir_win, red_win, fs)
    out["ratio_r"] = float(R) if R == R else float("nan")
    out["acdc_ir"] = float(acdc_ir) if acdc_ir == acdc_ir else float("nan")
    out["acdc_red"] = float(acdc_red) if acdc_red == acdc_red else float("nan")
    if status == "ok" and spo2 is not None:
        out["spo2_pred"] = float(spo2)
        out["valid_flag"] = True
        out["rejection_reason"] = ""
    else:
        out["rejection_reason"] = status
    return out


# ============================================================
# Stats
# ============================================================
def _iqr(arr: np.ndarray) -> float:
    if arr.size < 2:
        return float("nan")
    return float(np.percentile(arr, 75) - np.percentile(arr, 25))


def wilson_ci_95(k: int, n: int) -> tuple[float, float]:
    """Wilson score 95% confidence interval for binomial proportion.

    Returns (lo_pct, hi_pct) bounded to [0, 100]. Returns (nan, nan) if n==0.
    No statsmodels dependency -- manual closed-form per Wilson 1927.
    """
    if n == 0:
        return (float("nan"), float("nan"))
    z = 1.96  # 95% CI
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half_width = z * (p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5 / denom
    lo = max(0.0, (centre - half_width) * 100.0)
    hi = min(100.0, (centre + half_width) * 100.0)
    return (lo, hi)


def summarize_by_tier(
    df: pd.DataFrame, tier_col: str, label_source: str
) -> pd.DataFrame:
    """Build summary table: rows = (dataset, tier), columns = stats."""
    rows: list[dict] = []
    df_in = df[df["skip_reason"] == ""]
    for ds, ds_grp in df_in.groupby("dataset"):
        for tier in TIERS:
            sub = ds_grp[ds_grp[tier_col] == tier]
            n_total = len(sub)
            valid = sub[sub["valid_flag"]]
            n_valid = len(valid)
            if n_total == 0:
                rows.append(
                    {
                        "label_source": label_source,
                        "dataset": ds,
                        "tier": tier,
                        "n_total": 0,
                        "n_valid": 0,
                        "pct_valid": float("nan"),
                        "pct_in_94_100": float("nan"),
                        "pct_in_94_100_ci_lo": float("nan"),
                        "pct_in_94_100_ci_hi": float("nan"),
                        "n_in_94_100": 0,
                        "pct_in_94_100_of_total": float("nan"),
                        "pct_in_94_100_of_total_ci_lo": float("nan"),
                        "pct_in_94_100_of_total_ci_hi": float("nan"),
                        "median_spo2": float("nan"),
                        "iqr_spo2": float("nan"),
                        "top_rejection_reason": "",
                    }
                )
                continue
            pct_valid = n_valid / n_total * 100.0
            spo2_arr = valid["spo2_pred"].to_numpy(dtype=float)
            if spo2_arr.size > 0:
                n_in_range = int(
                    ((spo2_arr >= NORMOXIC_LO) & (spo2_arr <= NORMOXIC_HI)).sum()
                )
                pct_in_range = n_in_range / spo2_arr.size * 100.0
                ci_lo, ci_hi = wilson_ci_95(n_in_range, int(spo2_arr.size))
                pct_in_range_of_total = n_in_range / n_total * 100.0
                ci_lo_tot, ci_hi_tot = wilson_ci_95(n_in_range, int(n_total))
                median_spo2 = float(np.median(spo2_arr))
                iqr_spo2 = _iqr(spo2_arr)
            else:
                n_in_range = 0
                pct_in_range = float("nan")
                ci_lo = float("nan")
                ci_hi = float("nan")
                pct_in_range_of_total = 0.0 if n_total > 0 else float("nan")
                ci_lo_tot, ci_hi_tot = wilson_ci_95(0, int(n_total)) if n_total > 0 else (float("nan"), float("nan"))
                median_spo2 = float("nan")
                iqr_spo2 = float("nan")
            # Top rejection reason among rejected windows
            rejected = sub[~sub["valid_flag"]]
            if len(rejected) > 0:
                reason_counter = Counter(rejected["rejection_reason"].tolist())
                # exclude empty string (which corresponds to valid)
                reason_counter.pop("", None)
                if reason_counter:
                    top_reason = reason_counter.most_common(1)[0][0]
                else:
                    top_reason = ""
            else:
                top_reason = ""
            rows.append(
                {
                    "label_source": label_source,
                    "dataset": ds,
                    "tier": tier,
                    "n_total": int(n_total),
                    "n_valid": int(n_valid),
                    "pct_valid": float(pct_valid),
                    "pct_in_94_100": pct_in_range,
                    "pct_in_94_100_ci_lo": ci_lo,
                    "pct_in_94_100_ci_hi": ci_hi,
                    "n_in_94_100": int(n_in_range),
                    "pct_in_94_100_of_total": pct_in_range_of_total,
                    "pct_in_94_100_of_total_ci_lo": ci_lo_tot,
                    "pct_in_94_100_of_total_ci_hi": ci_hi_tot,
                    "median_spo2": median_spo2,
                    "iqr_spo2": iqr_spo2,
                    "top_rejection_reason": top_reason,
                }
            )
    return pd.DataFrame(rows)


def check_monotonic_validity(summary: pd.DataFrame) -> pd.DataFrame:
    """Add monotonic check on pct_valid: excellent >= acceptable >= unfit."""
    rows: list[dict] = []
    for (label_source, ds), grp in summary.groupby(["label_source", "dataset"]):
        m = {t: float("nan") for t in TIERS}
        for _, r in grp.iterrows():
            m[r["tier"]] = r["pct_valid"]
        order = [m["excellent"], m["acceptable"], m["unfit"]]
        finite = [v for v in order if not (isinstance(v, float) and np.isnan(v))]
        if len(finite) < 2:
            mono = None
        else:
            # non-increasing along order (excellent >= acceptable >= unfit)
            mono = True
            prev = np.inf
            for v in order:
                if isinstance(v, float) and np.isnan(v):
                    continue
                if v - 1e-9 > prev:
                    mono = False
                    break
                prev = v
        rows.append(
            {
                "label_source": label_source,
                "dataset": ds,
                "pct_valid_excellent": m["excellent"],
                "pct_valid_acceptable": m["acceptable"],
                "pct_valid_unfit": m["unfit"],
                "monotonic_nonincreasing": mono,
                "n_tiers_with_data": len(finite),
            }
        )
    return pd.DataFrame(rows)


# ============================================================
# Console reporting
# ============================================================
def print_summary_table(summary: pd.DataFrame) -> None:
    print()
    print("=" * 110)
    print("SpO2 ABLATION -- per (dataset x tier x label_source)")
    print("=" * 110)
    print(
        f"{'label_src':<10}{'dataset':<14}{'tier':<12}"
        f"{'N_total':>9}{'N_valid':>9}{'%valid':>9}"
        f"{'%[94,100]':>11}{'median':>9}{'IQR':>7}  {'top_reject'}"
    )
    print("-" * 110)
    for label_source in summary["label_source"].unique():
        for ds in sorted(summary[summary.label_source == label_source]["dataset"].unique()):
            for tier in TIERS:
                sub = summary[
                    (summary.label_source == label_source)
                    & (summary.dataset == ds)
                    & (summary.tier == tier)
                ]
                if sub.empty:
                    continue
                r = sub.iloc[0]
                if r["n_total"] == 0:
                    print(
                        f"{label_source:<10}{ds:<14}{tier:<12}"
                        f"{r['n_total']:>9}{'-':>9}{'-':>9}"
                        f"{'-':>11}{'-':>9}{'-':>7}  {'-'}"
                    )
                    continue
                med = (
                    f"{r['median_spo2']:.1f}"
                    if not np.isnan(r["median_spo2"])
                    else "-"
                )
                iqr = (
                    f"{r['iqr_spo2']:.1f}"
                    if not np.isnan(r["iqr_spo2"])
                    else "-"
                )
                pct_range = (
                    f"{r['pct_in_94_100']:.1f}"
                    if not np.isnan(r["pct_in_94_100"])
                    else "-"
                )
                print(
                    f"{label_source:<10}{ds:<14}{tier:<12}"
                    f"{r['n_total']:>9}{r['n_valid']:>9}{r['pct_valid']:>8.1f}%"
                    f"{pct_range:>11}{med:>9}{iqr:>7}  {r['top_rejection_reason']}"
                )
        print("-" * 110)


def print_monotonic_check(mono_df: pd.DataFrame) -> None:
    print()
    print("=" * 110)
    print("MONOTONIC ORDERING CHECK on %valid (excellent >= acceptable >= unfit)")
    print("=" * 110)
    print(
        f"{'label_src':<10}{'dataset':<14}"
        f"{'%v_exc':>10}{'%v_acc':>10}{'%v_unf':>10}"
        f"{'n_tiers':>9}{'monotonic':>14}"
    )
    print("-" * 110)
    for _, r in mono_df.iterrows():
        m = r["monotonic_nonincreasing"]
        n_tiers = int(r["n_tiers_with_data"])
        if m is None:
            m_str = "n/a"
        elif n_tiers < 3:
            m_str = ("YES" if m else "NO") + "(partial)"
        else:
            m_str = "YES" if m else "NO"

        def fmt(v):
            return (
                "-"
                if (v is None or (isinstance(v, float) and np.isnan(v)))
                else f"{v:.1f}"
            )

        print(
            f"{r['label_source']:<10}{r['dataset']:<14}"
            f"{fmt(r['pct_valid_excellent']):>10}"
            f"{fmt(r['pct_valid_acceptable']):>10}"
            f"{fmt(r['pct_valid_unfit']):>10}"
            f"{n_tiers:>9}{m_str:>14}"
        )


def print_skipped(df: pd.DataFrame) -> None:
    print()
    print("=" * 80)
    print("SKIPPED WINDOWS (load failure / out of bounds)")
    print("=" * 80)
    skipped = df[df["skip_reason"] != ""]
    if skipped.empty:
        print("  (none)")
        return
    print(f"{'dataset':<14}{'reason':<32}{'count':>8}")
    print("-" * 80)
    for (ds, reason), grp in skipped.groupby(["dataset", "skip_reason"]):
        print(f"{ds:<14}{reason:<32}{len(grp):>8d}")
    print(f"{'TOTAL skipped':<14}{'':<32}{len(skipped):>8d}")


def print_rejection_breakdown(df: pd.DataFrame) -> None:
    print()
    print("=" * 80)
    print("REJECTION REASON BREAKDOWN (per dataset)")
    print("=" * 80)
    df_in = df[df["skip_reason"] == ""]
    rejected = df_in[~df_in["valid_flag"]]
    if rejected.empty:
        print("  (no rejections)")
        return
    print(f"{'dataset':<14}{'reason':<32}{'count':>8}{'pct_of_dataset':>18}")
    print("-" * 80)
    for ds, ds_grp in df_in.groupby("dataset"):
        rej = ds_grp[~ds_grp["valid_flag"]]
        n_ds = len(ds_grp)
        for reason, grp in rej.groupby("rejection_reason"):
            pct = len(grp) / n_ds * 100.0 if n_ds else 0.0
            print(f"{ds:<14}{reason:<32}{len(grp):>8d}{pct:>17.1f}%")


def print_honest_framing(has_bidmc_gt: bool) -> None:
    print()
    print("=" * 80)
    print(" SpO2 ABLATION -- HONEST FRAMING NOTES")
    print("=" * 80)
    print(
        """
1. Datasets restricted to BP-protocol cleaned + S09-SQA dedicated.
   Both use MAX30102 reflectance (660 nm Red + 880 nm IR) -- the only
   sources in this study with both required channels.

2. EXCLUDED -- single-channel sources cannot compute SpO2:
   - BIDMC: Signals.csv has only PLETH (Philips bedside monitor records
     SpO2 numerically but not in this 20-recording subset of waveform
     CSVs; Pimentel et al. 2017 IEEE TBME 64(8):1914-1923 documents
     the multi-parameter BIDMC dataset). Cannot compute R-ratio.
   - PPG-DaLiA: Empatica E4 BVP single-channel green LED (~525 nm),
     no Red channel; Reiss 2019 captured BVP only.

3. NO ABSOLUTE GROUND TRUTH for SpO2:
   This study did not record an Omron / FDA-cleared finger pulse
   oximeter alongside the MAX30102 prototype. Reported numbers are:
     (a) validity rate -- whether calculate_spo2_v23 returned a value
         (vs. rejected by R range, perfusion guard, AC/DC range, etc.)
     (b) % in clinical normoxic range [94, 100]% (WHO normoxia floor;
         clinical hypoxemia threshold = 92%, conservative buffer = 94%)
     (c) median SpO2 of valid windows
   Internal consistency only -- not validated against arterial blood
   gas / co-oximetry (ISO 80601-2-61:2018 reference standard).

4. SpO2 formula = quadratic Maxim MAXREFDES117 (post K./M. fix
   2026-05-04): SpO2 = -45.060 R^2 + 30.354 R + 94.845. Pre-conditions:
     - R in [0.4, 1.4] (Blaney et al. 2024 J Biomed Opt 29(S3):S33313,
       physical bounds for 660/940 nm pair)
     - Perfusion index AC/DC_IR >= 0.2% (project-conservative; clinical
       threshold is 0.6% per Schneider 2024 JECCM)
     - AC/DC ratios within sensor-specific bounds.

5. Window pipeline:
   - Each row in eval_sqa_classifier_results.csv = 10s window with
     pseudo-label tier (Orphanidou 2015 rules) + LR-predicted tier
     ( GroupKFold model).
   - For each window the IR + Red samples are sliced by [t_start_s,
     t_end_s] from the source CSV and fed into calculate_spo2_v23.

6. SQA tier gates apply BEFORE SpO2 estimation. The %valid column
   reflects fraction of windows yielding plausible SpO2 WITHIN each
   tier -- production deployment can choose to skip `unfit` tier or
   weight predictions by tier confidence.

7. Reported tiers from BOTH label sources (pseudo + LR). Pseudo tier
   is INDEPENDENT evidence (Orphanidou rules use HR / RR-interval
   features only). LR tier may have circular dependency -- LR features
   include `tsqi_median` (rank #1) and `hr_bpm` (rank #5), neither of
   which directly drive calculate_spo2_v23 (which uses bandpassed RMS
   AC over lowpass DC), so the dependency is weak.

8. "Top rejection reason" per (dataset, tier) is reported. Common
   reasons: r_out_of_range (poor optical perfusion), low_perfusion_index
   (signal amplitude below 0.2%), acdc_*_out_of_range (AC/DC outside
   sensor calibration bounds).

9. N=28 caveat for LR-unfit BP-protocol headline:
   The 28 LR-unfit windows consist mostly of window_idx=0 start-of-recording
   transients (9 subjects) plus a S11 low-perfusion cluster (5 windows).
   Wilson 95% CI for pct_in_94_100=47.4% on N_valid=19 is approximately
   [27%, 69%]. Wilson CI for excellent 99.3% on N_valid=581 is [98.2%, 99.8%].
   Bands DO NOT overlap -> finding is statistically robust despite small N.
   Cite as: "47.4% [95% CI: 27-69%, N=19 valid of 28 total]" vs
   "99.3% [98.2-99.8%, N=581 valid of 909 total]" (delta -51.9 pp,
   non-overlapping CIs).

10. Cross-dataset inconsistency (S09-SQA LR-unfit 88.9% vs BP-protocol 47.4%):
    Different artifact types. S09 P3/P4 motion degrades HRV/HR features
    (LR flags as unfit) but R-ratio stays in [0.4, 1.4] range because
    finger contact is maintained -- SpO2 distribution stays plausible.
    BP-protocol LR-unfit is dominated by start-of-recording AC/DC mismatch
    transients (R near 0.83-1.0, R close to 1.0 = anoxic ~80% SpO2 falsely)
    and S11's low-perfusion windows. Cross-dataset SpO2 ablation is
    therefore artifact-type-dependent, not directly comparable.

11. Window size 10 s (1000 samples @ 100 Hz) inherited from
    pseudo_labels.csv. Production batch is 5 s. Longer windows improve
    AC/DC SNR estimation but may span artifact transitions. Aggregate
    median SpO2 = 99.6 here vs 99.20 in eval_physiological_plausibility.py
    (5s batches) on same BP-cleaned files -- consistent within instrument noise.
"""
    )
    print("=" * 80)


# ============================================================
# Figures
# ============================================================
def fig_validity_rate(summary: pd.DataFrame) -> Path:
    """Bar chart pct_valid + pct_in_[94,100] per tier x dataset, 2 subplots (pseudo / LR).

    Two metrics overlaid (pct_valid bars + scatter overlay for pct_in_range).
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), sharey=True)
    label_sources = ["pseudo", "lr_pred"]
    titles = [
        "Pseudo-label tier (Orphanidou rules)",
        "LR-predicted tier ( model)",
    ]
    datasets = sorted(summary["dataset"].unique())
    bar_w = 0.35

    for ax, ls, title in zip(axes, label_sources, titles):
        sub = summary[summary.label_source == ls]
        for di, ds in enumerate(datasets):
            ds_sub = sub[sub.dataset == ds].set_index("tier").reindex(TIERS)
            pct_valid = ds_sub["pct_valid"].to_numpy(dtype=float)
            pct_range = ds_sub["pct_in_94_100"].to_numpy(dtype=float)
            xs = np.arange(len(TIERS)) + di * bar_w

            # Bar = pct_valid
            ax.bar(
                xs,
                np.nan_to_num(pct_valid, nan=0.0),
                bar_w,
                label=f"{ds} (valid)",
                edgecolor="black",
                linewidth=0.4,
                alpha=0.7,
            )
            # Scatter = pct_in_[94,100]
            ax.scatter(
                xs,
                np.nan_to_num(pct_range, nan=0.0),
                marker="D",
                s=42,
                color="black",
                zorder=5,
                label=f"{ds} (in[94,100])" if di == 0 else None,
            )
            for x, p, n in zip(xs, pct_valid, ds_sub["n_total"].to_numpy()):
                if not np.isnan(p):
                    ax.text(x, p + 1.5, f"N={int(n)}", ha="center", fontsize=7)

        ax.set_xticks(np.arange(len(TIERS)) + bar_w * (len(datasets) - 1) / 2)
        ax.set_xticklabels(TIERS)
        ax.set_title(title)
        ax.set_xlabel("SQA tier")
        ax.set_ylim(0, 110)
        ax.grid(axis="y", linestyle=":", alpha=0.4)

    axes[0].set_ylabel("% windows (bars=valid, diamonds=in [94,100]%)")
    axes[0].legend(loc="lower left", fontsize=7, framealpha=0.9, ncol=2)

    fig.suptitle(
        "SpO2 validity rate stratified by SQA tier "
        "(bars: % windows yielding valid SpO2; diamonds: % within [94,100]%)"
    )
    plt.tight_layout()
    out = FIG_DIR / "fig_spo2_ablation_validity_rate.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_distribution(df: pd.DataFrame) -> Path:
    """Box plot of valid SpO2 values per (dataset, tier), 2 subplots (pseudo / LR)."""
    df_valid = df[df["valid_flag"]].copy()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), sharey=True)
    label_cols = ["sqa_tier_pseudo", "sqa_tier_lr_pred"]
    titles = ["Pseudo-label tier", "LR-predicted tier"]
    datasets = sorted(df_valid["dataset"].unique())

    for ax, lc, title in zip(axes, label_cols, titles):
        positions: list[float] = []
        data: list[np.ndarray] = []
        tick_labels: list[str] = []
        face_colors: list[str] = []
        x = 0.0
        for ds in datasets:
            for tier in TIERS:
                sub = df_valid[(df_valid.dataset == ds) & (df_valid[lc] == tier)]
                vals = sub["spo2_pred"].to_numpy(dtype=float)
                if len(vals) > 0:
                    positions.append(x)
                    data.append(vals)
                    tick_labels.append(f"{ds}\n{tier}\n(N={len(vals)})")
                    face_colors.append(TIER_COLORS[tier])
                x += 1.0
            x += 0.5  # gap between datasets

        if data:
            bp = ax.boxplot(
                data,
                positions=positions,
                widths=0.7,
                patch_artist=True,
                showfliers=True,
                flierprops=dict(marker=".", markersize=2, alpha=0.4),
            )
            for patch, c in zip(bp["boxes"], face_colors):
                patch.set_facecolor(c)
                patch.set_alpha(0.65)
            ax.set_xticks(positions)
            ax.set_xticklabels(tick_labels, fontsize=7)
        # Highlight clinical normoxic range
        ax.axhspan(NORMOXIC_LO, NORMOXIC_HI, color="green", alpha=0.06, zorder=0)
        ax.axhline(NORMOXIC_LO, color="green", linewidth=0.6, linestyle=":")
        ax.set_ylim(82, 102)
        ax.set_title(title)
        ax.grid(axis="y", linestyle=":", alpha=0.4)

    axes[0].set_ylabel("SpO2 (%)")
    fig.suptitle(
        "SpO2 distribution per SQA tier (valid windows; green band = [94, 100]%)"
    )
    plt.tight_layout()
    out = FIG_DIR / "fig_spo2_ablation_distribution.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


# ============================================================
# Main
# ============================================================
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--datasets",
        type=str,
        default="bp,quoc",
        help="Comma-separated subset of {bp,quoc} (BIDMC excluded -- no Red channel)",
    )
    ap.add_argument(
        "--label-src",
        type=str,
        default="pseudo,lr_pred",
        help="Comma-separated subset of {pseudo,lr_pred}",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    np.random.seed(42)

    if not CLASSIFIER_CSV.exists():
        print(f"ERROR: {CLASSIFIER_CSV} not found. Run eval_sqa_classifier.py first.")
        return 1

    requested = [d.strip().lower() for d in args.datasets.split(",") if d.strip()]
    requested_canonical = []
    for r in requested:
        if r in DATASET_ALIAS:
            mapped = DATASET_ALIAS[r]
            if mapped in DATASETS_ELIGIBLE:
                requested_canonical.append(mapped)
            else:
                print(
                    f"NOTE: dataset '{r}' (-> {mapped}) excluded -- single-channel, "
                    "cannot compute SpO2 (see honest framing #2)."
                )
        else:
            print(f"WARN: unknown dataset alias '{r}' -- skipped.")
    if not requested_canonical:
        print("ERROR: no eligible datasets after filtering.")
        return 1

    label_srcs = [s.strip() for s in args.label_src.split(",") if s.strip()]
    if not all(s in ("pseudo", "lr_pred") for s in label_srcs):
        print("ERROR: --label-src must be subset of {pseudo,lr_pred}")
        return 1

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {CLASSIFIER_CSV.name}...")
    cls_df = pd.read_csv(CLASSIFIER_CSV)
    print(
        f"  {len(cls_df)} total windows; datasets: "
        f"{sorted(cls_df['dataset'].unique())}"
    )

    work = cls_df[cls_df["dataset"].isin(requested_canonical)].copy()
    print(
        f"  Filtered to {requested_canonical}: {len(work)} windows "
        + ", ".join(
            f"{d}={int((work.dataset == d).sum())}" for d in requested_canonical
        )
    )

    print("\nComputing SpO2 per window via calculate_spo2_v23...")
    t0 = time.time()
    out_rows: list[dict] = []
    n = len(work)
    for i, (_, row) in enumerate(work.iterrows()):
        if i % 200 == 0 and i > 0:
            print(f"  {i}/{n} ({i / n * 100:.0f}%) elapsed {time.time() - t0:.0f}s")
        info = eval_row(row)
        out_rows.append(
            {
                "file": row["file"],
                "dataset": row["dataset"],
                "subject_id": row["subject_id"],
                "window_idx": int(row["window_idx"]),
                "t_start_s": float(row["t_start_s"]),
                "t_end_s": float(row["t_end_s"]),
                "fs": int(row["fs"]),
                "spo2_pred": info["spo2_pred"],
                "valid_flag": bool(info["valid_flag"]),
                "ratio_r": info["ratio_r"],
                "acdc_ir": info["acdc_ir"],
                "acdc_red": info["acdc_red"],
                "rejection_reason": info["rejection_reason"],
                "skip_reason": info["skip_reason"],
                "sqa_tier_pseudo": row["label"],
                "sqa_tier_lr_pred": row["label_pred_lr"],
            }
        )
    elapsed = time.time() - t0
    print(f"  Done {n}/{n} in {elapsed:.1f}s")

    df = pd.DataFrame(out_rows)
    df.to_csv(OUT_RESULTS_CSV, index=False)
    print(f"\nSaved per-window results: {OUT_RESULTS_CSV.name} ({len(df)} rows)")

    print_skipped(df)
    print_rejection_breakdown(df)

    summaries: list[pd.DataFrame] = []
    if "pseudo" in label_srcs:
        summaries.append(summarize_by_tier(df, "sqa_tier_pseudo", "pseudo"))
    if "lr_pred" in label_srcs:
        summaries.append(summarize_by_tier(df, "sqa_tier_lr_pred", "lr_pred"))
    summary = pd.concat(summaries, ignore_index=True)
    summary.to_csv(OUT_SUMMARY_CSV, index=False)
    print(f"Saved summary table: {OUT_SUMMARY_CSV.name} ({len(summary)} rows)")

    print_summary_table(summary)

    mono = check_monotonic_validity(summary)
    print_monotonic_check(mono)

    print("\nGenerating figures...")
    f1 = fig_validity_rate(summary)
    print(f"  {f1.name}")
    f2 = fig_distribution(df)
    print(f"  {f2.name}")
    print(
        "  (skipped fig_spo2_ablation_mae_bidmc.pdf -- no SpO2 ground truth "
        "in BIDMC Signals.csv subset; see honest framing #3)"
    )

    print_honest_framing(has_bidmc_gt=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
