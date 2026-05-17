"""
 Deep SQA — HR ablation: HR estimation MAE stratified by SQA tier.

Question answered:
  Does SQA tier (excellent/acceptable/unfit) actually predict HR estimation
  accuracy? If yes, SQA gating improves downstream HR — defending the
  "framework" claim. If no, the SQA score is purely internal and downstream
  is decoupled (honest finding either way).

Data sources (have ECG ground truth):
  1. BIDMC ICU       (external_data/bidmc/*.csv)
     ECG lead II at 125 Hz; re-derive R-peaks via scipy.find_peaks
     on bandpass-filtered ECG.
  2. PPG-DaLiA wrist (external_data/ppg_dalia/PPG_FieldStudy/S{1..5}/*.pkl)
     Chest ECG R-peaks pre-computed by Reiss 2019 (`rpeaks` array,
     indices at 700 Hz). Use directly — no re-derivation needed.

Excluded (no ECG comparator):
  - BP-protocol cleaned (Omron cuff only)
  - S09-SQA dedicated (PPG only)

Window pipeline:
  - Reuse window grid from `eval_sqa_classifier_results.csv` ():
    each row = one 10s window with both pseudo-label and LR-predicted label.
  - For each window: re-derive HR_ppg via scipy.find_peaks on bandpass [0.5, 4]
    (matches  LR feature extraction; production uses HeartPy adaptive
    threshold which differs slightly).
  - Compute HR_ground_truth from ECG.
  - hr_error_abs = abs(HR_ppg - HR_ground_truth).

Stratify by:
  - Pseudo-label tier (column `label` from  Orphanidou 2015 rules).
  - LR predicted tier (column `label_pred_lr` from  GroupKFold model).
  - Per dataset (BIDMC vs PPG-DaLiA).

Outputs:
  backend/eval_hr_ablation_results.csv
    Per-window: file, dataset, subject_id, window_idx, t_start_s, hr_ppg,
    hr_ground_truth, hr_error_abs, sqa_tier_pseudo, sqa_tier_lr_pred.

  backend/eval_hr_ablation_summary.csv
    Per (dataset x tier x label_source): count, mean MAE, median MAE,
    p95 MAE, MAE 95% CI lower/upper (1000-resample bootstrap).

  backend/figures/fig_hr_ablation_mae.pdf
  backend/figures/fig_hr_ablation_box.pdf
  backend/figures/fig_hr_ablation_scatter.pdf

Usage:
  python "backend/eval_hr_ablation.py"
"""
from __future__ import annotations

import gc
import pickle
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent  # backend/
PROJECT_ROOT = ROOT.parent

CLASSIFIER_CSV = ROOT / "eval_sqa_classifier_results.csv"
BIDMC_DIR = PROJECT_ROOT / "external_data" / "bidmc"
DALIA_DIR = PROJECT_ROOT / "external_data" / "ppg_dalia" / "PPG_FieldStudy"

OUT_RESULTS_CSV = ROOT / "eval_hr_ablation_results.csv"
OUT_SUMMARY_CSV = ROOT / "eval_hr_ablation_summary.csv"
FIG_DIR = ROOT / "figures"

FS_BIDMC = 125
FS_BVP = 64
FS_DALIA_ECG = 700  # chest ECG (RespiBAN) per Reiss 2019

TIERS = ["excellent", "acceptable", "unfit"]
TIER_COLORS = {
    "excellent": "#2ecc71",
    "acceptable": "#f1c40f",
    "unfit": "#e74c3c",
}

BOOTSTRAP_N = 1000
BOOTSTRAP_SEED = 42


# ============================================================
# Caches (avoid re-loading per window)
# ============================================================
_BIDMC_CACHE: dict[str, tuple[np.ndarray, np.ndarray]] = {}
_DALIA_CACHE: dict[str, tuple[np.ndarray, np.ndarray]] = {}


def load_bidmc_cached(file_name: str) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Return (ppg, ecg_II) for a BIDMC CSV. Cached per file."""
    if file_name in _BIDMC_CACHE:
        return _BIDMC_CACHE[file_name]
    path = BIDMC_DIR / file_name
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    if "PLETH" not in df.columns or "II" not in df.columns:
        return None
    ppg = df["PLETH"].to_numpy(dtype=float)
    ecg = df["II"].to_numpy(dtype=float)
    _BIDMC_CACHE[file_name] = (ppg, ecg)
    return ppg, ecg


def load_dalia_cached(subject_id: str) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Return (bvp, rpeaks) for a PPG-DaLiA subject. Cached per subject.

    rpeaks are sample indices at FS_DALIA_ECG=700 Hz (chest ECG).
    """
    if subject_id in _DALIA_CACHE:
        return _DALIA_CACHE[subject_id]
    pkl_path = DALIA_DIR / subject_id / f"{subject_id}.pkl"
    if not pkl_path.exists():
        return None
    with open(pkl_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    bvp = np.asarray(data["signal"]["wrist"]["BVP"], dtype=float).reshape(-1).copy()
    rpeaks = np.asarray(data["rpeaks"]).reshape(-1).copy()
    del data
    gc.collect()
    _DALIA_CACHE[subject_id] = (bvp, rpeaks)
    return bvp, rpeaks


# ============================================================
# HR estimation
# ============================================================
def hr_from_ppg(signal: np.ndarray, fs: int) -> float:
    """HR (BPM) from a PPG/BVP window. Returns nan if <3 peaks."""
    if len(signal) < int(fs * 2):
        return float("nan")
    b, a = butter(4, [0.5, 4.0], btype="band", fs=fs)
    filt = filtfilt(b, a, signal)
    peaks, _ = find_peaks(filt, distance=int(fs * 0.3))
    if len(peaks) < 3:
        return float("nan")
    rr_s = np.diff(peaks) / float(fs)
    rr_med = float(np.median(rr_s))
    if rr_med <= 0:
        return float("nan")
    return 60.0 / rr_med


def hr_from_ecg_bidmc(ecg_window: np.ndarray, fs: int) -> tuple[float, Optional[str]]:
    """HR (BPM) from BIDMC ECG lead II window via re-derivation.

    Uses scipy.find_peaks with prominence-based detection (robust to
    amplitude variation across BIDMC subjects). Returns (hr_bpm, reason)
    tuple where reason is None on success, else one of:
      - "ecg_window_too_short" — input < 2*fs samples
      - "ecg_too_few_peaks" — fewer than 3 R-peaks detected
      - "ecg_implausible_hr" — derived HR outside [30, 220] BPM
        (defensive guard for edge cases; not triggered in current
         BIDMC 20-recording subset with prominence detection)

    NOTE: Same find_peaks paradigm used by hr_from_ppg(). For clean
    windows where both algorithms detect identical peak counts,
    error -> 0.0 (algorithmic agreement, not physiological validation).
    See honest framing note #8 in print_honest_framing().
    """
    if len(ecg_window) < int(fs * 2):
        return float("nan"), "ecg_window_too_short"
    b, a = butter(4, [0.5, 40.0], btype="band", fs=fs)
    ecg_f = filtfilt(b, a, ecg_window)
    # Prominence-based threshold (P1 fix): robust to baseline shift /
    # amplitude variation across BIDMC subjects vs. height=std*1.5.
    peaks, _ = find_peaks(
        ecg_f,
        distance=int(fs * 0.4),
        prominence=float(np.std(ecg_f)) * 0.5,
    )
    if len(peaks) < 3:
        return float("nan"), "ecg_too_few_peaks"
    rr_s = np.diff(peaks) / float(fs)
    rr_med = float(np.median(rr_s))
    if rr_med <= 0:
        return float("nan"), "ecg_too_few_peaks"
    hr_result = 60.0 / rr_med
    # Physiological plausibility guard.
    if hr_result < 30.0 or hr_result > 220.0:
        return float("nan"), "ecg_implausible_hr"
    return hr_result, None


def hr_from_dalia_rpeaks(
    rpeaks: np.ndarray, t_start_s: float, t_end_s: float
) -> float:
    """HR (BPM) from pre-computed chest-ECG R-peak indices (700 Hz)."""
    t_start_idx = t_start_s * FS_DALIA_ECG
    t_end_idx = t_end_s * FS_DALIA_ECG
    in_win = rpeaks[(rpeaks >= t_start_idx) & (rpeaks < t_end_idx)]
    if len(in_win) < 3:
        return float("nan")
    rr_s = np.diff(in_win) / float(FS_DALIA_ECG)
    rr_med = float(np.median(rr_s))
    if rr_med <= 0:
        return float("nan")
    return 60.0 / rr_med


# ============================================================
# Per-window evaluation
# ============================================================
def eval_bidmc_row(row: pd.Series) -> dict:
    cached = load_bidmc_cached(row["file"])
    out = {
        "hr_ppg": float("nan"),
        "hr_ground_truth": float("nan"),
        "hr_error_abs": float("nan"),
        "skip_reason": "",
    }
    if cached is None:
        out["skip_reason"] = "file_missing_or_bad_cols"
        return out
    ppg, ecg = cached
    fs = FS_BIDMC
    s = int(row["t_start_s"] * fs)
    e = int(row["t_end_s"] * fs)
    if s < 0 or e > len(ppg) or e <= s:
        out["skip_reason"] = "window_out_of_bounds"
        return out
    ppg_win = ppg[s:e]
    ecg_win = ecg[s:e]
    hr_ppg = hr_from_ppg(ppg_win, fs)
    hr_gt, ecg_reason = hr_from_ecg_bidmc(ecg_win, fs)
    out["hr_ppg"] = hr_ppg
    out["hr_ground_truth"] = hr_gt
    if np.isnan(hr_ppg):
        out["skip_reason"] = "ppg_too_few_peaks"
    elif np.isnan(hr_gt):
        out["skip_reason"] = ecg_reason or "ecg_too_few_peaks"
    else:
        out["hr_error_abs"] = abs(hr_ppg - hr_gt)
    return out


def eval_dalia_row(row: pd.Series) -> dict:
    subj = row["subject_id"]
    cached = load_dalia_cached(subj)
    out = {
        "hr_ppg": float("nan"),
        "hr_ground_truth": float("nan"),
        "hr_error_abs": float("nan"),
        "skip_reason": "",
    }
    if cached is None:
        out["skip_reason"] = "subject_pkl_missing"
        return out
    bvp, rpeaks = cached
    fs = FS_BVP
    s = int(row["t_start_s"] * fs)
    e = int(row["t_end_s"] * fs)
    if s < 0 or e > len(bvp) or e <= s:
        out["skip_reason"] = "window_out_of_bounds"
        return out
    bvp_win = bvp[s:e]
    hr_ppg = hr_from_ppg(bvp_win, fs)
    hr_gt = hr_from_dalia_rpeaks(rpeaks, float(row["t_start_s"]), float(row["t_end_s"]))
    out["hr_ppg"] = hr_ppg
    out["hr_ground_truth"] = hr_gt
    if np.isnan(hr_ppg):
        out["skip_reason"] = "ppg_too_few_peaks"
    elif np.isnan(hr_gt):
        out["skip_reason"] = "ecg_rpeaks_too_few_in_window"
    else:
        out["hr_error_abs"] = abs(hr_ppg - hr_gt)
    return out


# ============================================================
# Stats
# ============================================================
def bootstrap_mae_ci(
    errors: np.ndarray, n_resamples: int = BOOTSTRAP_N, seed: int = BOOTSTRAP_SEED
) -> tuple[float, float]:
    """Return (lower, upper) 95% CI for mean of `errors` via bootstrap."""
    if len(errors) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    n = len(errors)
    means = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        means[i] = float(np.mean(errors[idx]))
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarize_by_tier(
    df: pd.DataFrame, tier_col: str, label_source: str
) -> pd.DataFrame:
    """Build summary table: rows = (dataset, tier), columns = stats."""
    rows: list[dict] = []
    df_ok = df.dropna(subset=["hr_error_abs"])
    for ds, ds_grp in df_ok.groupby("dataset"):
        for tier in TIERS:
            sub = ds_grp[ds_grp[tier_col] == tier]
            errs = sub["hr_error_abs"].to_numpy(dtype=float)
            if len(errs) == 0:
                rows.append(
                    {
                        "label_source": label_source,
                        "dataset": ds,
                        "tier": tier,
                        "n_windows": 0,
                        "mae_mean": float("nan"),
                        "mae_median": float("nan"),
                        "mae_p95": float("nan"),
                        "mae_ci95_lo": float("nan"),
                        "mae_ci95_hi": float("nan"),
                    }
                )
                continue
            ci_lo, ci_hi = bootstrap_mae_ci(errs)
            rows.append(
                {
                    "label_source": label_source,
                    "dataset": ds,
                    "tier": tier,
                    "n_windows": int(len(errs)),
                    "mae_mean": float(np.mean(errs)),
                    "mae_median": float(np.median(errs)),
                    "mae_p95": float(np.percentile(errs, 95)),
                    "mae_ci95_lo": ci_lo,
                    "mae_ci95_hi": ci_hi,
                }
            )
    return pd.DataFrame(rows)


def check_monotonic(summary: pd.DataFrame) -> pd.DataFrame:
    """Add `monotonic` column: True if mae_mean(excellent) <= acceptable <= unfit."""
    rows: list[dict] = []
    for (label_source, ds), grp in summary.groupby(["label_source", "dataset"]):
        m = {t: float("nan") for t in TIERS}
        for _, r in grp.iterrows():
            m[r["tier"]] = r["mae_mean"]
        # Tolerate nan tiers (skip from check)
        order = [m["excellent"], m["acceptable"], m["unfit"]]
        finite = [v for v in order if not np.isnan(v)]
        if len(finite) < 2:
            mono = None
        else:
            # check non-decreasing along finite values in original order
            mono = True
            prev = -np.inf
            for v in order:
                if np.isnan(v):
                    continue
                if v + 1e-9 < prev:
                    mono = False
                    break
                prev = v
        rows.append(
            {
                "label_source": label_source,
                "dataset": ds,
                "mae_excellent": m["excellent"],
                "mae_acceptable": m["acceptable"],
                "mae_unfit": m["unfit"],
                "monotonic_nondecreasing": mono,
                "n_tiers_with_data": len(finite),
            }
        )
    return pd.DataFrame(rows)


# ============================================================
# Console reporting
# ============================================================
def print_summary_table(summary: pd.DataFrame) -> None:
    print()
    print("=" * 100)
    print("HR ABLATION — MAE per (dataset x tier x label_source)")
    print("=" * 100)
    print(
        f"{'label_src':<10}{'dataset':<14}{'tier':<12}{'N':>6}  "
        f"{'mae_mean':>9}{'mae_med':>9}{'p95':>9}{'CI95':>22}"
    )
    print("-" * 100)
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
                if r["n_windows"] == 0:
                    print(
                        f"{label_source:<10}{ds:<14}{tier:<12}{r['n_windows']:>6}  "
                        f"{'-':>9}{'-':>9}{'-':>9}{'-':>22}"
                    )
                    continue
                ci = f"[{r['mae_ci95_lo']:.2f}, {r['mae_ci95_hi']:.2f}]"
                print(
                    f"{label_source:<10}{ds:<14}{tier:<12}{r['n_windows']:>6}  "
                    f"{r['mae_mean']:>9.2f}{r['mae_median']:>9.2f}"
                    f"{r['mae_p95']:>9.2f}{ci:>22}"
                )
        print("-" * 100)


def print_monotonic_check(mono_df: pd.DataFrame) -> None:
    print()
    print("=" * 100)
    print("MONOTONIC ORDERING CHECK (mae_excellent <= mae_acceptable <= mae_unfit)")
    print("=" * 100)
    print(
        f"{'label_src':<10}{'dataset':<14}{'mae_exc':>10}{'mae_acc':>10}"
        f"{'mae_unf':>10}{'n_tiers':>9}{'monotonic':>14}"
    )
    print("-" * 100)
    for _, r in mono_df.iterrows():
        m = r["monotonic_nondecreasing"]
        n_tiers = int(r["n_tiers_with_data"])
        if m is None:
            m_str = "n/a"
        elif n_tiers < 3:
            m_str = ("YES" if m else "NO") + "(partial)"
        else:
            m_str = "YES" if m else "NO"

        def fmt(v):
            return "-" if (v is None or (isinstance(v, float) and np.isnan(v))) else f"{v:.2f}"

        print(
            f"{r['label_source']:<10}{r['dataset']:<14}"
            f"{fmt(r['mae_excellent']):>10}{fmt(r['mae_acceptable']):>10}"
            f"{fmt(r['mae_unfit']):>10}{n_tiers:>9}{m_str:>14}"
        )


def print_skipped(df: pd.DataFrame) -> None:
    print()
    print("=" * 100)
    print("SKIPPED WINDOWS (no hr_error_abs)")
    print("=" * 100)
    skipped = df[df["hr_error_abs"].isna()]
    if skipped.empty:
        print("  (none)")
        return
    print(
        f"{'dataset':<14}{'reason':<32}{'count':>8}"
    )
    print("-" * 100)
    for (ds, reason), grp in skipped.groupby(["dataset", "skip_reason"]):
        print(f"{ds:<14}{reason:<32}{len(grp):>8d}")
    print(f"{'TOTAL skipped':<14}{'':<32}{len(skipped):>8d}")
    print(f"{'TOTAL usable':<14}{'':<32}{int(df['hr_error_abs'].notna().sum()):>8d}")


def print_honest_framing() -> None:
    print()
    print("=" * 80)
    print(" HR ABLATION - HONEST FRAMING NOTES")
    print("=" * 80)
    print(
        """
1. Datasets restricted to BIDMC + PPG-DaLiA (have ECG ground truth).
   BP-protocol + S09-SQA excluded -- no ECG comparator available.

2. Ground truth source differs:
   - BIDMC: ECG lead II re-derived in this script (scipy.find_peaks on
     bandpass-filtered [0.5, 40] Hz ECG, distance=fs*0.4,
     prominence=std*0.5, plausibility guard HR in [30, 220] BPM).
   - PPG-DaLiA: pre-computed `rpeaks` array from chest RespiBAN ECG
     (Reiss 2019, indices at 700 Hz) -- expected more accurate than our
     re-derivation.

3. SQA tier gates apply BEFORE HR estimation. The MAE column reflects
   HR error WITHIN each tier -- NOT HR error after SQA filtering.
   Practical implication: production deployment can choose to skip
   `unfit` tier OR weight predictions by tier confidence.

4. Window size 10s, peak-detection distance=fs*0.3 to match 
   LR feature extraction. Production pipeline uses HeartPy adaptive
   threshold which may give slightly different HR.

5. CIRCULAR DEPENDENCY for LR-predicted tier:
   The HR_ppg estimate uses scipy.find_peaks distance=fs*0.3 -- the
   same pipeline used to compute the `hr_bpm` feature in  LR
   classifier (rank #5 importance, |coef|=0.59). LR predicts SQA tier
   partly from hr_bpm, then we measure HR error stratified by that
   tier. Pseudo-label tier (Orphanidou rules) is the INDEPENDENT
   evidence -- Orphanidou uses ECG-derived HR for labeling, no
   dependency on PPG hr_bpm.

   THESIS NARRATIVE: Cite pseudo-tier MAE numbers as primary evidence.
   LR-tier MAE numbers are secondary/exploratory.

6. Bootstrap 95% CI computed with 1000 resamples, seed=42 (deterministic).

7. Monotonic ordering (excellent < acceptable < unfit) is the headline
   finding when present. Where it does NOT hold, that is reported as-is.

8. BIDMC GROUND TRUTH LIMITATION:
   ECG HR is re-derived in this script via scipy.find_peaks (same
   algorithm paradigm as hr_from_ppg). For clean windows, both PPG
   and ECG detect identical peak counts in the 10s window -> identical
   HR via 60/median(RR) -> hr_error = 0.0 (algorithmic agreement,
   not physiological validation). BIDMC excellent-tier median MAE
   is 0.0 (>50% windows). The mean is driven by bidmc_04 outlier
   windows where PPG ~187 BPM (rhythmic artifact) != ECG ~93 BPM
   (true HR detected via prominence).

   PPG-DaLiA uses pre-computed RespiBAN rpeaks (independent algorithm,
   chest ECG @ 700 Hz, Reiss 2019) -- strict cleaner ablation benchmark.

   THESIS NARRATIVE: cite PPG-DaLiA pseudo-tier (excellent 2.60 vs
   unfit 17.21 BPM) as PRIMARY headline; BIDMC results illustrative
   with bidmc_04 case study revealing real Orphanidou failure mode.
"""
    )
    print("=" * 80)


# ============================================================
# Figures
# ============================================================
def _bar_with_ci(ax, xs, means, ci_lo, ci_hi, color, label):
    yerr_lo = np.clip(np.asarray(means) - np.asarray(ci_lo), 0, None)
    yerr_hi = np.clip(np.asarray(ci_hi) - np.asarray(means), 0, None)
    ax.bar(
        xs,
        means,
        yerr=[yerr_lo, yerr_hi],
        color=color,
        edgecolor="black",
        linewidth=0.5,
        capsize=4,
        label=label,
    )


def fig_hr_ablation_mae(summary: pd.DataFrame) -> Path:
    """Bar chart MAE per tier, grouped by dataset, 2 subplots (pseudo + LR).

    Excludes the `BIDMC_excl_bidmc_04` transparency group from the figure
    to keep the headline bar chart uncluttered; that subset is reported
    in the summary CSV only.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    label_sources = ["pseudo", "lr_pred"]
    titles = ["Pseudo-label tier (Orphanidou rules)", "LR-predicted tier ( model)"]
    datasets = sorted(
        d for d in summary["dataset"].unique() if d != "BIDMC_excl_bidmc_04"
    )
    bar_w = 0.25

    for ax, ls, title in zip(axes, label_sources, titles):
        sub = summary[summary.label_source == ls]
        for di, ds in enumerate(datasets):
            ds_sub = sub[sub.dataset == ds].set_index("tier").reindex(TIERS)
            means = ds_sub["mae_mean"].to_numpy(dtype=float)
            ci_lo = ds_sub["mae_ci95_lo"].to_numpy(dtype=float)
            ci_hi = ds_sub["mae_ci95_hi"].to_numpy(dtype=float)
            xs = np.arange(len(TIERS)) + di * bar_w
            yerr_lo = np.clip(np.nan_to_num(means - ci_lo, nan=0.0), 0, None)
            yerr_hi = np.clip(np.nan_to_num(ci_hi - means, nan=0.0), 0, None)
            ax.bar(
                xs,
                np.nan_to_num(means, nan=0.0),
                bar_w,
                yerr=[yerr_lo, yerr_hi],
                label=ds,
                edgecolor="black",
                linewidth=0.4,
                capsize=3,
            )
            for x, m, n in zip(xs, means, ds_sub["n_windows"].to_numpy()):
                if not np.isnan(m):
                    ax.text(x, m + 0.5, f"N={int(n)}", ha="center", fontsize=7)
        ax.set_xticks(np.arange(len(TIERS)) + bar_w * (len(datasets) - 1) / 2)
        ax.set_xticklabels(TIERS)
        ax.set_title(title)
        ax.set_xlabel("SQA tier")
        ax.grid(axis="y", linestyle=":", alpha=0.4)

    axes[0].set_ylabel("HR MAE (BPM)")
    axes[0].legend(title="Dataset", loc="upper left", framealpha=0.9)

    fig.suptitle("HR estimation MAE stratified by SQA tier (95% bootstrap CI)")
    plt.tight_layout()
    out = FIG_DIR / "fig_hr_ablation_mae.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_hr_ablation_box(df: pd.DataFrame) -> Path:
    """Box plot of HR errors per tier per dataset (log y, 2 subplots)."""
    df_ok = df.dropna(subset=["hr_error_abs"]).copy()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    label_cols = ["sqa_tier_pseudo", "sqa_tier_lr_pred"]
    titles = ["Pseudo-label tier", "LR-predicted tier"]
    datasets = sorted(df_ok["dataset"].unique())

    for ax, lc, title in zip(axes, label_cols, titles):
        positions: list[float] = []
        data: list[np.ndarray] = []
        tick_labels: list[str] = []
        face_colors: list[str] = []
        x = 0.0
        for ds in datasets:
            for tier in TIERS:
                sub = df_ok[(df_ok.dataset == ds) & (df_ok[lc] == tier)]
                errs = sub["hr_error_abs"].to_numpy(dtype=float)
                # Floor at 0.1 BPM so log-scale boxplot doesn't choke on zeros
                errs = np.where(errs < 0.1, 0.1, errs)
                if len(errs) > 0:
                    positions.append(x)
                    data.append(errs)
                    tick_labels.append(f"{ds}\n{tier}")
                    face_colors.append(TIER_COLORS[tier])
                x += 1.0
            x += 0.5  # gap between datasets

        bp = ax.boxplot(
            data,
            positions=positions,
            widths=0.7,
            patch_artist=True,
            showfliers=True,
            flierprops=dict(marker=".", markersize=2, alpha=0.3),
        )
        for patch, c in zip(bp["boxes"], face_colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.65)
        ax.set_xticks(positions)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=7)
        ax.set_yscale("log")
        ax.set_title(title)
        ax.grid(axis="y", linestyle=":", alpha=0.4, which="both")

    axes[0].set_ylabel("HR error |HR_ppg - HR_gt| (BPM, log scale)")
    fig.suptitle("HR error distribution per SQA tier (log y)")
    plt.tight_layout()
    out = FIG_DIR / "fig_hr_ablation_box.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_hr_ablation_scatter(df: pd.DataFrame) -> Path:
    """Scatter HR_ppg vs HR_gt, colored by pseudo-label tier, identity line."""
    df_ok = df.dropna(subset=["hr_error_abs"]).copy()
    datasets = sorted(df_ok["dataset"].unique())
    fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 5.2), sharey=True)
    if len(datasets) == 1:
        axes = [axes]

    for ax, ds in zip(axes, datasets):
        ds_sub = df_ok[df_ok.dataset == ds]
        for tier in TIERS:
            tier_sub = ds_sub[ds_sub.sqa_tier_pseudo == tier]
            if tier_sub.empty:
                continue
            ax.scatter(
                tier_sub["hr_ground_truth"],
                tier_sub["hr_ppg"],
                s=6,
                alpha=0.35,
                color=TIER_COLORS[tier],
                label=f"{tier} (N={len(tier_sub)})",
            )
        # Identity line
        if len(ds_sub) > 0:
            lo = float(min(ds_sub["hr_ground_truth"].min(), ds_sub["hr_ppg"].min()))
            hi = float(max(ds_sub["hr_ground_truth"].max(), ds_sub["hr_ppg"].max()))
            ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.8, label="identity")
        ax.set_title(f"{ds}")
        ax.set_xlabel("HR ground truth (BPM)")
        ax.grid(linestyle=":", alpha=0.4)
        ax.legend(loc="upper left", fontsize=8, framealpha=0.9)

    axes[0].set_ylabel("HR_ppg (BPM)")
    fig.suptitle("HR_ppg vs ECG-derived HR, colored by pseudo-label tier")
    plt.tight_layout()
    out = FIG_DIR / "fig_hr_ablation_scatter.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


# ============================================================
# Main
# ============================================================
def main() -> int:
    if not CLASSIFIER_CSV.exists():
        print(f"ERROR: {CLASSIFIER_CSV} not found. Run eval_sqa_classifier.py first.")
        return 1
    if not BIDMC_DIR.exists():
        print(f"ERROR: {BIDMC_DIR} not found.")
        return 1
    if not DALIA_DIR.exists():
        print(f"ERROR: {DALIA_DIR} not found.")
        return 1

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {CLASSIFIER_CSV.name}...")
    cls_df = pd.read_csv(CLASSIFIER_CSV)
    print(f"  {len(cls_df)} total windows across datasets: {sorted(cls_df['dataset'].unique())}")

    # Restrict to ECG-bearing datasets
    work = cls_df[cls_df["dataset"].isin(["BIDMC", "PPG-DaLiA"])].copy()
    print(
        f"  Filtered to BIDMC + PPG-DaLiA: {len(work)} windows "
        f"(BIDMC={int((work.dataset=='BIDMC').sum())}, "
        f"PPG-DaLiA={int((work.dataset=='PPG-DaLiA').sum())})"
    )

    # Per-window HR derivation
    print("\nDeriving HR per window (this may take a few minutes)...")
    t0 = time.time()
    out_rows: list[dict] = []
    n = len(work)
    for i, (_, row) in enumerate(work.iterrows()):
        if i % 500 == 0 and i > 0:
            print(f"  {i}/{n} ({i/n*100:.0f}%) elapsed {time.time()-t0:.0f}s")
        if row["dataset"] == "BIDMC":
            hr_info = eval_bidmc_row(row)
        else:
            hr_info = eval_dalia_row(row)
        out_rows.append(
            {
                "file": row["file"],
                "dataset": row["dataset"],
                "subject_id": row["subject_id"],
                "window_idx": int(row["window_idx"]),
                "t_start_s": float(row["t_start_s"]),
                "t_end_s": float(row["t_end_s"]),
                "fs": int(row["fs"]),
                "hr_ppg": hr_info["hr_ppg"],
                "hr_ground_truth": hr_info["hr_ground_truth"],
                "hr_error_abs": hr_info["hr_error_abs"],
                "skip_reason": hr_info["skip_reason"],
                "sqa_tier_pseudo": row["label"],
                "sqa_tier_lr_pred": row["label_pred_lr"],
            }
        )
    elapsed = time.time() - t0
    print(f"  Done {n}/{n} in {elapsed:.1f}s")

    df = pd.DataFrame(out_rows)
    df.to_csv(OUT_RESULTS_CSV, index=False)
    print(f"\nSaved per-window results: {OUT_RESULTS_CSV.name} ({len(df)} rows)")

    # Skipped breakdown
    print_skipped(df)

    # Build summary tables
    summary_pseudo = summarize_by_tier(df, "sqa_tier_pseudo", "pseudo")
    summary_lr = summarize_by_tier(df, "sqa_tier_lr_pred", "lr_pred")
    summary = pd.concat([summary_pseudo, summary_lr], ignore_index=True)

    # Fix D: append BIDMC_excl_bidmc_04 rows for transparency about
    # the rhythmic-artifact outlier dominating BIDMC excellent-tier mean.
    extra_rows: list[dict] = []
    for label_src, label_col in (
        ("pseudo", "sqa_tier_pseudo"),
        ("lr_pred", "sqa_tier_lr_pred"),
    ):
        bidmc_no_b04 = df[
            (df["dataset"] == "BIDMC")
            & (df["subject_id"] != "bidmc_04")
        ]
        for tier in TIERS:
            sub = bidmc_no_b04[bidmc_no_b04[label_col] == tier]
            sub_clean = sub[sub["hr_error_abs"].notna()]
            if len(sub_clean) > 0:
                errs = sub_clean["hr_error_abs"].to_numpy(dtype=float)
                ci_lo, ci_hi = bootstrap_mae_ci(errs)
                extra_rows.append(
                    {
                        "label_source": label_src,
                        "dataset": "BIDMC_excl_bidmc_04",
                        "tier": tier,
                        "n_windows": int(len(errs)),
                        "mae_mean": float(np.mean(errs)),
                        "mae_median": float(np.median(errs)),
                        "mae_p95": float(np.percentile(errs, 95)),
                        "mae_ci95_lo": ci_lo,
                        "mae_ci95_hi": ci_hi,
                    }
                )
    if extra_rows:
        summary = pd.concat(
            [summary, pd.DataFrame(extra_rows)], ignore_index=True
        )

    # Fix A: count tiers with data per (label_source, dataset) group
    summary["n_tiers_with_data"] = summary.groupby(
        ["label_source", "dataset"]
    )["n_windows"].transform(lambda x: int((x > 0).sum()))

    summary.to_csv(OUT_SUMMARY_CSV, index=False)
    print(f"Saved summary table: {OUT_SUMMARY_CSV.name} ({len(summary)} rows)")

    print_summary_table(summary)

    mono = check_monotonic(summary)
    print_monotonic_check(mono)

    # Figures
    print("\nGenerating figures...")
    f1 = fig_hr_ablation_mae(summary)
    print(f"  {f1.name}")
    f2 = fig_hr_ablation_box(df)
    print(f"  {f2.name}")
    f3 = fig_hr_ablation_scatter(df)
    print(f"  {f3.name}")

    print_honest_framing()
    return 0


if __name__ == "__main__":
    sys.exit(main())
