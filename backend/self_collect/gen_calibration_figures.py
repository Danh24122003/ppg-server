"""
Generate thesis figures from calibration drift CSV results + LOSO eval CSV.

Inputs:
  _drift_results_sbp.csv (output of eval_subject_holdout_calibration.py --target sbp)
  _drift_results_dbp.csv
  eval_true_loso_report.csv (output of eval_true_loso.py)

Outputs (Thesis/figures/):
  fig1_drift_vs_days.pdf            — drift vs days_from_anchor scatter + linear regression + 95% CI + AAMI 5 mmHg line
  fig2_bland_altman_sbp.pdf         — Bland-Altman SBP (calibrated, ±1.96 SD LoA)
  fig2_bland_altman_dbp.pdf         — Bland-Altman DBP
  fig3_per_subject_mae.pdf          — Per-subject calibrated MAE bar chart, color by AHA 2017 BP class
  fig4_calibration_gain.pdf         — Raw vs calibrated MAE per subject
  fig5_model_comparison.pdf         — Drift-based + LOSO + naive baseline comparison
  fig6_loso_regression_to_mean.pdf  — LOSO predictions converge to ~113/72 mmHg (regression to mean)
  fig7_loso_error_vs_deviation.pdf  — LOSO error scales with |true BP - training mean|
  fig8_naive_vs_model_vs_calibrated.pdf — Naive baseline vs model LOSO vs calibrated (calibration MANDATORY story)

Usage:
  python gen_calibration_figures.py [--input-dir DIR] [--output-dir DIR]
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "Thesis" / "figures"

# AAMI/ISO 81060-2 thresholds
AAMI_ME_LIMIT = 5.0      # mmHg
AAMI_SD_LIMIT = 8.0      # mmHg

# Defense-grade reference numbers, computed 2026-05-06 chiều (post recruit batch 3 +S14 +S13).
# Source: eval_true_loso.py output `eval_true_loso_report.csv` (N=15 unique persons LOSO),
#   NOT eval_naive_baseline.py (gives different naive baseline at session-level granularity).
# LOSO MAE = mean(|sbp_err|) from per-person held-out predictions.
# Naive LOO baseline = predict mean(train_fold cuff_truth) for each held-out person.
# Both metrics use 15 unique persons groupkfold structure post recruit batch 3 (was N=13 pre 6/5 chiều).
LOSO_SBP_MAE = 13.86     # N=15 unique persons (was 12.01 N=13, +S14 +S13 Stage 1/2 BP reading)
LOSO_DBP_MAE = 7.29      # N=15
NAIVE_SBP_LOO = 12.93    # N=15 LOO subject mean (was 10.92 N=13)
NAIVE_DBP_LOO = 8.24     # N=15

# Color palette (colorblind-safe)
PALETTE = sns.color_palette("colorblind", 8)

# Global rcParams (P2#12)
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.format': 'pdf',
    'savefig.bbox': 'tight',
})


def _bp_category(sbp: float, dbp: float) -> str:
    """AHA 2017 BP classification."""
    if sbp >= 140 or dbp >= 90:
        return "Stage 2 BP reading"
    if sbp >= 130 or dbp >= 80:
        return "Stage 1 BP reading"
    if sbp >= 120:
        return "Elevated"
    return "Normal"


def _setup_axes(ax, title: str, xlabel: str, ylabel: str):
    ax.set_title(title, fontsize=12, weight='bold')
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, alpha=0.3)


def fig1_drift_vs_days(df_sbp: pd.DataFrame, df_dbp: pd.DataFrame, output_path: Path):
    """Figure 1: drift vs days_from_anchor for SBP + DBP, with regression + 95% CI."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)

    n_drift = int((df_sbp.included_in_drift_eval == True).sum())

    for ax, df, target, color in zip(axes, [df_sbp, df_dbp], ['SBP', 'DBP'], [PALETTE[0], PALETTE[1]]):
        drift = df[df.included_in_drift_eval == True].copy()
        if len(drift) == 0:
            ax.text(0.5, 0.5, f"No drift sessions yet\n(need >=2 sessions per subject)",
                    transform=ax.transAxes, ha='center', va='center', fontsize=11)
            _setup_axes(ax, f"{target} Calibration Drift", "Days from anchor", f"|Calibrated - cuff| ({target}, mmHg)")
            continue

        # Scatter per subject
        sns.scatterplot(data=drift, x='days_from_anchor', y='abs_err_calibrated',
                        hue='subject_id', s=80, ax=ax, palette=PALETTE[:drift['subject_id'].nunique()])

        # Linear regression with 95% CI band (need >=3 points)
        if len(drift) >= 3:
            xs = drift.days_from_anchor.to_numpy()
            ys = drift.abs_err_calibrated.to_numpy()
            lr = stats.linregress(xs, ys)
            x_range = np.linspace(xs.min(), xs.max(), 100)
            y_pred = lr.intercept + lr.slope * x_range
            ax.plot(x_range, y_pred, 'k--', alpha=0.7, lw=1.5,
                    label=f'y = {lr.intercept:.2f} + {lr.slope:.3f}*d  (R={lr.rvalue:.2f}, p={lr.pvalue:.3f})')
            # 95% CI band via standard error of prediction
            n = len(xs)
            t_val = stats.t.ppf(0.975, n - 2) if n > 2 else 1.96
            mean_x = xs.mean()
            ss_x = np.sum((xs - mean_x)**2)
            se_pred = lr.stderr * np.sqrt(1 + 1/n + (x_range - mean_x)**2 / ss_x) if ss_x > 0 else 0
            # P0#2: clip CI lower bound to 0 (|err| is always >= 0)
            ci_lower = np.maximum(0, y_pred - t_val * se_pred)
            ci_upper = y_pred + t_val * se_pred
            ax.fill_between(x_range, ci_lower, ci_upper,
                            alpha=0.15, color='gray', label='95% CI (clipped at 0)')

        # AAMI threshold line
        ax.axhline(AAMI_ME_LIMIT, color='red', linestyle=':', lw=1.5, alpha=0.7,
                   label=f'AAMI |ME| <= {AAMI_ME_LIMIT} mmHg')

        _setup_axes(ax, f"{target} Calibration Drift Over Time",
                    "Days from anchor session", f"|Calibrated - cuff| ({target}, mmHg)")
        ax.legend(fontsize=9, loc='best')

    # P1#4: rephrase suptitle
    fig.suptitle(f"Figure 1: Per-Subject Calibration Drift (n = {n_drift} drift sessions; preliminary)",
                 fontsize=13, weight='bold', y=1.02)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


def fig2_bland_altman(df: pd.DataFrame, target: str, output_path: Path):
    """Figure 2: Bland-Altman calibrated vs cuff truth (drift sessions only)."""
    drift = df[df.included_in_drift_eval == True].copy()
    if len(drift) < 2:
        print(f"  Skip BA {target}: only {len(drift)} drift sessions")
        return

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)

    means = (drift.pred_calibrated + drift.cuff_truth) / 2
    diffs = drift.pred_calibrated - drift.cuff_truth
    bias = diffs.mean()
    sd = diffs.std(ddof=1) if len(diffs) > 1 else 0
    lower = bias - 1.96 * sd
    upper = bias + 1.96 * sd

    sns.scatterplot(x=means, y=diffs, hue=drift['subject_id'], s=80, ax=ax,
                    palette=PALETTE[:drift['subject_id'].nunique()])
    ax.axhline(bias, color='red', lw=1.5,
               label=f'Mean bias = {bias:+.2f} mmHg')
    # P1#5: add mmHg units to LoA labels
    ax.axhline(upper, color='blue', linestyle='--', lw=1.2,
               label=f'+1.96 SD = {upper:+.2f} mmHg')
    ax.axhline(lower, color='blue', linestyle='--', lw=1.2,
               label=f'-1.96 SD = {lower:+.2f} mmHg')
    ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(AAMI_ME_LIMIT, color='orange', linestyle=':', alpha=0.5,
               label=f'AAMI ME +/-{AAMI_ME_LIMIT} mmHg')
    ax.axhline(-AAMI_ME_LIMIT, color='orange', linestyle=':', alpha=0.5)

    # P1#5: n=4 disclaimer
    ax.text(0.02, 0.02,
            f"Note: n={len(drift)}, underpowered.\nLoA unreliable at this N.\nPreliminary only.",
            transform=ax.transAxes, fontsize=8, color='gray',
            ha='left', va='bottom',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.7, edgecolor='gray'))

    _setup_axes(ax, f"Bland-Altman: {target} Calibrated vs Cuff Truth",
                f"Mean({target}_calibrated, {target}_cuff)  [mmHg]",
                f"{target}_calibrated - {target}_cuff  [mmHg]")
    ax.legend(fontsize=9, loc='best')

    fig.suptitle(f"Figure 2: Bland-Altman {target} (n={len(drift)})", fontsize=13, weight='bold', y=1.02)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


def fig3_per_subject_mae(df_sbp: pd.DataFrame, df_dbp: pd.DataFrame, output_path: Path):
    """Figure 3: Per-subject calibrated MAE bar chart, AHA 2017 coloring uses BOTH SBP+DBP."""
    drift_sbp = df_sbp[df_sbp.included_in_drift_eval == True].copy()
    drift_dbp = df_dbp[df_dbp.included_in_drift_eval == True].copy()

    if len(drift_sbp) == 0 and len(drift_dbp) == 0:
        print(f"  Skip per-subject MAE: no drift data")
        return

    # P0#3: build joint per-subject table with BOTH sbp_mean and dbp_mean for correct AHA classification
    sbp_agg = drift_sbp.groupby('subject_id').agg(
        sbp_mae=('abs_err_calibrated', 'mean'),
        sbp_mean=('cuff_truth', 'mean'),
    ).reset_index() if len(drift_sbp) > 0 else pd.DataFrame(columns=['subject_id', 'sbp_mae', 'sbp_mean'])
    dbp_agg = drift_dbp.groupby('subject_id').agg(
        dbp_mae=('abs_err_calibrated', 'mean'),
        dbp_mean=('cuff_truth', 'mean'),
    ).reset_index() if len(drift_dbp) > 0 else pd.DataFrame(columns=['subject_id', 'dbp_mae', 'dbp_mean'])
    joint = sbp_agg.merge(dbp_agg, on='subject_id', how='outer')

    bp_class_map = {'Normal': PALETTE[2], 'Elevated': PALETTE[3],
                    'Stage 1 BP reading': PALETTE[4], 'Stage 2 BP reading': PALETTE[5]}

    fig, axes = plt.subplots(2, 1, figsize=(11, 7.5), constrained_layout=True)

    for ax, target, mae_col in zip(axes, ['SBP', 'DBP'], ['sbp_mae', 'dbp_mae']):
        sub_df = joint.dropna(subset=[mae_col]).sort_values(mae_col).reset_index(drop=True)
        if len(sub_df) == 0:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha='center', va='center')
            _setup_axes(ax, f"{target} per-subject MAE", "Subject", f"{target} MAE (mmHg)")
            continue

        # AHA classification uses BOTH sbp_mean and dbp_mean (fall back to neutral 70/120 if missing)
        categories = []
        for _, row in sub_df.iterrows():
            s = row['sbp_mean'] if pd.notna(row.get('sbp_mean')) else 120.0
            d = row['dbp_mean'] if pd.notna(row.get('dbp_mean')) else 70.0
            categories.append(_bp_category(s, d))
        colors = [bp_class_map.get(cat, PALETTE[0]) for cat in categories]

        bars = ax.bar(sub_df.subject_id, sub_df[mae_col], color=colors, edgecolor='black', linewidth=0.5)
        aami_line = ax.axhline(AAMI_ME_LIMIT, color='red', linestyle=':', alpha=0.7,
                                label=f'AAMI {AAMI_ME_LIMIT} mmHg')
        # Value labels on bars
        for bar, val in zip(bars, sub_df[mae_col]):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.2,
                    f'{val:.1f}', ha='center', fontsize=9)

        _setup_axes(ax, f"{target} Calibrated MAE per Subject",
                    "Subject", f"{target} MAE (mmHg)")
        # P2#11: explicit AHA category legend patches + AAMI line
        legend_elements = [
            Patch(facecolor=PALETTE[2], label='Normal'),
            Patch(facecolor=PALETTE[3], label='Elevated'),
            Patch(facecolor=PALETTE[4], label='Stage 1 BP reading'),
            Patch(facecolor=PALETTE[5], label='Stage 2 BP reading'),
            aami_line,
        ]
        ax.legend(handles=legend_elements, fontsize=9, loc='upper right')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    fig.suptitle(f"Figure 3: Per-subject Calibrated MAE Distribution (AHA 2017 coloring uses both SBP+DBP)",
                 fontsize=13, weight='bold', y=1.02)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


def fig4_calibration_gain(df_sbp: pd.DataFrame, df_dbp: pd.DataFrame, output_path: Path):
    """Figure 4: Calibration gain (raw vs calibrated MAE per subject)."""
    drift_sbp = df_sbp[df_sbp.included_in_drift_eval == True].copy()
    drift_dbp = df_dbp[df_dbp.included_in_drift_eval == True].copy()

    if len(drift_sbp) == 0:
        print(f"  Skip calibration gain: no drift data")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)

    for ax, drift, target in zip(axes, [drift_sbp, drift_dbp], ['SBP', 'DBP']):
        if len(drift) == 0:
            continue
        per_sub = drift.groupby('subject_id').agg(
            raw_mae=('abs_err_raw', 'mean'),
            cal_mae=('abs_err_calibrated', 'mean'),
        ).reset_index()

        x = np.arange(len(per_sub))
        width = 0.35
        ax.bar(x - width/2, per_sub.raw_mae, width, label='Raw (no calibration)', color=PALETTE[6])
        ax.bar(x + width/2, per_sub.cal_mae, width, label='Calibrated (1-anchor)', color=PALETTE[0])
        ax.set_xticks(x)
        ax.set_xticklabels(per_sub.subject_id, rotation=45, ha='right')

        # Average lines
        avg_raw = per_sub.raw_mae.mean()
        avg_cal = per_sub.cal_mae.mean()
        ax.axhline(avg_raw, color=PALETTE[6], linestyle='--', alpha=0.5,
                   label=f'Avg raw = {avg_raw:.1f}')
        ax.axhline(avg_cal, color=PALETTE[0], linestyle='--', alpha=0.5,
                   label=f'Avg calibrated = {avg_cal:.1f}')
        ax.axhline(AAMI_ME_LIMIT, color='red', linestyle=':', alpha=0.6,
                   label=f'AAMI {AAMI_ME_LIMIT}')

        # P1#6: per-target annotation about calibration effect
        n_drift = len(drift)
        if target == 'SBP':
            delta = avg_raw - avg_cal
            pct = (delta / avg_raw * 100.0) if avg_raw > 0 else 0
            note = f"SBP calibration reduces MAE\n{delta:.2f} mmHg ({pct:.0f}%)\n(n_drift={n_drift})"
            note_color = 'darkgreen'
        else:
            delta = avg_raw - avg_cal
            note = (f"DBP calibration effect marginal\n"
                    f"(Delta MAE {delta:+.2f} mmHg,\n"
                    f"n_drift={n_drift} underpowered)")
            note_color = 'darkorange'
        ax.text(0.02, 0.97, note, transform=ax.transAxes, fontsize=8,
                color=note_color, ha='left', va='top',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85, edgecolor=note_color))

        _setup_axes(ax, f"{target}: Calibration Gain per Subject", "Subject", f"{target} MAE (mmHg)")
        ax.legend(fontsize=8, loc='upper right')

    fig.suptitle(f"Figure 4: Per-Subject Gain from 1-Reading Anchor Calibration",
                 fontsize=13, weight='bold', y=1.02)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


def fig5_model_comparison(df_sbp: pd.DataFrame, df_dbp: pd.DataFrame, output_path: Path,
                           extra_results: dict = None):
    """Figure 5: Drift-based bars + LOSO ref line + naive baseline ref line."""
    drift_sbp = df_sbp[df_sbp.included_in_drift_eval == True].copy()
    drift_dbp = df_dbp[df_dbp.included_in_drift_eval == True].copy()

    if len(drift_sbp) == 0:
        print(f"  Skip model comparison: no drift data")
        return

    n_drift_sbp = len(drift_sbp)
    n_drift_dbp = len(drift_dbp)

    models = ['Current SVR/RF']
    sbp_calibrated = [drift_sbp.abs_err_calibrated.mean()]
    sbp_raw = [drift_sbp.abs_err_raw.mean()]
    dbp_calibrated = [drift_dbp.abs_err_calibrated.mean()] if len(drift_dbp) > 0 else [np.nan]
    dbp_raw = [drift_dbp.abs_err_raw.mean()] if len(drift_dbp) > 0 else [np.nan]

    if extra_results:
        for name, res in extra_results.items():
            models.append(name)
            sbp_calibrated.append(res.get('sbp_cal', np.nan))
            sbp_raw.append(res.get('sbp_raw', np.nan))
            dbp_calibrated.append(res.get('dbp_cal', np.nan))
            dbp_raw.append(res.get('dbp_raw', np.nan))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)

    for ax, raw_arr, cal_arr, target, loso_mae, naive_mae, n_drift in zip(
            axes,
            [sbp_raw, dbp_raw],
            [sbp_calibrated, dbp_calibrated],
            ['SBP', 'DBP'],
            [LOSO_SBP_MAE, LOSO_DBP_MAE],
            [NAIVE_SBP_LOO, NAIVE_DBP_LOO],
            [n_drift_sbp, n_drift_dbp]):
        x = np.arange(len(models))
        width = 0.35
        ax.bar(x - width/2, raw_arr, width,
               label=f'Calibration-FREE (drift n={n_drift})', color=PALETTE[6])
        ax.bar(x + width/2, cal_arr, width,
               label=f'Calibrated 1-anchor (drift n={n_drift})', color=PALETTE[0])
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=15, ha='right')

        # P0#1: LOSO + naive reference lines
        ax.axhline(loso_mae, color='purple', linestyle='-.', lw=1.8, alpha=0.85,
                   label=f'True LOSO MAE = {loso_mae:.2f} (N=15 unique persons)')
        ax.axhline(naive_mae, color='brown', linestyle=(0, (3, 1, 1, 1)), lw=1.5, alpha=0.85,
                   label=f'Naive LOO baseline = {naive_mae:.2f}')
        ax.axhline(AAMI_ME_LIMIT, color='red', linestyle=':', alpha=0.6,
                   label=f'AAMI {AAMI_ME_LIMIT} mmHg')

        # Value labels
        for i, (r, c) in enumerate(zip(raw_arr, cal_arr)):
            if not np.isnan(r):
                ax.text(i - width/2, r + 0.3, f'{r:.1f}', ha='center', fontsize=9)
            if not np.isnan(c):
                ax.text(i + width/2, c + 0.3, f'{c:.1f}', ha='center', fontsize=9)

        _setup_axes(ax, f"{target}: Model Comparison", "Model", f"{target} MAE (mmHg)")
        ax.legend(fontsize=8, loc='upper right')

    # P0#1 + P1#7: ASCII suptitle, clarify data sources
    fig.suptitle(
        "Figure 5: Model Comparison - Cross-session drift bars vs LOSO and naive baseline references\n"
        "(extend with AutoGluon, IW-SVR in Week 2)",
        fontsize=12, weight='bold', y=1.04)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


# ---------- A2: New LOSO defense figures ----------

def fig6_loso_regression_to_mean(loso_df: pd.DataFrame, output_path: Path):
    """Figure 6: LOSO predictions regress to training mean ~113/72 mmHg."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)

    # Subjects to annotate (extremes + mid example)
    annotate_subs = {'S15', 'S08', 'S02'}

    for ax, target_true, target_pred, target_label in zip(
            axes,
            ['sbp_true', 'dbp_true'],
            ['sbp_pred', 'dbp_pred'],
            ['SBP', 'DBP']):
        sub_df = loso_df.sort_values(target_true).reset_index(drop=True)
        x = np.arange(len(sub_df))
        true_vals = sub_df[target_true].to_numpy()
        pred_vals = sub_df[target_pred].to_numpy()
        mean_pred = float(np.mean(pred_vals))

        # Connector lines true -> pred
        for xi, t, p in zip(x, true_vals, pred_vals):
            ax.plot([xi, xi], [t, p], color='lightgray', lw=1.0, alpha=0.7, zorder=1)

        ax.scatter(x, true_vals, marker='o', s=70, color=PALETTE[0],
                   edgecolor='black', linewidth=0.5, zorder=3, label=f'{target_label} cuff truth')
        ax.scatter(x, pred_vals, marker='s', s=70, facecolor='none',
                   edgecolor=PALETTE[1], linewidth=1.5, zorder=3, label=f'{target_label} LOSO prediction')

        ax.axhline(mean_pred, color='red', linestyle='--', lw=1.5, alpha=0.8,
                   label=f'Model predicts ~ {mean_pred:.1f} mmHg (regression to mean)')

        # Annotate extremes
        for xi, row in zip(x, sub_df.itertuples()):
            if row.person in annotate_subs:
                err = getattr(row, target_true.replace('true', 'err'))
                ax.annotate(
                    f"{row.person}\n(err {err:+.1f})",
                    xy=(xi, true_vals[xi]),
                    xytext=(0, 16 if true_vals[xi] < pred_vals[xi] else -28),
                    textcoords='offset points',
                    fontsize=8, ha='center',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                              alpha=0.9, edgecolor='gray', linewidth=0.5),
                )

        ax.set_xticks(x)
        ax.set_xticklabels(sub_df['person'], rotation=45, ha='right')
        _setup_axes(ax, f"{target_label}: LOSO True vs Predicted (sorted by true)",
                    "Subject", f"{target_label} (mmHg)")
        ax.legend(fontsize=9, loc='best')

    fig.suptitle(
        "Figure 6: LOSO Predictions Show Regression to Training Mean (~113/72 mmHg)",
        fontsize=13, weight='bold', y=1.03)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


def fig7_loso_error_vs_deviation(loso_df: pd.DataFrame, output_path: Path):
    """Figure 7: |LOSO error| scales with |true - mean(pred)|."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)

    for ax, target_true, target_pred, target_err, target_label in zip(
            axes,
            ['sbp_true', 'dbp_true'],
            ['sbp_pred', 'dbp_pred'],
            ['sbp_err', 'dbp_err'],
            ['SBP', 'DBP']):
        mean_pred = float(loso_df[target_pred].mean())
        deviation = (loso_df[target_true] - mean_pred).abs().to_numpy()
        abs_err = loso_df[target_err].abs().to_numpy()

        # Per-subject color
        n_sub = len(loso_df)
        palette = sns.color_palette("colorblind", n_sub)
        for i, (xi, yi, name) in enumerate(zip(deviation, abs_err, loso_df['person'])):
            ax.scatter(xi, yi, s=80, color=palette[i], edgecolor='black',
                       linewidth=0.5, zorder=3, label=name)

        # Linear regression
        if len(deviation) >= 3:
            lr = stats.linregress(deviation, abs_err)
            x_range = np.linspace(0, deviation.max() * 1.05, 100)
            y_pred = lr.intercept + lr.slope * x_range
            ax.plot(x_range, y_pred, 'k--', alpha=0.7, lw=1.5,
                    label=f'fit: y = {lr.intercept:.2f} + {lr.slope:.2f}*x  '
                          f'(R^2={lr.rvalue**2:.2f}, p={lr.pvalue:.3g})')
            # Identity line y=x
            xy_max = max(deviation.max(), abs_err.max()) * 1.05
            ax.plot([0, xy_max], [0, xy_max], color='gray', linestyle=':', lw=1.0,
                    alpha=0.7, label='y = x (perfect inverse calibration)')

        _setup_axes(ax,
                    f"{target_label}: |LOSO error| vs |true - model_mean({mean_pred:.1f})|",
                    f"|{target_label}_true - mean(pred)|  (mmHg)",
                    f"|{target_label}_err|  (mmHg)")
        ax.legend(fontsize=7, loc='upper left', ncol=2)

    fig.suptitle("Figure 7: LOSO Error Scales with BP Deviation from Training Mean",
                 fontsize=13, weight='bold', y=1.03)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


def fig8_naive_vs_model_vs_calibrated(df_sbp: pd.DataFrame, df_dbp: pd.DataFrame, output_path: Path):
    """Figure 8: 3-bar comparison per target — calibration MANDATORY narrative."""
    drift_sbp = df_sbp[df_sbp.included_in_drift_eval == True].copy()
    drift_dbp = df_dbp[df_dbp.included_in_drift_eval == True].copy()

    sbp_calibrated = float(drift_sbp.abs_err_calibrated.mean()) if len(drift_sbp) > 0 else float('nan')
    dbp_calibrated = float(drift_dbp.abs_err_calibrated.mean()) if len(drift_dbp) > 0 else float('nan')

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

    groups = ['SBP', 'DBP']
    bar_labels = ['Naive LOO baseline', 'Model LOSO calibration-free', 'Model + per-subject calibration']
    sbp_vals = [NAIVE_SBP_LOO, LOSO_SBP_MAE, sbp_calibrated]
    dbp_vals = [NAIVE_DBP_LOO, LOSO_DBP_MAE, dbp_calibrated]
    bar_colors = ['gray', 'steelblue', 'green']

    n_groups = len(groups)
    n_bars = len(bar_labels)
    bar_width = 0.25
    x = np.arange(n_groups)

    for i, (label, color) in enumerate(zip(bar_labels, bar_colors)):
        vals = [sbp_vals[i], dbp_vals[i]]
        offset = (i - (n_bars - 1) / 2) * bar_width
        bars = ax.bar(x + offset, vals, bar_width, label=label, color=color,
                      edgecolor='black', linewidth=0.6)
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2, val + 0.15,
                        f'{val:.2f}', ha='center', fontsize=9, weight='bold')

    ax.axhline(AAMI_ME_LIMIT, color='red', linestyle='--', lw=1.5, alpha=0.7,
               label=f'AAMI ME limit ({AAMI_ME_LIMIT} mmHg)')

    # P0#1 narrative annotations
    sbp_gap = LOSO_SBP_MAE - NAIVE_SBP_LOO
    dbp_gap = LOSO_DBP_MAE - NAIVE_DBP_LOO
    sbp_note = (f"SBP: model loses to naive\nbaseline by {sbp_gap:+.2f} mmHg\n"
                f"-> calibration MANDATORY")
    dbp_note = (f"DBP: model marginal vs naive\n(Delta = {dbp_gap:+.2f} mmHg)")
    ax.annotate(sbp_note, xy=(x[0], max(sbp_vals[:2]) + 0.5),
                xytext=(x[0] - 0.15, max(sbp_vals[:2]) + 4),
                fontsize=9, color='darkred', ha='left',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='mistyrose',
                          edgecolor='darkred', alpha=0.9),
                arrowprops=dict(arrowstyle='->', color='darkred', lw=1.0))
    ax.annotate(dbp_note, xy=(x[1], max(dbp_vals[:2]) + 0.3),
                xytext=(x[1] - 0.4, max(dbp_vals[:2]) + 3),
                fontsize=9, color='darkorange', ha='left',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='oldlace',
                          edgecolor='darkorange', alpha=0.9),
                arrowprops=dict(arrowstyle='->', color='darkorange', lw=1.0))

    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    _setup_axes(ax, "Naive Baseline vs Calibration-Free Model vs Per-Subject Calibration",
                "BP Target", "MAE (mmHg)")
    ax.legend(fontsize=9, loc='upper right')

    fig.suptitle(
        "Figure 8: Calibration Is Mandatory - Calibration-Free Model Does Not Beat Naive Baseline",
        fontsize=13, weight='bold', y=1.02)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=SCRIPT_DIR,
                        help="Directory with _drift_results_*.csv + eval_true_loso_report.csv")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR,
                        help="Directory to write figures into")
    args = parser.parse_args()

    in_dir = args.input_dir
    out_dir = args.output_dir

    sbp_csv = in_dir / "_drift_results_sbp.csv"
    dbp_csv = in_dir / "_drift_results_dbp.csv"
    loso_csv = in_dir / "eval_true_loso_report.csv"

    if not sbp_csv.exists() or not dbp_csv.exists():
        print(f"[ERROR] Missing input CSV. Run eval_subject_holdout_calibration.py first.")
        return 1

    df_sbp = pd.read_csv(sbp_csv)
    df_dbp = pd.read_csv(dbp_csv)

    print(f"Loaded SBP: {len(df_sbp)} rows, {df_sbp['subject_id'].nunique()} subjects")
    print(f"Loaded DBP: {len(df_dbp)} rows, {df_dbp['subject_id'].nunique()} subjects")
    print(f"Drift sessions SBP: {(df_sbp.included_in_drift_eval == True).sum()}")
    print(f"Drift sessions DBP: {(df_dbp.included_in_drift_eval == True).sum()}")

    if loso_csv.exists():
        loso_df = pd.read_csv(loso_csv)
        print(f"Loaded LOSO: {len(loso_df)} subjects from {loso_csv.name}")
    else:
        loso_df = None
        print(f"[WARN] {loso_csv.name} missing; figures 6/7 skipped")

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nGenerating figures to: {out_dir}\n")

    fig1_drift_vs_days(df_sbp, df_dbp, out_dir / "fig1_drift_vs_days.pdf")
    fig2_bland_altman(df_sbp, "SBP", out_dir / "fig2_bland_altman_sbp.pdf")
    fig2_bland_altman(df_dbp, "DBP", out_dir / "fig2_bland_altman_dbp.pdf")
    fig3_per_subject_mae(df_sbp, df_dbp, out_dir / "fig3_per_subject_mae.pdf")
    fig4_calibration_gain(df_sbp, df_dbp, out_dir / "fig4_calibration_gain.pdf")
    fig5_model_comparison(df_sbp, df_dbp, out_dir / "fig5_model_comparison.pdf")

    if loso_df is not None:
        fig6_loso_regression_to_mean(loso_df, out_dir / "fig6_loso_regression_to_mean.pdf")
        fig7_loso_error_vs_deviation(loso_df, out_dir / "fig7_loso_error_vs_deviation.pdf")

    fig8_naive_vs_model_vs_calibrated(df_sbp, df_dbp, out_dir / "fig8_naive_vs_model_vs_calibrated.pdf")

    print(f"\n[OK] All figures saved to {out_dir}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
