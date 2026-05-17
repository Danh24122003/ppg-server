"""
SpO2 software-only fixes exploration: 5 transforms vs baseline.

this work diagnostic (eval-only). User constraint: cannot hardware re-test.
this work-3 confirmed cohort R cluster 0.40-0.50 -> Maxim quadratic flat ->
SpO2 stuck 99% (algorithm-level fixes failed). This script explores
software-only output transforms to map cohort SpO2 from stuck 99% to
realistic 95-98% range.

5 approaches tested:
  M1 (A/B/C). R-rescale variants -- R x {1.20, 1.30, 1.45} then Maxim quadratic.
  M2. SpO2 affine output: 0.95 * spo2_raw + 2.0
  M3. Custom piecewise: low-R region (< 0.50) replaced with linear curve.
  M4. Quantile mapping to literature N(97, 1.5) (Hafen & Sharma 2023 prior).
  M5. Multi-formula ensemble: mean(Maxim, Coniferconifer, Bent, Webster).

CRITICAL CAVEATS:
1. All 5 transforms are COSMETIC. No ground truth (co-oximetry) available
   to validate which gives "correct" SpO2.
2. Cohort R cluster 0.40-0.50 is outside calibration domain of all 4
   reference formulas (M5 ensemble). Each individual formula is
   extrapolation; average of extrapolations != truth.
3. Quantile mapping (M4) assumes cohort = healthy young distribution
   (SpO2 ~ N(97, 1.5)). This prior applies only if assumption holds;
   outlier subjects (S04 Stage 1 HTN low PI, S11 white-coat, S07
   hypotensive PI 0.58%) violate the assumption.
4. R-rescale (M1) and ensemble (M5) treat outlier R values (S04 R=0.77,
   S11 R=0.85) the same as cohort cluster -- may push outliers too low.
5. Piecewise (M3) has discontinuous junction at R=0.50 (~2 pp jump from
   custom 96.0 to Maxim 98.76). Acceptable for cohort R<0.50 but visible
   at boundary.

Production chain (Backend/main.py:calculate_spo2_v23) NOT changed by this
script. Decision to adopt any method requires user judgment based on this
eval + hardware verification (TEST 1 deferred).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Methods
# ---------------------------------------------------------------------------

def baseline_maxim(R):
    """Current production: Maxim MAXREFDES117 quadratic."""
    if R is None or pd.isna(R) or R < 0.4 or R > 1.4:
        return None
    spo2 = -45.060 * R * R + 30.354 * R + 94.845
    return float(np.clip(spo2, 70, 100))


# === Approach 1: R-rescale aggressive (3 scales) ===

def method_1A_rescale_120(R):
    if R is None or pd.isna(R) or R < 0.4 or R > 1.4:
        return None
    R_eff = R * 1.20
    spo2 = -45.060 * R_eff * R_eff + 30.354 * R_eff + 94.845
    return float(np.clip(spo2, 70, 100))


def method_1B_rescale_130(R):
    if R is None or pd.isna(R) or R < 0.4 or R > 1.4:
        return None
    R_eff = R * 1.30
    spo2 = -45.060 * R_eff * R_eff + 30.354 * R_eff + 94.845
    return float(np.clip(spo2, 70, 100))


def method_1C_rescale_145(R):
    if R is None or pd.isna(R) or R < 0.4 or R > 1.4:
        return None
    R_eff = R * 1.45
    spo2 = -45.060 * R_eff * R_eff + 30.354 * R_eff + 94.845
    return float(np.clip(spo2, 70, 100))


# === Approach 2: SpO2 affine output transform ===

def method_2_affine(spo2_baseline):
    """SpO2_display = 0.95 * SpO2_raw + 2.0
    R=0.43 -> baseline 99.5 -> 96.5 cohort target.
    """
    if spo2_baseline is None or pd.isna(spo2_baseline):
        return None
    return float(np.clip(0.95 * spo2_baseline + 2.0, 70, 100))


# === Approach 3: Custom piecewise low-R replacement ===

def method_3_piecewise(R):
    """For R < 0.50 (plateau region): custom linear curve.
    For R >= 0.50: keep Maxim quadratic.

    Custom: SpO2 = 97.0 - 5.0 * (R - 0.40)
      R=0.40 -> 97.0
      R=0.45 -> 96.5
      R=0.50 -> 96.0  (Maxim gives 98.76, jump 2.76 pp)
    """
    if R is None or pd.isna(R) or R < 0.4 or R > 1.4:
        return None
    if R < 0.50:
        spo2 = 97.0 - 5.0 * (R - 0.40)
    else:
        spo2 = -45.060 * R * R + 30.354 * R + 94.845
    return float(np.clip(spo2, 70, 100))


# === Approach 4: Quantile mapping to literature N(97, 1.5) ===

def make_method_4_quantile(all_baseline_spo2_values):
    """Closure that maps cohort baseline SpO2 percentile to N(97, 1.5).

    Assumes cohort = healthy young at sea level = SpO2 ~ N(97, 1.5)
    per Hafen & Sharma 2023 StatPearls.

    Implementation: rank-based mapping.
      raw_percentile = rank(value) / N
      mapped = scipy.stats.norm.ppf(raw_percentile, loc=97, scale=1.5)
    """
    valid = np.array([
        v for v in all_baseline_spo2_values
        if v is not None and not (isinstance(v, float) and np.isnan(v))
    ])
    valid_sorted = np.sort(valid)
    N = len(valid_sorted)

    def transform(spo2_baseline):
        if spo2_baseline is None or pd.isna(spo2_baseline):
            return None
        if N == 0:
            return None
        rank = np.searchsorted(valid_sorted, spo2_baseline, side='right') / N
        rank = float(np.clip(rank, 0.001, 0.999))
        mapped = norm.ppf(rank, loc=97.0, scale=1.5)
        return float(np.clip(mapped, 70, 100))

    return transform


# === Approach 5: Multi-formula ensemble ===

def method_5_ensemble(R):
    """Average of 4 cited formulas:
      - Maxim MAXREFDES117 quadratic (current)
      - Coniferconifer linear (-23.3*(R-0.4)+100, GitHub esp32_max30102)
      - Bent et al. 2021 (-21.54*R + 106.69, PMC8699050, sternum)
      - Webster 1997 (110 - 25*R, transmission textbook)

    Honest caveat: All 4 formulas tested on project R values is extrapolation
    across calibration domain (transmission vs reflectance, peak-valley vs RMS
    AC method). Average of 4 invalid extrapolations != truth, but smoother
    than single formula.
    """
    if R is None or pd.isna(R) or R < 0.4 or R > 1.4:
        return None
    spo2_maxim = -45.060 * R * R + 30.354 * R + 94.845
    spo2_conifer = -23.3 * (R - 0.4) + 100.0
    spo2_bent = -21.54 * R + 106.69
    spo2_webster = 110.0 - 25.0 * R
    avg = (spo2_maxim + spo2_conifer + spo2_bent + spo2_webster) / 4.0
    return float(np.clip(avg, 70, 100))


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

METHODS = [
    'M0_baseline',
    'M1A_rescale_1.20',
    'M1B_rescale_1.30',
    'M1C_rescale_1.45',
    'M2_affine',
    'M3_piecewise',
    'M4_quantile',
    'M5_ensemble',
]


def analyze(df):
    """Apply all 8 methods (baseline + 5 fixes) on each window."""
    df['M0_baseline'] = df['project_R'].apply(baseline_maxim)
    df['M1A_rescale_1.20'] = df['project_R'].apply(method_1A_rescale_120)
    df['M1B_rescale_1.30'] = df['project_R'].apply(method_1B_rescale_130)
    df['M1C_rescale_1.45'] = df['project_R'].apply(method_1C_rescale_145)
    df['M2_affine'] = df['M0_baseline'].apply(method_2_affine)
    df['M3_piecewise'] = df['project_R'].apply(method_3_piecewise)
    method_4 = make_method_4_quantile(df['M0_baseline'].values)
    df['M4_quantile'] = df['M0_baseline'].apply(method_4)
    df['M5_ensemble'] = df['project_R'].apply(method_5_ensemble)
    return df


def make_summary(df):
    """Per-file x per-method medians."""
    rows = []
    for fname, grp in df.groupby('file'):
        row = {'file': fname, 'n_windows': len(grp)}
        for m in METHODS:
            valid = grp[m].dropna()
            row[f'{m}_n_valid'] = len(valid)
            row[f'{m}_median'] = float(valid.median()) if len(valid) else None
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

OUTLIER_FILES = {
    'S04_01_20260501_175617_cleaned.csv',
    'S11_20260506_183827_stripped_cleaned.csv',
    'S07_01_20260429_145144_cleaned.csv',
}


def make_figures(df_out, summary, outdir):
    # Figure 1: 2x4 histogram grid
    fig, axes = plt.subplots(2, 4, figsize=(20, 9), sharex=True, sharey=False)
    axes = axes.flatten()
    for i, m in enumerate(METHODS):
        ax = axes[i]
        valid = df_out[m].dropna()
        if len(valid) == 0:
            ax.set_title(f'{m} (no data)')
            continue
        ax.hist(valid, bins=40, range=(80, 100), color='steelblue', edgecolor='black', alpha=0.7)
        med = float(valid.median())
        p10 = float(valid.quantile(0.1))
        p90 = float(valid.quantile(0.9))
        ax.axvline(med, color='red', linestyle='-', linewidth=2, label=f'median={med:.2f}')
        ax.axvline(p10, color='orange', linestyle='--', linewidth=1.2, label=f'p10={p10:.1f}')
        ax.axvline(p90, color='orange', linestyle='--', linewidth=1.2, label=f'p90={p90:.1f}')
        ax.set_title(f'{m}\nmedian={med:.2f}')
        ax.set_xlabel('SpO2 (%)')
        ax.set_ylabel('count')
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(True, alpha=0.3)
    fig.suptitle('Cohort SpO2 distribution: 8 methods (baseline + 7 variants)', fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(outdir / 'fig_spo2_software_fixes_histograms.pdf', dpi=120)
    plt.close(fig)

    # Figure 2: per-subject grouped bar chart
    summary_sorted = summary.sort_values('file').reset_index(drop=True)
    n_files = len(summary_sorted)
    n_methods = len(METHODS)
    x = np.arange(n_files)
    bar_w = 0.8 / n_methods

    fig, ax = plt.subplots(figsize=(max(16, n_files * 0.8), 8))
    colors = plt.cm.tab10(np.linspace(0, 1, n_methods))
    for j, m in enumerate(METHODS):
        col = f'{m}_median'
        vals = summary_sorted[col].astype(float).values
        ax.bar(x + j * bar_w, vals, bar_w, label=m, color=colors[j], edgecolor='black', linewidth=0.3)

    # Highlight outlier subjects (red x-tick labels)
    labels = []
    label_colors = []
    for fname in summary_sorted['file']:
        short = fname.replace('_cleaned.csv', '').replace('_stripped', '')
        labels.append(short[:20])
        label_colors.append('red' if fname in OUTLIER_FILES else 'black')
    ax.set_xticks(x + bar_w * (n_methods - 1) / 2)
    ax.set_xticklabels(labels, rotation=75, ha='right', fontsize=7)
    for tick, color in zip(ax.get_xticklabels(), label_colors):
        tick.set_color(color)

    ax.set_ylabel('SpO2 median (%)')
    ax.set_ylim(80, 102)
    ax.set_title('Per-subject SpO2 median across 8 methods (red labels = outlier subjects)')
    ax.legend(loc='lower right', fontsize=8, ncol=4)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    fig.savefig(outdir / 'fig_spo2_software_fixes_per_subject.pdf', dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    src = Path(__file__).resolve().parent / 'eval_spo2_sparkfun_chain_results.csv'
    if not src.exists():
        print(f'ERROR: {src} not found. Run this work first.')
        return
    df = pd.read_csv(src)
    print(f'Loaded {len(df)} windows from {df["file"].nunique()} files.')

    df_out = analyze(df)
    summary = make_summary(df_out)

    results_path = Path(__file__).resolve().parent / 'eval_spo2_software_fixes_results.csv'
    summary_path = Path(__file__).resolve().parent / 'eval_spo2_software_fixes_summary.csv'
    df_out.to_csv(results_path, index=False)
    summary.to_csv(summary_path, index=False)

    print('\n=== Cohort aggregate medians ===')
    for m in METHODS:
        valid = df_out[m].dropna()
        if len(valid):
            print(
                f'  {m:<25}  median={valid.median():.2f}  '
                f'p10={valid.quantile(0.1):.1f}  '
                f'p90={valid.quantile(0.9):.1f}  '
                f'n={len(valid)}'
            )
        else:
            print(f'  {m:<25}  no valid values')

    print('\n=== Outlier subjects (S04, S11, S07_01) ===')
    outliers = [
        'S04_01_20260501_175617_cleaned.csv',
        'S11_20260506_183827_stripped_cleaned.csv',
        'S07_01_20260429_145144_cleaned.csv',
    ]
    for o in outliers:
        row = summary[summary['file'] == o]
        if len(row) > 0:
            r = row.iloc[0]
            print(f'\n  {o[:45]}:')
            for m in METHODS:
                v = r.get(f'{m}_median')
                if v is not None and not pd.isna(v):
                    print(f'    {m:<25} = {v:.2f}')
        else:
            print(f'\n  {o[:45]}: NOT FOUND in summary')

    figures_dir = Path(__file__).resolve().parent / 'figures'
    figures_dir.mkdir(exist_ok=True)
    make_figures(df_out, summary, figures_dir)
    print(f'\nFigures saved: {figures_dir}/fig_spo2_software_fixes_*.pdf')


if __name__ == '__main__':
    main()
