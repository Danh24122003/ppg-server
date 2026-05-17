"""
SpO2 chain comparison: Project vs SparkFun MAX3010x.

this work diagnostic (eval-only, no production change). Hypothesis: SparkFun
chain (peak-valley AC + pulse-mean DC + Maxim lookup table) gives more
realistic SpO2 (~95-97%) than project chain (RMS bandpass AC + hybrid DC +
Maxim quadratic approximation) which stucks at ~99% on the cohort R cluster
0.40-0.50.

CAVEATS:
1. SparkFun chain implemented from public source code
   (github.com/sparkfun/SparkFun_MAX3010x_Sensor_Library, MIT license).
   May differ slightly from running on actual ESP32 SparkFun firmware due to
   integer arithmetic + adaptive thresholds in C++ implementation.

2. Lookup table USPO2_TABLE behavior at low R (R<0.5) is empirical Maxim
   choice, NOT clinically validated. Maxim's documentation explicitly states
   these are 'approximations, not clinically validated values'.

3. Cohort R cluster 0.40-0.50 is OUTSIDE Maxim's typical test range (R~0.5-1.0
   for healthy adults transmission probe). Both project quadratic AND SparkFun
   lookup are extrapolating in this range -- neither is 'correct', they just
   extrapolate differently.

4. No co-oximetry GT -- cannot determine which chain is closer to truth.
   This script ESTABLISHES that the two chains diverge in the cohort R range,
   but does NOT establish which is more accurate.

Production chain (Backend/main.py:calculate_spo2_v23) NOT changed by this script.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

sys.path.insert(0, str(Path(__file__).resolve().parent))
from main import calculate_spo2_v23  # noqa: E402  production chain reuse


# SparkFun MAX3010x lookup table.
# Source: github.com/sparkfun/SparkFun_MAX3010x_Sensor_Library/blob/master/src/spo2_algorithm.cpp
# NOTE: SparkFun C++ declares `uch_spo2_table[184]` but the meaningful values stop
# at index 175 (last entry = 1). Indices 176-183 in the C array are zero-padding
# (would map to SpO2=0, non-physiological). Python implementation uses 176 entries
# with explicit lookup_oor guard for R >= 1.76 — boundary differs from C only in
# the (1.76, 1.83] R range which is non-physiological (SpO2 < 1%) and never
# observed in cohort (R range 0.33-0.94). For verbatim defense, see test
# `test_table_length_is_184_sparkfun_source` in test_spo2_sparkfun_chain.py.
USPO2_TABLE = np.array([
    95, 95, 95, 96, 96, 96, 97, 97, 97, 97, 97, 98, 98, 98, 98, 98, 99, 99, 99, 99,
    99, 99, 99, 99, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
    100, 100, 100, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 98, 98, 98, 98, 98, 98, 97,
    97, 97, 97, 96, 96, 96, 96, 95, 95, 95, 94, 94, 94, 93, 93, 93, 92, 92, 92, 91,
    91, 90, 90, 89, 89, 89, 88, 88, 87, 87, 86, 86, 85, 85, 84, 84, 83, 82, 82, 81,
    81, 80, 80, 79, 78, 78, 77, 76, 76, 75, 74, 74, 73, 72, 72, 71, 70, 69, 69, 68,
    67, 66, 66, 65, 64, 63, 62, 62, 61, 60, 59, 58, 57, 56, 56, 55, 54, 53, 52, 51,
    50, 49, 48, 47, 46, 45, 44, 43, 41, 40, 39, 38, 37, 36, 35, 33, 32, 31, 29, 28,
    27, 25, 24, 22, 21, 19, 18, 16, 14, 13, 11, 9, 7, 5, 3, 1
], dtype=np.float64)


def compute_spo2_sparkfun(ir, red, fs):
    """
    Replicate SparkFun MAX3010x spo2_algorithm.cpp chain.

    Returns: (spo2, R, ac_ir, ac_red, dc_ir, dc_red, n_pulses, status)
        status: 'ok' | 'too_few_pulses' | 'invalid_R' | 'lookup_oor'
    """
    ir = np.asarray(ir, dtype=float)
    red = np.asarray(red, dtype=float)

    # Step 1: DC removal -- subtract mean
    ir_centered = ir - np.mean(ir)
    red_centered = red - np.mean(red)

    # Step 2: 4-tap moving average smoothing
    kernel = np.ones(4) / 4.0
    ir_smooth = np.convolve(ir_centered, kernel, mode='same')
    red_smooth = np.convolve(red_centered, kernel, mode='same')

    # Step 3: Peak detection on -ir_smooth (Maxim detects DC drops = pulses)
    std_ir = float(np.std(ir_smooth))
    if std_ir <= 0:
        return None, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 'too_few_pulses'

    peaks_ir, _ = find_peaks(
        -ir_smooth,
        distance=int(fs * 0.3),
        prominence=std_ir * 0.5,
    )

    if len(peaks_ir) < 3:
        return None, 0.0, 0.0, 0.0, 0.0, 0.0, len(peaks_ir), 'too_few_pulses'

    # Step 4: per-pulse AC = max-min, DC = (max+min)/2 on RAW signal
    ac_ir_list, dc_ir_list = [], []
    ac_red_list, dc_red_list = [], []

    for i in range(len(peaks_ir) - 1):
        start, end = int(peaks_ir[i]), int(peaks_ir[i + 1])
        if end - start < 10:
            continue

        ir_seg = ir[start:end]
        red_seg = red[start:end]

        ir_max, ir_min = float(np.max(ir_seg)), float(np.min(ir_seg))
        red_max, red_min = float(np.max(red_seg)), float(np.min(red_seg))

        ac_ir_list.append(ir_max - ir_min)
        dc_ir_list.append((ir_max + ir_min) / 2.0)
        ac_red_list.append(red_max - red_min)
        dc_red_list.append((red_max + red_min) / 2.0)

    if len(ac_ir_list) == 0:
        return None, 0.0, 0.0, 0.0, 0.0, 0.0, len(peaks_ir), 'too_few_pulses'

    ac_ir = float(np.mean(ac_ir_list))
    dc_ir = float(np.mean(dc_ir_list))
    ac_red = float(np.mean(ac_red_list))
    dc_red = float(np.mean(dc_red_list))

    if dc_ir <= 0 or dc_red <= 0 or ac_ir <= 0 or ac_red <= 0:
        return None, 0.0, ac_ir, ac_red, dc_ir, dc_red, len(peaks_ir), 'invalid_R'

    R = (ac_red / dc_red) / (ac_ir / dc_ir)

    # Step 5: lookup table
    table_idx = int(round(R * 100))
    if table_idx < 0 or table_idx >= len(USPO2_TABLE):
        return None, R, ac_ir, ac_red, dc_ir, dc_red, len(peaks_ir), 'lookup_oor'

    spo2 = float(USPO2_TABLE[table_idx])
    return spo2, R, ac_ir, ac_red, dc_ir, dc_red, len(peaks_ir), 'ok'


def analyze_file(csv_path, fs=100, win_sec=5):
    """Run BOTH chains per 5s window, return per-window dataframe."""
    df_raw = pd.read_csv(csv_path, comment='#', usecols=['timestamp_ms', 'ir', 'red'])
    win_n = win_sec * fs
    n_win = len(df_raw) // win_n
    rows = []

    for i in range(n_win):
        ir = df_raw['ir'].values[i * win_n:(i + 1) * win_n].astype(float)
        red = df_raw['red'].values[i * win_n:(i + 1) * win_n].astype(float)

        # Chain A: project (current main.py)
        spo2_p, R_p, acdc_ir_p, acdc_red_p, rej_p = calculate_spo2_v23(ir, red, fs)

        # Chain B: SparkFun
        spo2_s, R_s, ac_ir_s, ac_red_s, dc_ir_s, dc_red_s, n_pulses_s, status_s = \
            compute_spo2_sparkfun(ir, red, fs)

        rows.append({
            'file': csv_path.name,
            'window_idx': i,
            'project_spo2': spo2_p,
            'project_R': R_p,
            'project_status': rej_p,
            'sparkfun_spo2': spo2_s,
            'sparkfun_R': R_s,
            'sparkfun_status': status_s,
            'sparkfun_n_pulses': n_pulses_s,
        })
    return pd.DataFrame(rows)


def make_summary(df_all):
    """Group by file -> median project + sparkfun + R, with valid-only filtering."""
    rows = []
    for fname, grp in df_all.groupby('file'):
        valid_p = grp[grp['project_spo2'].notna()]
        valid_s = grp[grp['sparkfun_spo2'].notna()]
        rows.append({
            'file': fname,
            'n_windows': len(grp),
            'project_n_valid': len(valid_p),
            'sparkfun_n_valid': len(valid_s),
            'project_spo2_median': float(valid_p['project_spo2'].median()) if len(valid_p) else None,
            'sparkfun_spo2_median': float(valid_s['sparkfun_spo2'].median()) if len(valid_s) else None,
            'project_R_median': float(valid_p['project_R'].median()) if len(valid_p) else None,
            'sparkfun_R_median': float(valid_s['sparkfun_R'].median()) if len(valid_s) else None,
            'spo2_diff_median': (
                float(valid_s['sparkfun_spo2'].median()) - float(valid_p['project_spo2'].median())
                if len(valid_p) and len(valid_s) else None
            ),
        })
    return pd.DataFrame(rows)


def make_figures(df_summary, df_all, outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Figure 1: Per-subject SpO2 median grouped bar
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(df_summary))
    width = 0.35
    ax.bar(x - width / 2, df_summary['project_spo2_median'], width,
           label='Project (Maxim quadratic)', alpha=0.8)
    ax.bar(x + width / 2, df_summary['sparkfun_spo2_median'], width,
           label='SparkFun (lookup table)', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f.replace('_cleaned.csv', '').replace('.csv', '')[:18] for f in df_summary['file']],
        rotation=45, ha='right', fontsize=8,
    )
    ax.set_ylabel('SpO2 median (%)')
    ax.set_title('SpO2 per-subject: Project vs SparkFun chain on same samples')
    ax.legend()
    ax.set_ylim(80, 102)
    ax.axhline(y=95, ls='--', color='gray', alpha=0.5)
    ax.axhline(y=99, ls='--', color='gray', alpha=0.5)
    plt.tight_layout()
    fig.savefig(outdir / 'fig_spo2_sparkfun_comparison.pdf', dpi=120)
    plt.close(fig)

    # Figure 2: SpO2 vs R scatter, dual chain
    fig, ax = plt.subplots(figsize=(10, 6))
    valid_p = df_all[df_all['project_spo2'].notna()]
    valid_s = df_all[df_all['sparkfun_spo2'].notna()]
    ax.scatter(valid_p['project_R'], valid_p['project_spo2'],
               alpha=0.4, s=10, label=f'Project (n={len(valid_p)})')
    ax.scatter(valid_s['sparkfun_R'], valid_s['sparkfun_spo2'],
               alpha=0.4, s=10, label=f'SparkFun (n={len(valid_s)})', color='orange')
    ax.set_xlabel('R-ratio')
    ax.set_ylabel('SpO2 (%)')
    ax.set_title('SpO2 vs R: Project quadratic vs SparkFun lookup table')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(outdir / 'fig_spo2_sparkfun_scatter.pdf', dpi=120)
    plt.close(fig)


def main():
    root = Path(__file__).resolve().parent.parent
    cleaned_dir = root / 'collected_data' / 'cleaned'
    cleaned = sorted(cleaned_dir.glob('*_cleaned.csv'))
    extra = [
        root / 'collected_data' / 'S03_02_20260505_190801.csv',
        root / 'collected_data' / 'S10_02_20260503_155516.csv',
    ]
    files = list(cleaned) + [p for p in extra if p.exists()]

    print(f'Processing {len(files)} files...')
    all_results = []
    for f in files:
        try:
            df = analyze_file(f)
            all_results.append(df)
        except Exception as e:
            print(f'  SKIP {f.name}: {e}')
    df_all = pd.concat(all_results, ignore_index=True)
    df_summary = make_summary(df_all)

    out_results = root / 'Backend code' / 'eval_spo2_sparkfun_chain_results.csv'
    out_summary = root / 'Backend code' / 'eval_spo2_sparkfun_chain_summary.csv'
    df_all.to_csv(out_results, index=False)
    df_summary.to_csv(out_summary, index=False)

    print('\n=== Headline ===')
    print(df_summary[['file', 'project_spo2_median', 'sparkfun_spo2_median',
                      'spo2_diff_median', 'project_R_median', 'sparkfun_R_median']].to_string(index=False))

    print('\nCohort:')
    p_med = df_all['project_spo2'].dropna().median()
    s_med = df_all['sparkfun_spo2'].dropna().median()
    p_R = df_all['project_R'].dropna().median()
    s_R = df_all['sparkfun_R'].dropna().median()
    print(f'  Project median SpO2:  {p_med:.2f}' if p_med == p_med else '  Project median SpO2:  nan')
    print(f'  SparkFun median SpO2: {s_med:.2f}' if s_med == s_med else '  SparkFun median SpO2: nan')
    print(f'  Project R median:     {p_R:.3f}' if p_R == p_R else '  Project R median:     nan')
    print(f'  SparkFun R median:    {s_R:.3f}' if s_R == s_R else '  SparkFun R median:    nan')

    figures_dir = root / 'Backend code' / 'figures'
    make_figures(df_summary, df_all, figures_dir)
    print(f'\nFigures: {figures_dir}/fig_spo2_sparkfun_*.pdf')


if __name__ == '__main__':
    main()
