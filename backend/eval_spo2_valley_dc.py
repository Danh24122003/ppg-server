"""
SpO2 multi-method comparison: address root cause R cluster low.

this work diagnostic (eval-only). Tests hypothesis: project's hybrid DC formula
inflates denominator -> R cluster 0.40-0.50 -> quadratic flat region -> SpO2 stuck 99%.

Methods tested:
  A. Project current: Butterworth bandpass [0.5, 4Hz] RMS AC + hybrid DC
     ((median(raw) + mean(lowpass))/2)
  B. SparkFun peak-valley: max-min AC + (max+min)/2 DC (from this work)
  C. Maxim/aromring valley-interp: AC = peak - linear_baseline_between_valleys,
     DC = (valley[i] + valley[i+1])/2 -- community gold standard for reflectance
  D. ml-engineer rescale: R_project x 1.152 -> Maxim quadratic
     (cosmetic, mapping cohort R median 0.434 to sweet-spot R 0.5)

For each method, compute R + Maxim quadratic SpO2.
For Method C also compute coniferconifer linear formula
`SpO2 = -23.3*(R-0.4) + 100` (purpose-built for reflectance R 0.4-0.7).

Channel swap diagnostic: verify mean(raw_IR) > mean(raw_RED) per file.
40% of cheap MH-ET LIVE MAX30102 boards have FIFO swap -> invert R interpretation.

CAVEATS:
1. NO co-oximetry GT -- cannot determine which method is most accurate.
2. Different methods may have different validity rates (different rejection criteria).
3. Coniferconifer formula is from a single GitHub repo, NOT peer-reviewed.
   Empirically calibrated, citable as community alternative.
4. Production chain (Backend/main.py:calculate_spo2_v23) NOT changed.

If Method C shows R cluster shift to 0.50+ AND cohort SpO2 medians 95-98%,
this validates root-cause fix path -> next step would be production migration
(separate PR, with full test coverage + cross-server sync).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

sys.path.insert(0, str(Path(__file__).resolve().parent))
from main import bandpass_filter, calculate_spo2_v23  # noqa: E402  production reuse
from eval_spo2_sparkfun_chain import compute_spo2_sparkfun  # noqa: E402  this work reuse


def diagnose_channel_swap(ir, red):
    """
    Per-file diagnostic: physical reality on reflectance MAX30102 finger,
    IR 880 nm penetrates deeper -> mean(raw_IR) > mean(raw_RED).
    If reversed -> possible FIFO swap (40% of MH-ET LIVE clones).

    Returns: ('normal' | 'possibly_swapped' | 'inconclusive', mean_ir, mean_red)
    """
    mean_ir = float(np.mean(ir))
    mean_red = float(np.mean(red))
    if mean_ir > mean_red * 1.05:
        return 'normal', mean_ir, mean_red
    elif mean_red > mean_ir * 1.05:
        return 'possibly_swapped', mean_ir, mean_red
    else:
        return 'inconclusive', mean_ir, mean_red


def compute_R_method_A_project(ir, red, fs):
    """Project current: bandpass RMS AC + hybrid DC + Maxim quadratic."""
    spo2, R, _acdc_ir, _acdc_red, status = calculate_spo2_v23(ir, red, fs)
    return R if R else None, spo2, status


def compute_R_method_B_sparkfun_peakvalley(ir, red, fs):
    """SparkFun: peak-valley AC + (max+min)/2 DC. R + Maxim quadratic SpO2."""
    spo2_lookup, R, _ac_ir, _ac_red, _dc_ir, _dc_red, _n_pulses, status = \
        compute_spo2_sparkfun(ir, red, fs)
    if status != 'ok' or R is None or R <= 0:
        return None, None, status
    # Apply Maxim quadratic on SparkFun's R (not lookup), to isolate AC/DC method
    # contribution from calibration-curve contribution.
    spo2_maxim = -45.060 * R * R + 30.354 * R + 94.845
    spo2_maxim = float(np.clip(spo2_maxim, 70.0, 100.0))
    return float(R), spo2_maxim, status


def compute_R_method_C_valley_interp(ir, red, fs):
    """
    Maxim original / doug-burrell / aromring chain:
    Linear-interpolated baseline between consecutive valleys.
    AC at peak = peak - interpolated_baseline_at_peak_position
    DC = (valley[i] + valley[i+1]) / 2

    Returns: (R, spo2_maxim, spo2_conifer, status)
    """
    ir = np.asarray(ir, dtype=float)
    red = np.asarray(red, dtype=float)

    # Bandpass IR for valley detection only (signal cleanup), use RAW for AC/DC calc
    ir_filt = bandpass_filter(ir, fs, 0.5, 4.0)
    std_filt = float(np.std(ir_filt))
    if std_filt <= 0:
        return None, None, None, 'flat_signal'

    # Detect valleys = negated peaks
    valleys_ir, _ = find_peaks(
        -ir_filt,
        distance=int(fs * 0.3),
        prominence=std_filt * 0.3,
    )
    if len(valleys_ir) < 3:
        return None, None, None, 'too_few_valleys'

    ac_ir_list = []
    dc_ir_list = []
    ac_red_list = []
    dc_red_list = []
    for i in range(len(valleys_ir) - 1):
        v1, v2 = int(valleys_ir[i]), int(valleys_ir[i + 1])
        if v2 - v1 < 10:
            continue
        # IR: peak position within [v1, v2]
        peak_pos_ir = v1 + int(np.argmax(ir[v1:v2]))
        frac_ir = (peak_pos_ir - v1) / (v2 - v1)
        baseline_ir = ir[v1] * (1.0 - frac_ir) + ir[v2] * frac_ir
        ac_ir_beat = float(ir[peak_pos_ir] - baseline_ir)
        dc_ir_beat = float((ir[v1] + ir[v2]) / 2.0)
        if ac_ir_beat <= 0 or dc_ir_beat <= 0:
            continue
        # Red: peak position within same beat (may differ slightly from IR)
        peak_pos_red = v1 + int(np.argmax(red[v1:v2]))
        frac_red = (peak_pos_red - v1) / (v2 - v1)
        baseline_red = red[v1] * (1.0 - frac_red) + red[v2] * frac_red
        ac_red_beat = float(red[peak_pos_red] - baseline_red)
        dc_red_beat = float((red[v1] + red[v2]) / 2.0)
        if ac_red_beat <= 0 or dc_red_beat <= 0:
            continue
        ac_ir_list.append(ac_ir_beat)
        dc_ir_list.append(dc_ir_beat)
        ac_red_list.append(ac_red_beat)
        dc_red_list.append(dc_red_beat)

    if len(ac_ir_list) == 0:
        return None, None, None, 'no_valid_beats'

    ac_ir = float(np.mean(ac_ir_list))
    dc_ir = float(np.mean(dc_ir_list))
    ac_red = float(np.mean(ac_red_list))
    dc_red = float(np.mean(dc_red_list))

    if dc_ir <= 0 or dc_red <= 0 or ac_ir <= 0 or ac_red <= 0:
        return None, None, None, 'invalid_R'

    R = (ac_red / dc_red) / (ac_ir / dc_ir)

    # Maxim quadratic
    spo2_maxim = -45.060 * R * R + 30.354 * R + 94.845
    spo2_maxim = float(np.clip(spo2_maxim, 70.0, 100.0))

    # Coniferconifer linear: SpO2 = -23.3*(R - 0.4) + 100
    # Source: github.com/coniferconifer/ESP32_MAX30102_simple-SpO2_plotter
    # Empirical, purpose-built for reflectance R 0.4-0.7. NOT peer-reviewed.
    spo2_conifer = -23.3 * (R - 0.4) + 100.0
    spo2_conifer = float(np.clip(spo2_conifer, 70.0, 100.0))

    return float(R), spo2_maxim, spo2_conifer, 'ok'


def compute_R_method_D_rescale(ir, red, fs):
    """
    ml-engineer recommendation: project R x 1.152 -> Maxim quadratic.
    Cosmetic rescale to map cohort R median (~0.434) to sweet spot R~0.5.
    """
    R_proj, _spo2_proj, status = compute_R_method_A_project(ir, red, fs)
    if R_proj is None or status != 'ok':
        return None, None, status
    R_eff = R_proj * 1.152
    spo2 = -45.060 * R_eff * R_eff + 30.354 * R_eff + 94.845
    spo2 = float(np.clip(spo2, 70.0, 100.0))
    return float(R_eff), spo2, 'ok'


def analyze_file(csv_path, fs=100, win_sec=5):
    """Run all 4 methods per win_sec window, return per-window dataframe."""
    df_raw = pd.read_csv(csv_path, comment='#', usecols=['timestamp_ms', 'ir', 'red'])

    # Channel swap diagnostic on full file
    swap_status, mean_ir_full, mean_red_full = diagnose_channel_swap(
        df_raw['ir'].values, df_raw['red'].values
    )

    win_n = win_sec * fs
    n_win = len(df_raw) // win_n
    rows = []
    for i in range(n_win):
        ir = df_raw['ir'].values[i * win_n:(i + 1) * win_n].astype(float)
        red = df_raw['red'].values[i * win_n:(i + 1) * win_n].astype(float)

        R_a, spo2_a, status_a = compute_R_method_A_project(ir, red, fs)
        R_b, spo2_b, status_b = compute_R_method_B_sparkfun_peakvalley(ir, red, fs)
        R_c, spo2_c_maxim, spo2_c_conifer, status_c = compute_R_method_C_valley_interp(ir, red, fs)
        R_d, spo2_d, status_d = compute_R_method_D_rescale(ir, red, fs)

        rows.append({
            'file': csv_path.name,
            'window_idx': i,
            'channel_swap': swap_status,
            'mean_ir_raw': mean_ir_full,
            'mean_red_raw': mean_red_full,
            'R_A': R_a, 'spo2_A': spo2_a, 'status_A': status_a,
            'R_B': R_b, 'spo2_B': spo2_b, 'status_B': status_b,
            'R_C': R_c, 'spo2_C_maxim': spo2_c_maxim, 'spo2_C_conifer': spo2_c_conifer, 'status_C': status_c,
            'R_D': R_d, 'spo2_D': spo2_d, 'status_D': status_d,
        })
    return pd.DataFrame(rows)


def make_summary(df_all):
    """Group by file -> per-method medians + n_valid + channel swap status."""
    rows = []
    for fname, grp in df_all.groupby('file'):
        def med(col):
            s = grp[col].dropna()
            return float(s.median()) if len(s) else None

        def nval(col):
            return int(grp[col].notna().sum())

        rows.append({
            'file': fname,
            'n_windows': len(grp),
            'channel_swap': grp['channel_swap'].iloc[0],
            'mean_ir_raw': float(grp['mean_ir_raw'].iloc[0]),
            'mean_red_raw': float(grp['mean_red_raw'].iloc[0]),
            'A_R_med': med('R_A'),
            'A_spo2_med': med('spo2_A'),
            'A_n_valid': nval('spo2_A'),
            'B_R_med': med('R_B'),
            'B_spo2_med': med('spo2_B'),
            'B_n_valid': nval('spo2_B'),
            'C_R_med': med('R_C'),
            'C_spo2_maxim_med': med('spo2_C_maxim'),
            'C_spo2_conifer_med': med('spo2_C_conifer'),
            'C_n_valid': nval('spo2_C_maxim'),
            'D_R_med': med('R_D'),
            'D_spo2_med': med('spo2_D'),
            'D_n_valid': nval('spo2_D'),
        })
    return pd.DataFrame(rows)


def make_figures(df_all, outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Figure 1: R distribution histogram overlay (4 methods)
    fig, ax = plt.subplots(figsize=(11, 6))
    bins = np.linspace(0.0, 1.5, 60)
    for col, label, color in [
        ('R_A', 'A: project (hybrid DC + RMS AC)', 'C0'),
        ('R_B', 'B: SparkFun (peak-valley)', 'C1'),
        ('R_C', 'C: valley-interp DC', 'C2'),
        ('R_D', 'D: project R x 1.152', 'C3'),
    ]:
        vals = df_all[col].dropna().values
        if len(vals) == 0:
            continue
        ax.hist(vals, bins=bins, alpha=0.45, label=f'{label} (n={len(vals)})', color=color)
    ax.axvspan(0.4, 0.5, color='gray', alpha=0.15, label='Cohort cluster region 0.40-0.50')
    ax.axvline(0.5, ls='--', color='gray', alpha=0.5)
    ax.set_xlabel('R-ratio')
    ax.set_ylabel('Window count')
    ax.set_title('R distribution across 4 SpO2 methods (cohort = 25 samples)')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(outdir / 'fig_spo2_valley_dc_R_distribution.pdf', dpi=120)
    plt.close(fig)

    # Figure 2: SpO2 cohort distribution per method (5 series)
    fig, ax = plt.subplots(figsize=(11, 6))
    series = [
        ('A: project', df_all['spo2_A'].dropna().values, 'C0'),
        ('B: SparkFun + Maxim', df_all['spo2_B'].dropna().values, 'C1'),
        ('C: valley + Maxim', df_all['spo2_C_maxim'].dropna().values, 'C2'),
        ('C: valley + Conifer', df_all['spo2_C_conifer'].dropna().values, 'C4'),
        ('D: rescale + Maxim', df_all['spo2_D'].dropna().values, 'C3'),
    ]
    data = [s[1] for s in series]
    labels = [f'{s[0]}\nn={len(s[1])}' for s in series]
    ax.boxplot(data, tick_labels=labels, showmeans=True, meanline=True, widths=0.6)
    ax.axhline(95, ls='--', color='gray', alpha=0.5, label='Healthy floor 95%')
    ax.axhline(99, ls=':', color='gray', alpha=0.5, label='Stuck region 99%')
    ax.set_ylabel('SpO2 (%)')
    ax.set_title('SpO2 cohort distribution per method (cohort = 25 samples)')
    ax.set_ylim(70, 102)
    ax.legend(fontsize=9, loc='lower left')
    ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    fig.savefig(outdir / 'fig_spo2_valley_dc_method_comparison.pdf', dpi=120)
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

    out_results = root / 'Backend code' / 'eval_spo2_valley_dc_results.csv'
    out_summary = root / 'Backend code' / 'eval_spo2_valley_dc_summary.csv'
    df_all.to_csv(out_results, index=False)
    df_summary.to_csv(out_summary, index=False)

    print('\n=== Channel swap audit (per file) ===')
    swap_counts = df_all.groupby('file')['channel_swap'].first().value_counts()
    print(swap_counts.to_string())

    def _med(col):
        s = df_all[col].dropna()
        return float(s.median()) if len(s) else float('nan')

    def _n(col):
        return int(df_all[col].notna().sum())

    print('\n=== Cohort aggregate medians ===')
    print(f'Method A (project hybrid DC + RMS AC + Maxim):     '
          f'R={_med("R_A"):.3f}  SpO2={_med("spo2_A"):.2f}  n={_n("spo2_A")}')
    print(f'Method B (SparkFun peak-valley + Maxim):           '
          f'R={_med("R_B"):.3f}  SpO2={_med("spo2_B"):.2f}  n={_n("spo2_B")}')
    print(f'Method C (valley-interp DC + Maxim):               '
          f'R={_med("R_C"):.3f}  SpO2={_med("spo2_C_maxim"):.2f}  n={_n("spo2_C_maxim")}')
    print(f'Method C (valley-interp DC + coniferconifer):      '
          f'R={_med("R_C"):.3f}  SpO2={_med("spo2_C_conifer"):.2f}')
    print(f'Method D (project R x 1.152 + Maxim):              '
          f'R={_med("R_D"):.3f}  SpO2={_med("spo2_D"):.2f}  n={_n("spo2_D")}')

    figures_dir = root / 'Backend code' / 'figures'
    make_figures(df_all, figures_dir)
    print(f'\nFigures saved: {figures_dir}/fig_spo2_valley_dc_*.pdf')
    print(f'Per-window CSV: {out_results}')
    print(f'Per-file summary CSV: {out_summary}')


if __name__ == '__main__':
    main()
