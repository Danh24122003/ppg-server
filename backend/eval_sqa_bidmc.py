"""Apply 7 SQI features pipeline ( batch +  per-beat) to BIDMC dataset.

Output:
1. CSV `eval_sqa_bidmc_results.csv` — per-recording stats
2. Console comparison: BIDMC ICU vs BP-protocol (17 self-collected) vs SQA-dedicated (4 S09)

Usage:
    python "backend/eval_sqa_bidmc.py"

Notes:
- BIDMC native fs = 125 Hz. We KEEP fs=125 (no resample) for BIDMC — preserves
  spectral resolution. the resampling design's resample is for self-collected only (drift fs).
- Each BIDMC recording = 8 minutes × 125 Hz = 60,000 samples. To keep per-beat
  template extraction tractable and align with batch-style analysis used for
  self-collected (5-min cleaned files), we analyze in chunks of CHUNK_SEC seconds
  and use the FIRST chunk for headline metrics.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks

ROOT = Path(__file__).resolve().parent  # backend/
sys.path.insert(0, str(ROOT))
from main import compute_sqi_features  # noqa: E402
from sqa_per_beat import compute_tsqi_per_beat, aggregate_tsqi  # noqa: E402


BIDMC_DIR = Path("external_data/bidmc")
DAY2_CSV = ROOT / "eval_sqa_preliminary_results.csv"
OUT_CSV = ROOT / "eval_sqa_bidmc_results.csv"
FS_BIDMC = 125  # native sample rate
CHUNK_SEC = 60  # analyze first 60s for headline metrics


def load_bidmc(path: Path) -> tuple[np.ndarray, int]:
    """Load BIDMC CSV. Returns (ppg_signal, fs).

    BIDMC columns: 'Time [s]', ' RESP', ' PLETH', ' V', ' AVR', ' II'
    """
    df = pd.read_csv(path)
    # Strip whitespace from column names (BIDMC has leading spaces)
    df.columns = [c.strip() for c in df.columns]
    if "PLETH" not in df.columns:
        raise ValueError(f"PLETH column not found. Columns: {list(df.columns)}")
    ppg = df["PLETH"].to_numpy(dtype=float)
    return ppg, FS_BIDMC


def analyze_bidmc(path: Path, chunk_sec: int = CHUNK_SEC) -> dict:
    """Run 6 batch SQI + 7 per-beat tSQI on first `chunk_sec` of recording."""
    ppg_full, fs = load_bidmc(path)
    n_samples_full = len(ppg_full)
    duration_full = n_samples_full / fs

    # Headline analysis on first chunk_sec
    n_chunk = min(int(chunk_sec * fs), n_samples_full)
    ppg_raw = ppg_full[:n_chunk]

    # Bandpass [0.5, 4.0] Hz at fs=125
    b, a = butter(4, [0.5, 4.0], btype="band", fs=fs)
    ppg_filt = filtfilt(b, a, ppg_raw)

    # Peak detection
    peaks, _ = find_peaks(ppg_filt, distance=int(fs * 0.3))

    # 6 batch SQI features
    feats = compute_sqi_features(ppg_raw, ppg_filt, list(peaks), fs)

    # Per-beat tSQI (default window_ms=600, template_n=5)
    tsqi_per_beat, _ = compute_tsqi_per_beat(ppg_filt, peaks, fs, 600, 5)
    tsqi_summary = aggregate_tsqi(tsqi_per_beat)

    out = {
        "file": path.name,
        "n_samples_full": n_samples_full,
        "duration_full_s": round(duration_full, 1),
        "fs": fs,
        "n_samples_analyzed": n_chunk,
        "duration_analyzed_s": round(n_chunk / fs, 1),
        "n_peaks": len(peaks),
    }
    out.update(feats)  # 6 batch SQI features
    # tsqi_summary keys: tsqi_median, tsqi_mean, tsqi_min, tsqi_max, tsqi_std,
    #                    bad_beat_ratio, n_beats
    out.update(tsqi_summary)
    return out


def print_distribution(label: str, values: pd.Series, n_files: int):
    """Print percentile summary of a metric."""
    if values.empty:
        print(f"  {label:30s} (no data)")
        return
    print(
        f"  {label:30s} N={n_files:3d}  "
        f"med={values.median():.3f}  mean={values.mean():.3f}  "
        f"min={values.min():.3f}  max={values.max():.3f}  std={values.std():.3f}"
    )


def cross_dataset_comparison(df_bidmc: pd.DataFrame):
    """Print cross-dataset comparison table."""
    if not DAY2_CSV.exists():
        print(f"\nWARNING: {DAY2_CSV} not found — skipping cross-dataset comparison")
        return

    df_self = pd.read_csv(DAY2_CSV)
    #  CSV has 'study_type' column: 'bp' (17 self-collected) or
    # 'sqa_dedicated' (4 S09 4-protocol)
    df_bp = df_self[df_self["study_type"] == "bp"]
    df_sqa = df_self[df_self["study_type"] == "sqa_dedicated"]

    print()
    print("=" * 90)
    print("CROSS-DATASET SQA COMPARISON")
    print("=" * 90)

    for feat, fmt_name in [
        ("spectral_purity", "Spectral purity"),
        ("hr_bpm", "HR (BPM)"),
        ("amplitude_ratio", "Amplitude ratio"),
        ("ssqi", "sSQI (skewness)"),
        ("ksqi", "kSQI (Fisher kurtosis)"),
        ("entropy_amp_dist", "Entropy amp dist"),
        ("tsqi_median", "tSQI median (per-beat)"),
        ("tsqi_mean", "tSQI mean"),
        ("bad_beat_ratio", "Bad beat ratio (r<0.86)"),
        ("n_beats", "N beats per recording"),
    ]:
        print(f"\n[{fmt_name}]")
        print_distribution("BP-protocol (self-collected)", df_bp[feat], len(df_bp))
        print_distribution("SQA-dedicated (S09)        ", df_sqa[feat], len(df_sqa))
        if feat in df_bidmc.columns:
            print_distribution("BIDMC ICU                   ", df_bidmc[feat], len(df_bidmc))

    print()
    print("=" * 90)
    print("HEADLINE COMPARISON TABLE (medians)")
    print("=" * 90)
    print(
        f"{'Dataset':30s} {'N':>4s}  {'tsqi_med':>8s}  "
        f"{'bad_beat':>8s}  {'spec_pur':>8s}  {'hr_bpm':>7s}"
    )
    print("-" * 90)
    for label, df in [
        ("BP-protocol (cleaned)", df_bp),
        ("SQA-dedicated (S09)", df_sqa),
        ("BIDMC ICU", df_bidmc),
    ]:
        if df.empty:
            continue
        print(
            f"{label:30s} {len(df):>4d}  "
            f"{df['tsqi_median'].median():>8.3f}  "
            f"{df['bad_beat_ratio'].median():>8.3f}  "
            f"{df['spectral_purity'].median():>8.3f}  "
            f"{df['hr_bpm'].median():>7.1f}"
        )


def main():
    if not BIDMC_DIR.exists():
        print(f"ERROR: {BIDMC_DIR} not found. Run download_bidmc.py first.")
        return 1

    csv_files = sorted(BIDMC_DIR.glob("bidmc_*_Signals.csv"))
    if not csv_files:
        print(f"ERROR: no BIDMC files in {BIDMC_DIR}. Run download_bidmc.py first.")
        return 1

    print(f"Analyzing {len(csv_files)} BIDMC recordings (first {CHUNK_SEC}s each)")
    print("-" * 70)

    results = []
    fail_count = 0
    t0 = time.time()
    for f in csv_files:
        try:
            row = analyze_bidmc(f)
            results.append(row)
            print(
                f"  {f.name:30s} peaks={row['n_peaks']:3d}  "
                f"tsqi_med={row['tsqi_median']:.3f}  "
                f"bad_beat={row['bad_beat_ratio']:.3f}  "
                f"spec_pur={row['spectral_purity']:.3f}"
            )
        except Exception as e:
            fail_count += 1
            print(f"  FAIL {f.name}: {type(e).__name__}: {e}")

    elapsed = time.time() - t0
    print("-" * 70)
    print(f"Analyzed: {len(results)} OK, {fail_count} FAIL ({elapsed:.1f}s)")

    if not results:
        print("ERROR: no results — aborting CSV write")
        return 1

    df_bidmc = pd.DataFrame(results)
    df_bidmc.to_csv(OUT_CSV, index=False)
    print(f"Saved: {OUT_CSV} ({len(df_bidmc)} rows)")

    cross_dataset_comparison(df_bidmc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
