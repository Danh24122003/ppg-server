"""
Clean PPG CSV bị gap do ESP32 disconnect + resample tới fs_target nominal.

Strategy v2 (2026-05-03 - Phương án 3):
  1. Strip REAL DISCONNECTS (gap > 1s = WiFi/firmware reset)
  2. Concat continuous segments
  3. Detect fs_real từ median interval của INTRA-segment samples
  4. Resample tới fs_target=100Hz nếu drift > 5% (scipy.signal.resample_poly)
  5. Output uniform timestamps tại fs_target

Lý do P3:
  - Sensor MAX30102 design intent: 100Hz (firmware v4.1.0)
  - fs_real observed often = 111Hz do server timestamp truncation bug
    (log_ppg_local.py:207, plan fix v4.1.1)
  - Resample tới 100Hz nominal đảm bảo pipeline ML uniform +
    match PPG-BP training resample 1000->100

Use cases (1/5/2026):
  - Tram_01: 6 disconnects (92.5s dead) -> ~279s clean @ 100Hz
  - S10:   8 disconnects (100s dead) -> ~292s clean @ 100Hz
  - S02_01:  0 disconnects, fs drift 11% -> resample-only
"""
from __future__ import annotations
import sys
from fractions import Fraction
from pathlib import Path
import re
import numpy as np
import pandas as pd
from scipy.signal import resample_poly

DISCONNECT_THRESHOLD_MS = 1000  # >1s = real disconnect (WiFi / firmware reset)
FS_TARGET_HZ = 100              # nominal output rate (matches firmware design)
FS_DRIFT_THRESHOLD = 0.05       # resample if |fs_real - target| / target > 5%
MIN_SEGMENT_SAMPLES = 100       # drop segments < 1s @ 100Hz
BOUNDARY_TRIM_SAMPLES = 50      # drop 50 samples each side of inner segment boundaries
                                #   để tránh ringing artifact khi bandpass filter
                                #   gặp discontinuity tại điểm concat (~0.5s @ 100Hz)
CLEANED_SUBDIR = "cleaned"


def _resample_to_target(
    ir: np.ndarray, red: np.ndarray, fs_real: float, fs_target: float
):
    """Resample IR + Red signals fs_real -> fs_target via polyphase filter.

    Uses scipy.signal.resample_poly which applies anti-aliasing FIR internally.
    Fraction.limit_denominator(1000) finds rational up/down within tolerance.

    Returns (ir_resampled, red_resampled, up, down).
    """
    frac = Fraction(fs_target / fs_real).limit_denominator(1000)
    up, down = frac.numerator, frac.denominator
    ir_rs = resample_poly(ir.astype(np.float64), up, down)
    red_rs = resample_poly(red.astype(np.float64), up, down)
    # Sensor data is integer; round back
    return ir_rs.round().astype(np.int64), red_rs.round().astype(np.int64), up, down


def clean_csv(in_path: Path) -> Path:
    """Read CSV, strip disconnects, resample to fs_target, save cleaned version.

    Output saved to: <input_dir>/cleaned/<stem>_cleaned.csv
    Skip output (return original path) only if NO disconnects AND fs drift OK.
    """
    print(f"\n=== {in_path.name} ===")

    # Preserve metadata header
    metadata_lines = []
    with open(in_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('#'):
                metadata_lines.append(line)
            else:
                break

    df = pd.read_csv(in_path, comment='#')
    n_orig = len(df)
    ir = df['ir'].to_numpy(dtype=np.int64)
    red = df['red'].to_numpy(dtype=np.int64)
    ts = df['timestamp_ms'].to_numpy(dtype=np.float64)
    dur_orig = (ts[-1] - ts[0]) / 1000
    print(f"  Original: {n_orig} samples, duration {dur_orig:.1f}s")

    # Detect disconnects + segments
    intervals = np.diff(ts)
    disconnect_idx = np.where(intervals > DISCONNECT_THRESHOLD_MS)[0]
    n_disconnects = len(disconnect_idx)

    if n_disconnects == 0:
        valid_segs = [(0, n_orig)]
        total_disconnect_ms = 0.0
        print(f"  [OK]No disconnects (>{DISCONNECT_THRESHOLD_MS/1000:.0f}s gaps).")
    else:
        total_disconnect_ms = float(np.sum(intervals[disconnect_idx]))
        print(f"  [!]Disconnects (>{DISCONNECT_THRESHOLD_MS/1000:.0f}s): {n_disconnects}")
        print(
            f"     Total dead time: {total_disconnect_ms/1000:.1f}s "
            f"({total_disconnect_ms/(dur_orig*10):.1f}% of duration)"
        )
        for i, idx in enumerate(disconnect_idx):
            print(f"     #{i+1} at sample {idx}: {intervals[idx]/1000:.1f}s")

        segment_starts = [0] + [int(d) + 1 for d in disconnect_idx]
        segment_ends = [int(d) + 1 for d in disconnect_idx] + [n_orig]
        valid_segs = [
            (s, e) for s, e in zip(segment_starts, segment_ends)
            if e - s >= MIN_SEGMENT_SAMPLES
        ]
        n_dropped = len(segment_starts) - len(valid_segs)
        print(f"  Valid segments: {len(valid_segs)} (dropped {n_dropped} tiny)")
        if not valid_segs:
            print(f"  [X]No valid segments after filtering. Skip.")
            return in_path
        for i, (s, e) in enumerate(valid_segs):
            seg_dur = (ts[e-1] - ts[s]) / 1000
            print(f"     Seg #{i+1}: samples [{s}-{e-1}] = {e-s} samples, {seg_dur:.1f}s")

    # Detect fs_real từ INTRA-segment intervals (loại disconnect gaps)
    intra_parts = [np.diff(ts[s:e]) for s, e in valid_segs if e - s > 1]
    if not intra_parts:
        print(f"  [X]No valid intervals to estimate fs_real. Skip.")
        return in_path
    intra_intervals = np.concatenate(intra_parts)
    median_interval = float(np.median(intra_intervals))
    if median_interval <= 0:
        print(f"  [X]Invalid median interval ({median_interval}). Skip.")
        return in_path
    fs_real = 1000.0 / median_interval
    fs_drift = abs(fs_real - FS_TARGET_HZ) / FS_TARGET_HZ
    print(
        f"  fs_real (median intra-segment) = {fs_real:.2f} Hz "
        f"({median_interval:.3f}ms period) - drift {fs_drift*100:.2f}% vs target {FS_TARGET_HZ}Hz"
    )

    # Concat valid segments. Drop BOUNDARY_TRIM_SAMPLES o moi end cua segment
    # khi khong phai first start hoac last end -> giam ringing artifact tai
    # diem concat khi downstream chay bandpass (filtfilt). Skip trim neu chi
    # 1 segment (no inner boundaries) hoac segment qua ngan sau trim.
    if len(valid_segs) == 1:
        s, e = valid_segs[0]
        ir_concat = ir[s:e]
        red_concat = red[s:e]
    else:
        trimmed_parts_ir = []
        trimmed_parts_red = []
        for i, (s, e) in enumerate(valid_segs):
            s_trim = s if i == 0 else s + BOUNDARY_TRIM_SAMPLES
            e_trim = e if i == len(valid_segs) - 1 else e - BOUNDARY_TRIM_SAMPLES
            if e_trim - s_trim < MIN_SEGMENT_SAMPLES:
                # segment qua ngan sau trim - drop
                continue
            trimmed_parts_ir.append(ir[s_trim:e_trim])
            trimmed_parts_red.append(red[s_trim:e_trim])
        if not trimmed_parts_ir:
            print(f"  [X]All segments too short after boundary trim. Skip.")
            return in_path
        ir_concat = np.concatenate(trimmed_parts_ir)
        red_concat = np.concatenate(trimmed_parts_red)
        n_dropped_trim = (sum(e - s for s, e in valid_segs)
                          - sum(len(p) for p in trimmed_parts_ir))
        print(f"  Boundary trim: dropped {n_dropped_trim} samples "
              f"(±{BOUNDARY_TRIM_SAMPLES} per inner edge)")
    n_concat = len(ir_concat)

    # Resample if drift > threshold
    if fs_drift > FS_DRIFT_THRESHOLD:
        ir_clean, red_clean, up, down = _resample_to_target(
            ir_concat, red_concat, fs_real, FS_TARGET_HZ
        )
        n_clean = len(ir_clean)
        print(
            f"  Resample {fs_real:.1f}Hz -> {FS_TARGET_HZ}Hz "
            f"(factor {up}/{down}): {n_concat} -> {n_clean} samples"
        )
        resampled = True
    else:
        ir_clean, red_clean = ir_concat, red_concat
        n_clean = n_concat
        up, down = 1, 1
        print(f"  fs drift <= {FS_DRIFT_THRESHOLD*100:.0f}% threshold - no resample")
        resampled = False

    # Skip output if nothing changed (no disconnect AND no resample)
    if n_disconnects == 0 and not resampled:
        print(f"  ->No changes needed; not creating cleaned file.")
        return in_path

    # Generate uniform timestamps tại FS_TARGET_HZ
    ts_start = ts[valid_segs[0][0]]
    period_target = 1000.0 / FS_TARGET_HZ
    ts_out = ts_start + np.arange(n_clean) * period_target
    new_duration = (ts_out[-1] - ts_out[0]) / 1000.0
    print(
        f"  ->Output: {n_clean} samples @ {FS_TARGET_HZ}Hz uniform, "
        f"duration = {new_duration:.1f}s"
    )

    # Save cleaned CSV
    out_dir = in_path.parent / CLEANED_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (in_path.stem + '_cleaned.csv')
    with open(out_path, 'w', encoding='utf-8', newline='') as f:
        for line in metadata_lines:
            line_strip = line.rstrip('\n')
            if 'fs=' in line_strip:
                line_strip = re.sub(r'fs=\d+', f'fs={FS_TARGET_HZ}', line_strip)
                line_strip = re.sub(
                    r'duration_s=[\d.]+', f'duration_s={new_duration:.1f}', line_strip
                )
                line_strip += (
                    f',cleaned=true'
                    f',fs_real_observed={fs_real:.2f}'
                    f',resampled={"true" if resampled else "false"}'
                    f',resample_factor={up}/{down}'
                    f',disconnects_removed={n_disconnects}'
                    f',dead_time_s={total_disconnect_ms/1000:.1f}'
                    f',original_samples={n_orig}'
                )
            f.write(line_strip + '\n')

        f.write('timestamp_ms,ir,red\n')
        # Dung round() thay vi int() de tranh drift tich luy 540ms voi 30k samples
        # khi period_target = 10.0 nhung float arithmetic tich luy nho
        for t, i, r in zip(ts_out, ir_clean, red_clean):
            f.write(f"{int(round(t))},{i},{r}\n")

    print(f"  ->Saved: {out_path.name}")
    return out_path


def main():
    if len(sys.argv) < 2:
        print("Usage: python clean_gap_csv.py <input.csv> [<input2.csv> ...]")
        print("       python clean_gap_csv.py --all  # process all *.csv except *_cleaned")
        return 1

    files = []
    if sys.argv[1] == '--all':
        collected = Path(__file__).resolve().parents[2] / "collected_data"
        files = sorted(collected.glob("*.csv"))
        files = [f for f in files if not f.stem.endswith('_cleaned')]
    else:
        files = [Path(p) for p in sys.argv[1:]]

    for f in files:
        if not f.exists():
            print(f"NOT FOUND: {f}")
            continue
        clean_csv(f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
