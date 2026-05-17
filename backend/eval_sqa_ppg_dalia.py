"""Apply 7 SQI features pipeline ( batch +  per-beat) to PPG-DaLiA.

Source: Reiss et al. 2019, Sensors 19(14):3079. UCI dataset 495.
Empatica E4 wrist BVP at 64 Hz, ~150 min per subject, 8 activities.

Output: backend/eval_sqa_ppg_dalia_results.csv  (one row per chunk per subject)

Pipeline alignment:
- BVP native fs = 64 Hz (different from BP=100 Hz, BIDMC=125 Hz). NO resample.
- compute_sqi_features uses Welch with nperseg=min(256, len(signal)//4) — at fs=64,
  256 samples = 4s window, OK at 60s chunk granularity.
- Bandpass [0.5, 4.0] at fs=64.
- Each subject pickle = ~150 min × 64 Hz = ~580k samples. We chunk at CHUNK_SEC=60s
  (64*60 = 3840 samples per chunk) — same chunking convention as BIDMC headline.
"""
from __future__ import annotations

import gc
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from main import compute_sqi_features  # noqa: E402
from sqa_per_beat import compute_tsqi_per_beat, aggregate_tsqi  # noqa: E402


DALIA_DIR = Path("external_data/ppg_dalia/PPG_FieldStudy")
OUT_CSV = ROOT / "eval_sqa_ppg_dalia_results.csv"
FS_BVP = 64
FS_ACTIVITY = 4  # Hz (1 label per 0.25s per Reiss 2019)
CHUNK_SEC = 60

ACTIVITY_NAMES = {
    0: "transient",
    1: "sitting",
    2: "stairs",
    3: "table_soccer",
    4: "cycling",
    5: "driving",
    6: "lunch",
    7: "working",
    8: "walking",
}


def load_bvp_and_activity(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load pickle, extract BVP + activity, immediately free the rest.

    PPG-DaLiA pickles are 1.3-1.5 GB each (peak ~1.3 GB Python heap when loaded
    raw). To process 5 subjects sequentially within ~2.5 GB free RAM we must
    drop the chest ECG / RespiBAN / EDA / TEMP arrays as soon as possible.
    """
    with open(path, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    bvp = np.asarray(data["signal"]["wrist"]["BVP"], dtype=float).reshape(-1).copy()
    activity = np.asarray(data["activity"]).reshape(-1).copy()
    del data
    gc.collect()
    return bvp, activity


def chunk_activity_label(activity: np.ndarray, t_start_s: float, t_end_s: float) -> str:
    """Return dominant activity label name during [t_start_s, t_end_s) interval."""
    idx_start = int(t_start_s * FS_ACTIVITY)
    idx_end = int(t_end_s * FS_ACTIVITY)
    if idx_end <= idx_start or idx_start >= len(activity):
        return "unknown"
    seg = activity[idx_start:min(idx_end, len(activity))]
    if seg.size == 0:
        return "unknown"
    vals, counts = np.unique(seg, return_counts=True)
    dominant = int(vals[np.argmax(counts)])
    return ACTIVITY_NAMES.get(dominant, f"act_{dominant}")


def analyze_chunk(bvp_chunk: np.ndarray, fs: int) -> dict:
    """Run 6 batch SQI + per-beat tSQI on a 1D BVP chunk."""
    b, a = butter(4, [0.5, 4.0], btype="band", fs=fs)
    bvp_filt = filtfilt(b, a, bvp_chunk)

    peaks, _ = find_peaks(bvp_filt, distance=int(fs * 0.3))

    feats = compute_sqi_features(bvp_chunk, bvp_filt, list(peaks), fs)

    tsqi_per_beat, _ = compute_tsqi_per_beat(bvp_filt, peaks, fs, 600, 5)
    tsqi_summary = aggregate_tsqi(tsqi_per_beat)

    out = {"n_peaks": len(peaks)}
    out.update(feats)
    out.update(tsqi_summary)
    return out


def analyze_subject(subj_dir: Path, chunk_sec: int = CHUNK_SEC) -> list[dict]:
    """Process one subject's S{i}.pkl, return list of chunk-level rows."""
    subj_id = subj_dir.name  # e.g. "S1"
    pkl_path = subj_dir / f"{subj_id}.pkl"
    bvp, activity = load_bvp_and_activity(pkl_path)
    n_samples = len(bvp)
    duration_s = n_samples / FS_BVP

    n_chunk = int(chunk_sec * FS_BVP)
    rows: list[dict] = []
    n_chunks_total = n_samples // n_chunk
    for i in range(n_chunks_total):
        start = i * n_chunk
        end = start + n_chunk
        seg = bvp[start:end]
        t_start_s = start / FS_BVP
        t_end_s = end / FS_BVP
        try:
            metrics = analyze_chunk(seg, FS_BVP)
        except Exception as e:
            print(f"    chunk {i} fail: {type(e).__name__}: {e}")
            continue
        row = {
            "subject_id": subj_id,
            "chunk_idx": i,
            "t_start_s": round(t_start_s, 1),
            "t_end_s": round(t_end_s, 1),
            "activity": chunk_activity_label(activity, t_start_s, t_end_s),
            "fs": FS_BVP,
            "n_samples_chunk": n_chunk,
        }
        row.update(metrics)
        rows.append(row)
    print(
        f"  {subj_id}: {n_chunks_total} chunks ({duration_s/60:.1f} min total),"
        f" {len(rows)} OK"
    )
    return rows


def print_per_activity_summary(df: pd.DataFrame) -> None:
    if df.empty:
        return
    print()
    print("=" * 90)
    print("PPG-DaLiA per-activity summary (medians across all chunks of all subjects)")
    print("=" * 90)
    print(
        f"{'activity':<16}{'N':>5}  {'tsqi_med':>10}{'bad_beat':>10}"
        f"{'spec_pur':>10}{'hr_bpm':>9}{'amp_rat':>10}"
    )
    print("-" * 90)
    for act, grp in df.groupby("activity"):
        print(
            f"{act:<16}{len(grp):>5d}  "
            f"{grp['tsqi_median'].median():>10.3f}"
            f"{grp['bad_beat_ratio'].median():>10.3f}"
            f"{grp['spectral_purity'].median():>10.3f}"
            f"{grp['hr_bpm'].median():>9.1f}"
            f"{grp['amplitude_ratio'].median():>10.3f}"
        )


def print_per_subject_summary(df: pd.DataFrame) -> None:
    if df.empty:
        return
    print()
    print("=" * 90)
    print("PPG-DaLiA per-subject summary (medians)")
    print("=" * 90)
    print(
        f"{'subject':<10}{'N':>5}  {'tsqi_med':>10}{'bad_beat':>10}"
        f"{'spec_pur':>10}{'hr_bpm':>9}"
    )
    print("-" * 90)
    for subj, grp in df.groupby("subject_id"):
        print(
            f"{subj:<10}{len(grp):>5d}  "
            f"{grp['tsqi_median'].median():>10.3f}"
            f"{grp['bad_beat_ratio'].median():>10.3f}"
            f"{grp['spectral_purity'].median():>10.3f}"
            f"{grp['hr_bpm'].median():>9.1f}"
        )


def main() -> int:
    if not DALIA_DIR.exists():
        print(f"ERROR: {DALIA_DIR} not found. Run download_ppg_dalia.py first.")
        return 1

    subj_dirs = sorted(
        [p for p in DALIA_DIR.iterdir() if p.is_dir() and (p / f"{p.name}.pkl").exists()]
    )
    if not subj_dirs:
        print(f"ERROR: no subject pickle files in {DALIA_DIR}")
        return 1

    print(f"Analyzing {len(subj_dirs)} PPG-DaLiA subjects (chunk={CHUNK_SEC}s)")
    print("-" * 70)

    all_rows: list[dict] = []
    fail_count = 0
    t0 = time.time()
    for subj_dir in subj_dirs:
        try:
            rows = analyze_subject(subj_dir)
            all_rows.extend(rows)
        except Exception as e:
            fail_count += 1
            print(f"  FAIL {subj_dir.name}: {type(e).__name__}: {e}")
        gc.collect()

    elapsed = time.time() - t0
    print("-" * 70)
    print(
        f"Subjects analyzed: {len(subj_dirs) - fail_count} OK, {fail_count} FAIL"
        f" ({elapsed:.1f}s, {len(all_rows)} chunks)"
    )

    if not all_rows:
        return 1

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"Saved: {OUT_CSV} ({len(df)} rows)")

    print_per_subject_summary(df)
    print_per_activity_summary(df)
    return 0


if __name__ == "__main__":
    sys.exit(main())
