"""
 Deep SQA — HRV ablation: SDNN/RMSSD MAE with vs without per-beat
tSQI filtering, stratified by SQA tier. EXTENDED + (8/5 evening) to
add Methods C / D / E for literature-correct + production + combined.

Question answered:
  Does per-beat tSQI filtering (Li & Clifford 2012, threshold 0.86 per
  Orphanidou 2015) improve HRV (SDNN, RMSSD) estimation accuracy compared
  to feeding ALL detected PPG peaks into HRV computation?

  - Method A (no filter): RR_A = diff(all PPG peaks) / fs * 1000 ms.
  - Method B (naive tSQI filter): keep only beats with tSQI >= 0.86, then
        RR_B = diff(surviving peaks) / fs * 1000 ms.  negative result
        baseline — gap-spanning RRs inflate SDNN/RMSSD on motion data.
  - Method C (literature-correct tSQI filter — HeartPy van Gent 2019
        source code analysis.py L293, vectorized):
        accepted = tsqi >= 0.86
        valid_mask = accepted[:-1] & accepted[1:]
        keep RR[i] iff peaks[i] AND peaks[i+1] are both accepted.
        Gap-spanning RRs CANNOT exist by construction.
        Then physiological gate [300, 2000] ms (Task Force 1996).
        SDNN: population std of kept RRs (ddof=0).
        RMSSD: successive diffs where both endpoints share a peak.
  - Method D (production HeartPy 2-stage only — no per-beat tSQI):
        Replicates backend/main.py:_reject_rr_outliers post K. fix
        2026-05-04 (mirrored in ml/main.py).
        Stage 1: RR in [300, 2000] ms (Task Force 1996 physiological)
        Stage 2: RR within ±20% of median of Stage-1 survivors
                 (van Gent 2019 HeartPy default RR_OUTLIER_TOLERANCE=0.20).
        SDNN: ddof=0 of survivors. RMSSD: np.diff on filtered series
        (production parity — non-strictly-adjacent).
  - Method E (combined — C then D Stage 2):
        Method C output (literature-correct tSQI excise) -> Stage 2 ±20%
        median quotient (van Gent 2019) on remaining RRs.
  - Ground truth: SDNN/RMSSD from ECG R-peaks.

  Hypothesis: Method C (HeartPy literature standard) outperforms Method B
  (naive) on motion-artifact data by avoiding gap-spanning RR inflation.
  Method D validates production approach. Method E tests stacked filtering.

Data sources (have ECG ground truth):
  1. BIDMC ICU       (external_data/bidmc/*.csv)
     ECG lead II at 125 Hz; re-derive R-peaks via scipy.find_peaks on
     bandpass-filtered ECG (matches  patterns).
  2. PPG-DaLiA wrist (external_data/ppg_dalia/PPG_FieldStudy/S{1..5}/*.pkl)
     Chest ECG R-peaks pre-computed by Reiss 2019 (`rpeaks` array,
     indices at 700 Hz).

Excluded (no ECG comparator): BP-protocol cleaned, S09-SQA dedicated.

Window pipeline (per row of eval_sqa_classifier_results.csv):
  - Reload raw signal (cache per file/subject).
  - Bandpass [0.5, 4] Hz Butterworth 4th-order, zero-phase filtfilt.
  - PPG peaks: scipy.find_peaks(distance=fs*0.3) (matches  LR feat).
  - Per-beat tSQI: compute_tsqi_per_beat(filtered, peaks, fs).
  - Method A:  RR_A_ms = np.diff(peaks) / fs * 1000.
  - Method B:  good_idx = filter_beats_by_tsqi(tsqi_list, threshold=0.86),
               filtered_peaks = peaks[good_idx],
               RR_B_ms = np.diff(filtered_peaks) / fs * 1000.

  Window dropped per-method if RR count < 3 (cannot compute meaningful
  SDNN). Tracked separately as `method_{a,b,c,d,e}_failed`.

  SDNN_X = np.std(RR_X, ddof=0)  (population std per Task Force 1996)
  RMSSD_A/B/D = np.sqrt(np.mean(np.diff(RR_X)**2))
  RMSSD_C/E = np.sqrt(np.mean(diff[adjacent_pairs_kept]**2))   adjacency-aware

  Ground-truth SDNN/RMSSD computed identically on ECG R-peaks within the
  window time-frame.

  err_sdnn_X = abs(SDNN_X - SDNN_gt) for X in {A, B, C, D, E}; same for RMSSD.

Stratify by:
  - Pseudo-label tier  (column `label`           from )
  - LR predicted tier  (column `label_pred_lr`   from )
  - Per dataset (BIDMC vs PPG-DaLiA)
  - Per method (A vs B vs C vs D vs E)

Outputs:
  backend/eval_hrv_ablation_results.csv
    Per-window: file, dataset, subject_id, window_idx, t_start_s, t_end_s,
    fs, n_peaks_total, n_peaks_kept_b, n_rr_kept_c, n_rr_kept_d,
    n_rr_kept_e, sdnn_{a,b,c,d,e,gt}, rmssd_{a,b,c,d,e,gt},
    err_sdnn_{a,b,c,d,e}, err_rmssd_{a,b,c,d,e}, method_{a,b,c,d,e}_failed,
    gt_failed, sqa_tier_pseudo, sqa_tier_lr_pred, skip_reason.

  backend/eval_hrv_ablation_summary.csv
    Per (label_source x dataset x tier x method x metric x comparison_mode):
      n_windows, mae_mean, mae_median, mae_p95, mae_ci95_lo, mae_ci95_hi.

  backend/figures/fig_hrv_ablation_sdnn.pdf           ( A vs B vs D)
  backend/figures/fig_hrv_ablation_rmssd.pdf          ( A vs B vs D)
  backend/figures/fig_hrv_ablation_improvement.pdf    ( pairwise)
  backend/figures/fig_hrv_ablation_methods_sdnn.pdf       (5-method)
  backend/figures/fig_hrv_ablation_methods_rmssd.pdf      (5-method)
  backend/figures/fig_hrv_ablation_methods_improvement.pdf (5-method)

Usage:
  python "backend/eval_hrv_ablation.py"             # all 5 methods
  python "backend/eval_hrv_ablation.py" --methods C,D,E  # extension only
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

# Direct import per task spec (sqa_per_beat is a standalone module).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sqa_per_beat import compute_tsqi_per_beat, filter_beats_by_tsqi  # noqa: E402

ROOT = Path(__file__).resolve().parent  # backend/
PROJECT_ROOT = ROOT.parent

CLASSIFIER_CSV = ROOT / "eval_sqa_classifier_results.csv"
BIDMC_DIR = PROJECT_ROOT / "external_data" / "bidmc"
DALIA_DIR = PROJECT_ROOT / "external_data" / "ppg_dalia" / "PPG_FieldStudy"

OUT_RESULTS_CSV = ROOT / "eval_hrv_ablation_results.csv"
OUT_SUMMARY_CSV = ROOT / "eval_hrv_ablation_summary.csv"
FIG_DIR = ROOT / "figures"

FS_BIDMC = 125
FS_BVP = 64
FS_DALIA_ECG = 700  # chest ECG (RespiBAN) per Reiss 2019

TIERS = ["excellent", "acceptable", "unfit"]
# Method canonical order. Letters lowercase a/b/c/d/e map to CSV column suffix.
# - A_no_filter:           Method A — no filter ( baseline)
# - B_tsqi_naive:          Method B — naive tSQI hard filter ( NEGATIVE)
# - C_tsqi_literature:     Method C — HeartPy van Gent 2019 vectorized mask
# - D_heartpy_2stage:      Method D — production HeartPy 2-stage RR rejection
#                          (mirrors main.py:_reject_rr_outliers post K. fix
#                          2026-05-04; ALIAS for prior file's "C_heartpy_rr")
# - E_combined:            Method E — Method C tSQI excise + Stage 2 quotient
METHODS = [
    "A_no_filter",
    "B_tsqi_naive",
    "C_tsqi_literature",
    "D_heartpy_2stage",
    "E_combined",
]
METHOD_LETTERS = {
    "A_no_filter": "a",
    "B_tsqi_naive": "b",
    "C_tsqi_literature": "c",
    "D_heartpy_2stage": "d",
    "E_combined": "e",
}
METRICS = ["sdnn", "rmssd"]
TIER_COLORS = {
    "excellent": "#2ecc71",
    "acceptable": "#f1c40f",
    "unfit": "#e74c3c",
}
METHOD_COLORS = {
    "A_no_filter": "#3498db",
    "B_tsqi_naive": "#9b59b6",
    "C_tsqi_literature": "#16a085",
    "D_heartpy_2stage": "#e67e22",
    "E_combined": "#34495e",
}
METHOD_HATCHES = {
    "A_no_filter": "",
    "B_tsqi_naive": "///",
    "C_tsqi_literature": "\\\\\\",
    "D_heartpy_2stage": "xxx",
    "E_combined": "++",
}
# Letter-only mapping for CLI (--methods A,B,C,D,E).
LETTER_TO_METHOD = {
    "A": "A_no_filter",
    "B": "B_tsqi_naive",
    "C": "C_tsqi_literature",
    "D": "D_heartpy_2stage",
    "E": "E_combined",
}

TSQI_THRESHOLD = 0.86       # Orphanidou 2015
MIN_RR_FOR_HRV = 3          # min RR intervals to compute SDNN/RMSSD
RR_PHYSIO_LO_MS = 300.0     # Task Force 1996 physiological lower bound (= 200 BPM)
RR_PHYSIO_HI_MS = 2000.0    # Task Force 1996 physiological upper bound (= 30 BPM)
RR_QUOTIENT_TOL = 0.20      # van Gent 2019 HeartPy ±20% median quotient

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
# HRV computation
# ============================================================
def sdnn_rmssd_from_rr_ms(rr_ms: np.ndarray) -> tuple[float, float]:
    """Return (SDNN, RMSSD) in ms from an array of RR intervals.

    SDNN: population std per Task Force 1996 (ddof=0).
    RMSSD: sqrt(mean(diff(RR)^2)).

    Returns (nan, nan) if len(rr_ms) < MIN_RR_FOR_HRV.
    RMSSD requires len(rr_ms) >= 2 (one diff). With min=3, both safe.
    """
    if len(rr_ms) < MIN_RR_FOR_HRV:
        return float("nan"), float("nan")
    sdnn = float(np.std(rr_ms, ddof=0))
    diff = np.diff(rr_ms)
    if len(diff) == 0:
        return sdnn, float("nan")
    rmssd = float(np.sqrt(np.mean(diff * diff)))
    return sdnn, rmssd


def reject_rr_2stage(rr_ms: np.ndarray) -> np.ndarray:
    """Method D: HeartPy 2-stage RR-level rejection (production reference).

    Mirrors backend/main.py:_reject_rr_outliers (post K. fix
    2026-05-04) and ml/main.py:_reject_rr_outliers.

    Stage 1: physiological gate [300, 2000] ms (Task Force 1996,
             Circulation 93:1043, PMID 8598068; = HR in [30, 200] BPM).
    Stage 2: median quotient [0.8, 1.2] (van Gent 2019 HeartPy default
             RR_OUTLIER_TOLERANCE=0.20).

    Returns boolean mask the same length as rr_ms (True = kept). Mask
    form (vs filtered array) is needed for adjacency-aware RMSSD.
    """
    n = len(rr_ms)
    if n == 0:
        return np.zeros(0, dtype=bool)
    keep_s1 = (rr_ms >= RR_PHYSIO_LO_MS) & (rr_ms <= RR_PHYSIO_HI_MS)
    if not np.any(keep_s1):
        return keep_s1
    med = float(np.median(rr_ms[keep_s1]))
    lo = med * (1.0 - RR_QUOTIENT_TOL)
    hi = med * (1.0 + RR_QUOTIENT_TOL)
    keep_s2 = (rr_ms >= lo) & (rr_ms <= hi)
    return keep_s1 & keep_s2


def sdnn_rmssd_method_d(rr_ms: np.ndarray) -> tuple[float, float, int, np.ndarray]:
    """Apply Method D (production HeartPy 2-stage) to RR intervals.

    Returns (sdnn_d, rmssd_d, n_kept, keep_mask).

    SDNN: np.std(kept_RRs, ddof=0) — adjacency irrelevant.
    RMSSD: only consecutive pairs (i, i+1) where BOTH keep_mask[i] AND
           keep_mask[i+1] are True contribute to successive_diff. This is
           STRICTER than production main.py:_compute_hrv_metrics, which
           applies np.diff to surviving RRs (potentially spanning gaps).
           Strict adjacency matches Task Force 1996 / van Gent 2019
           definition of successive-RR difference.

    NOTE: This script's RMSSD definition for Method D is stricter than
    production main.py (which np.diff's the filtered series including
    pairs that span gaps). The intent is to compare apples-to-apples
    with Methods C and E that use the same adjacency rule. Production
    parity for SDNN holds (ddof=0 of survivors).

    Returns (nan, nan, n_kept, mask) when fewer than MIN_RR_FOR_HRV RRs
    are kept overall, or when fewer than 1 valid adjacent pair exists
    for RMSSD.
    """
    keep = reject_rr_2stage(rr_ms)
    n_kept = int(keep.sum())
    if n_kept < MIN_RR_FOR_HRV:
        return float("nan"), float("nan"), n_kept, keep

    rr_kept = rr_ms[keep]
    sdnn = float(np.std(rr_kept, ddof=0))

    # Adjacency-aware RMSSD: only diff pairs (i, i+1) where both kept.
    pair_valid = keep[:-1] & keep[1:]
    if int(pair_valid.sum()) < 1:
        return sdnn, float("nan"), n_kept, keep
    diffs = np.diff(rr_ms)[pair_valid]
    if len(diffs) == 0:
        return sdnn, float("nan"), n_kept, keep
    rmssd = float(np.sqrt(np.mean(diffs * diffs)))
    return sdnn, rmssd, n_kept, keep


def sdnn_rmssd_method_c(
    peaks: np.ndarray,
    tsqi_list: list[float],
    fs: int,
    threshold: float = TSQI_THRESHOLD,
) -> tuple[float, float, int, int]:
    """Method C — Literature-correct tSQI filter (van Gent 2019 HeartPy).

    Source: heartpy/analysis.py L293, L295 (van Gent et al. 2019):
        rr_list = np.array([rr_source[i] for i in range(len(rr_source))
                            if b_peaklist[i] + b_peaklist[i+1] == 2])

    Vectorized equivalent: keep RR[i] iff peaks[i] AND peaks[i+1] are
    BOTH accepted (tSQI >= threshold). Gap-spanning RRs CANNOT exist by
    construction.

    Then applies a Task Force 1996 physiological gate [300, 2000] ms on
    the survivors to be safe against unphysiological residuals.

    RMSSD adjacency rule: an RR at index i covers peaks[i] -> peaks[i+1].
    Two valid RRs at indices i and j are "successive" iff j == i+1
    (sharing peak i+1). Successive diffs are computed on np.diff of the
    raw RR series, masked to those (i, i+1) pairs where both RRs survived
    AND were physiologically valid.

    Args:
        peaks:       1D np.ndarray of peak indices in the filtered signal.
                     Must be sorted ascending. Length N.
        tsqi_list:   list of length N (one tSQI per peak in `peaks`).
        fs:          sample rate Hz.
        threshold:   tSQI accept threshold (Orphanidou 2015).

    Returns:
        (sdnn_ms, rmssd_ms, n_rr_kept, n_dropped)
        n_dropped counts RRs dropped relative to N-1 raw RRs (= dropped
        by either the tSQI mask or the physiological gate).
    """
    if len(peaks) < 2 or len(tsqi_list) != len(peaks):
        return float("nan"), float("nan"), 0, 0

    accepted = np.asarray([t >= threshold for t in tsqi_list], dtype=bool)
    all_rr_ms = np.diff(peaks).astype(float) / float(fs) * 1000.0  # len = N-1
    n_raw = len(all_rr_ms)

    # HeartPy van Gent 2019 mask: keep RR[i] iff accepted[i] & accepted[i+1].
    valid_mask_tsqi = accepted[:-1] & accepted[1:]

    # Physiological gate (Task Force 1996) applied on top.
    valid_mask_phys = (all_rr_ms >= RR_PHYSIO_LO_MS) & (all_rr_ms <= RR_PHYSIO_HI_MS)
    valid_mask = valid_mask_tsqi & valid_mask_phys
    n_kept = int(valid_mask.sum())
    n_dropped = n_raw - n_kept

    if n_kept < MIN_RR_FOR_HRV:
        return float("nan"), float("nan"), n_kept, n_dropped

    rr_kept = all_rr_ms[valid_mask]
    sdnn = float(np.std(rr_kept, ddof=0))

    # Successive-pair adjacency: pair (i, i+1) both in valid_mask.
    pair_valid = valid_mask[:-1] & valid_mask[1:]
    if int(pair_valid.sum()) < 1:
        return sdnn, float("nan"), n_kept, n_dropped
    diffs = np.diff(all_rr_ms)[pair_valid]
    if len(diffs) == 0:
        return sdnn, float("nan"), n_kept, n_dropped
    rmssd = float(np.sqrt(np.mean(diffs * diffs)))
    return sdnn, rmssd, n_kept, n_dropped


def sdnn_rmssd_method_e(
    peaks: np.ndarray,
    tsqi_list: list[float],
    fs: int,
    threshold: float = TSQI_THRESHOLD,
) -> tuple[float, float, int, int]:
    """Method E — Combined: Method C tSQI excise + Stage 2 ±20% quotient.

    Apply Method C (literature-correct tSQI mask + physiological gate),
    then on the survivors apply Stage 2 median quotient filter
    (van Gent 2019 HeartPy default ±20%).

    RMSSD adjacency: only successive (i, i+1) pairs where BOTH RR(i) and
    RR(i+1) survived ALL three filters (tSQI mask + physio gate + Stage 2
    quotient).

    Returns (sdnn_ms, rmssd_ms, n_rr_kept, n_dropped). n_dropped is
    relative to N-1 raw RRs.
    """
    if len(peaks) < 2 or len(tsqi_list) != len(peaks):
        return float("nan"), float("nan"), 0, 0

    accepted = np.asarray([t >= threshold for t in tsqi_list], dtype=bool)
    all_rr_ms = np.diff(peaks).astype(float) / float(fs) * 1000.0
    n_raw = len(all_rr_ms)

    # Method C mask (tSQI + physio).
    mask_tsqi = accepted[:-1] & accepted[1:]
    mask_phys = (all_rr_ms >= RR_PHYSIO_LO_MS) & (all_rr_ms <= RR_PHYSIO_HI_MS)
    mask_c = mask_tsqi & mask_phys
    if int(mask_c.sum()) < MIN_RR_FOR_HRV:
        return float("nan"), float("nan"), int(mask_c.sum()), n_raw - int(mask_c.sum())

    # Stage 2 ±20% median quotient on Method-C survivors.
    rr_after_c = all_rr_ms[mask_c]
    med = float(np.median(rr_after_c))
    if med <= 0:
        return float("nan"), float("nan"), 0, n_raw
    lo = med * (1.0 - RR_QUOTIENT_TOL)
    hi = med * (1.0 + RR_QUOTIENT_TOL)
    # Build a global mask aligned with all_rr_ms by combining mask_c with
    # the per-element quotient test (test runs on entire all_rr_ms, then
    # AND'd with mask_c so non-C survivors stay False).
    mask_quotient = (all_rr_ms >= lo) & (all_rr_ms <= hi)
    valid_mask = mask_c & mask_quotient
    n_kept = int(valid_mask.sum())
    n_dropped = n_raw - n_kept

    if n_kept < MIN_RR_FOR_HRV:
        return float("nan"), float("nan"), n_kept, n_dropped

    rr_kept = all_rr_ms[valid_mask]
    sdnn = float(np.std(rr_kept, ddof=0))

    pair_valid = valid_mask[:-1] & valid_mask[1:]
    if int(pair_valid.sum()) < 1:
        return sdnn, float("nan"), n_kept, n_dropped
    diffs = np.diff(all_rr_ms)[pair_valid]
    if len(diffs) == 0:
        return sdnn, float("nan"), n_kept, n_dropped
    rmssd = float(np.sqrt(np.mean(diffs * diffs)))
    return sdnn, rmssd, n_kept, n_dropped


def ppg_peaks_and_tsqi(
    signal: np.ndarray, fs: int
) -> tuple[np.ndarray, list[float]]:
    """Bandpass [0.5, 4] Hz, find peaks (distance=fs*0.3), per-beat tSQI.

    Returns (peaks, tsqi_list). tsqi_list may be shorter than peaks if
    edge beats were dropped by compute_tsqi_per_beat (peak too close to
    signal edge for ±300ms window). In that case len(tsqi_list) ==
    n_valid_beats. Caller must align peaks accordingly.

    To keep alignment simple, we re-extract the same valid-beat subset
    here and return only those peaks.
    """
    if len(signal) < int(fs * 2):
        return np.asarray([], dtype=int), []
    b, a = butter(4, [0.5, 4.0], btype="band", fs=fs)
    filt = filtfilt(b, a, signal)
    peaks, _ = find_peaks(filt, distance=int(fs * 0.3))
    if len(peaks) == 0:
        return peaks.astype(int), []

    # Mirror compute_tsqi_per_beat's edge-trim logic to keep peaks aligned
    # with the returned tsqi list. compute_tsqi_per_beat uses
    # half_w = int(window_ms / 1000.0 * fs / 2) and accepts peaks where
    # p - half_w >= 0 and p + half_w < n.
    window_ms = 600
    half_w = int(window_ms / 1000.0 * fs / 2)
    n = len(filt)
    valid_mask = (peaks - half_w >= 0) & (peaks + half_w < n)
    valid_peaks = peaks[valid_mask].astype(int)

    tsqi_list, _tpl = compute_tsqi_per_beat(filt, valid_peaks, fs)
    # tsqi_list has same length as valid_peaks IF >= template_n+1=6 valid
    # beats; otherwise compute_tsqi_per_beat returns ([], None).
    if len(tsqi_list) != len(valid_peaks):
        return valid_peaks, []
    return valid_peaks, tsqi_list


def hrv_from_rpeaks_dalia(
    rpeaks: np.ndarray, t_start_s: float, t_end_s: float
) -> tuple[float, float, int]:
    """Return (SDNN, RMSSD, n_rr) from PPG-DaLiA pre-computed R-peaks
    within the window time-frame. R-peaks indexed at FS_DALIA_ECG=700 Hz.
    """
    t_start_idx = t_start_s * FS_DALIA_ECG
    t_end_idx = t_end_s * FS_DALIA_ECG
    in_win = rpeaks[(rpeaks >= t_start_idx) & (rpeaks < t_end_idx)]
    if len(in_win) < MIN_RR_FOR_HRV + 1:
        return float("nan"), float("nan"), 0
    rr_ms = np.diff(in_win) / float(FS_DALIA_ECG) * 1000.0
    sdnn, rmssd = sdnn_rmssd_from_rr_ms(rr_ms)
    return sdnn, rmssd, len(rr_ms)


def hrv_from_ecg_bidmc(
    ecg_window: np.ndarray, fs: int
) -> tuple[float, float, int, Optional[str]]:
    """Re-derive R-peaks on BIDMC ECG window, return (SDNN, RMSSD, n_rr,
    reason). reason is None on success, else 'ecg_*' string (mirrors 
    patterns).
    """
    if len(ecg_window) < int(fs * 2):
        return float("nan"), float("nan"), 0, "ecg_window_too_short"
    b, a = butter(4, [0.5, 40.0], btype="band", fs=fs)
    ecg_f = filtfilt(b, a, ecg_window)
    # Same prominence-based detection as  (P1 robust threshold).
    peaks, _ = find_peaks(
        ecg_f,
        distance=int(fs * 0.4),
        prominence=float(np.std(ecg_f)) * 0.5,
    )
    if len(peaks) < MIN_RR_FOR_HRV + 1:
        return float("nan"), float("nan"), 0, "ecg_too_few_peaks"
    rr_ms = np.diff(peaks) / float(fs) * 1000.0
    # Plausibility guard: RR median => HR in [30, 220] BPM.
    rr_med = float(np.median(rr_ms))
    if rr_med <= 0:
        return float("nan"), float("nan"), 0, "ecg_too_few_peaks"
    hr_med = 60_000.0 / rr_med
    if hr_med < 30.0 or hr_med > 220.0:
        return float("nan"), float("nan"), 0, "ecg_implausible_hr"
    sdnn, rmssd = sdnn_rmssd_from_rr_ms(rr_ms)
    return sdnn, rmssd, len(rr_ms), None


# ============================================================
# Per-window evaluation
# ============================================================
def _eval_methods_from_peaks_tsqi(
    peaks: np.ndarray,
    tsqi_list: list[float],
    fs: int,
    enabled_methods: set[str] | None = None,
) -> dict:
    """Compute Method A/B/C/D/E SDNN/RMSSD given peaks + tSQI.

    Args:
        peaks:           detected PPG peak indices.
        tsqi_list:       per-beat tSQI aligned with peaks.
        fs:              sample rate Hz.
        enabled_methods: set of method names to compute (e.g. {"A_no_filter",
                         "C_tsqi_literature"}). Disabled methods leave NaN
                         outputs and method_X_failed=True. None => all 5.

    Returns dict with sdnn_{a,b,c,d,e}, rmssd_{a,b,c,d,e}, n_peaks_total,
    n_peaks_kept_b, n_rr_kept_c, n_rr_kept_d, n_rr_kept_e,
    method_{a,b,c,d,e}_failed.
    """
    if enabled_methods is None:
        enabled_methods = set(METHODS)

    out = {
        "sdnn_a": float("nan"),
        "rmssd_a": float("nan"),
        "sdnn_b": float("nan"),
        "rmssd_b": float("nan"),
        "sdnn_c": float("nan"),
        "rmssd_c": float("nan"),
        "sdnn_d": float("nan"),
        "rmssd_d": float("nan"),
        "sdnn_e": float("nan"),
        "rmssd_e": float("nan"),
        "n_peaks_total": int(len(peaks)),
        "n_peaks_kept_b": 0,
        "n_rr_kept_c": 0,
        "n_rr_kept_d": 0,
        "n_rr_kept_e": 0,
        "n_dropped_c": 0,
        "n_dropped_d": 0,
        "n_dropped_e": 0,
        "method_a_failed": True,
        "method_b_failed": True,
        "method_c_failed": True,
        "method_d_failed": True,
        "method_e_failed": True,
    }

    # Method A: all peaks. Compute rr_a_ms anyway (used by D regardless of
    # whether A is reported, since D operates on Method-A's RR sequence).
    rr_a_ms = np.zeros(0, dtype=float)
    if len(peaks) >= MIN_RR_FOR_HRV + 1:
        rr_a_ms = np.diff(peaks).astype(float) / float(fs) * 1000.0
        if "A_no_filter" in enabled_methods:
            sdnn_a, rmssd_a = sdnn_rmssd_from_rr_ms(rr_a_ms)
            out["sdnn_a"] = sdnn_a
            out["rmssd_a"] = rmssd_a
            out["method_a_failed"] = bool(np.isnan(sdnn_a))

    # Method B: naive tSQI hard filter on peaks.
    if (
        "B_tsqi_naive" in enabled_methods
        and len(tsqi_list) == len(peaks)
        and len(tsqi_list) > 0
    ):
        good_idx = filter_beats_by_tsqi(tsqi_list, threshold=TSQI_THRESHOLD)
        kept_peaks = peaks[good_idx] if len(good_idx) > 0 else np.asarray([], dtype=int)
        out["n_peaks_kept_b"] = int(len(kept_peaks))
        if len(kept_peaks) >= MIN_RR_FOR_HRV + 1:
            # See  honest framing note 6: gap-spanning RRs retained
            # deliberately to expose naive filter consequence.
            rr_b_ms = np.diff(kept_peaks).astype(float) / float(fs) * 1000.0
            sdnn_b, rmssd_b = sdnn_rmssd_from_rr_ms(rr_b_ms)
            out["sdnn_b"] = sdnn_b
            out["rmssd_b"] = rmssd_b
            out["method_b_failed"] = bool(np.isnan(sdnn_b))

    # Method C: literature-correct tSQI filter (HeartPy van Gent 2019
    # vectorized mask: accepted[:-1] & accepted[1:]).
    if (
        "C_tsqi_literature" in enabled_methods
        and len(tsqi_list) == len(peaks)
        and len(tsqi_list) > 0
    ):
        sdnn_c, rmssd_c, n_kept_c, n_drop_c = sdnn_rmssd_method_c(
            peaks, tsqi_list, fs
        )
        out["sdnn_c"] = sdnn_c
        out["rmssd_c"] = rmssd_c
        out["n_rr_kept_c"] = int(n_kept_c)
        out["n_dropped_c"] = int(n_drop_c)
        out["method_c_failed"] = bool(np.isnan(sdnn_c))

    # Method D: HeartPy 2-stage RR-level rejection (production reference,
    # mirrors main.py:_reject_rr_outliers post K. fix 2026-05-04).
    if "D_heartpy_2stage" in enabled_methods and len(rr_a_ms) >= 1:
        sdnn_d, rmssd_d, n_kept_d, _keep_mask = sdnn_rmssd_method_d(rr_a_ms)
        out["sdnn_d"] = sdnn_d
        out["rmssd_d"] = rmssd_d
        out["n_rr_kept_d"] = int(n_kept_d)
        out["n_dropped_d"] = int(len(rr_a_ms) - n_kept_d)
        out["method_d_failed"] = bool(np.isnan(sdnn_d))

    # Method E: combined — Method C tSQI mask then Stage 2 quotient.
    if (
        "E_combined" in enabled_methods
        and len(tsqi_list) == len(peaks)
        and len(tsqi_list) > 0
    ):
        sdnn_e, rmssd_e, n_kept_e, n_drop_e = sdnn_rmssd_method_e(
            peaks, tsqi_list, fs
        )
        out["sdnn_e"] = sdnn_e
        out["rmssd_e"] = rmssd_e
        out["n_rr_kept_e"] = int(n_kept_e)
        out["n_dropped_e"] = int(n_drop_e)
        out["method_e_failed"] = bool(np.isnan(sdnn_e))

    return out


def _empty_eval_out() -> dict:
    """Default per-row output dict with all 5 methods nan-initialised."""
    return {
        "sdnn_a": float("nan"),
        "rmssd_a": float("nan"),
        "sdnn_b": float("nan"),
        "rmssd_b": float("nan"),
        "sdnn_c": float("nan"),
        "rmssd_c": float("nan"),
        "sdnn_d": float("nan"),
        "rmssd_d": float("nan"),
        "sdnn_e": float("nan"),
        "rmssd_e": float("nan"),
        "sdnn_gt": float("nan"),
        "rmssd_gt": float("nan"),
        "n_peaks_total": 0,
        "n_peaks_kept_b": 0,
        "n_rr_kept_c": 0,
        "n_rr_kept_d": 0,
        "n_rr_kept_e": 0,
        "n_dropped_c": 0,
        "n_dropped_d": 0,
        "n_dropped_e": 0,
        "method_a_failed": True,
        "method_b_failed": True,
        "method_c_failed": True,
        "method_d_failed": True,
        "method_e_failed": True,
        "gt_failed": True,
        "skip_reason": "",
    }


def _set_skip_reason_methods(
    out: dict, gt_reason: Optional[str], enabled_methods: set[str]
) -> None:
    """Categorise per-row skip_reason after methods + ground truth ran.

    Priority: gt failure > all enabled methods failed > partial failures.
    Only considers methods listed in `enabled_methods` so that disabled
    methods (NaN by design) don't trip the 'all_failed' branch.
    """
    if out["gt_failed"]:
        out["skip_reason"] = gt_reason or "ecg_too_few_peaks"
        return
    enabled_letters = {METHOD_LETTERS[m] for m in enabled_methods}
    ok_map = {
        letter: not out[f"method_{letter}_failed"] for letter in enabled_letters
    }
    any_ok = any(ok_map.values())
    if not any_ok:
        out["skip_reason"] = "ppg_all_methods_failed"
        return
    # If at least one enabled method succeeded, mark per-letter "only failed"
    # for the first failing letter (priority order a, b, c, d, e).
    for letter in ["a", "b", "c", "d", "e"]:
        if letter not in enabled_letters:
            continue
        if not ok_map[letter]:
            out["skip_reason"] = f"method_{letter}_only_failed"
            return


def eval_bidmc_row(row: pd.Series, enabled_methods: set[str]) -> dict:
    cached = load_bidmc_cached(row["file"])
    out = _empty_eval_out()
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

    peaks, tsqi_list = ppg_peaks_and_tsqi(ppg_win, fs)
    method_info = _eval_methods_from_peaks_tsqi(peaks, tsqi_list, fs, enabled_methods)
    out.update(method_info)

    sdnn_gt, rmssd_gt, _n_rr_gt, ecg_reason = hrv_from_ecg_bidmc(ecg_win, fs)
    out["sdnn_gt"] = sdnn_gt
    out["rmssd_gt"] = rmssd_gt
    out["gt_failed"] = bool(np.isnan(sdnn_gt))

    _set_skip_reason_methods(out, ecg_reason, enabled_methods)
    return out


def eval_dalia_row(row: pd.Series, enabled_methods: set[str]) -> dict:
    subj = row["subject_id"]
    cached = load_dalia_cached(subj)
    out = _empty_eval_out()
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

    peaks, tsqi_list = ppg_peaks_and_tsqi(bvp_win, fs)
    method_info = _eval_methods_from_peaks_tsqi(peaks, tsqi_list, fs, enabled_methods)
    out.update(method_info)

    sdnn_gt, rmssd_gt, _n_rr_gt = hrv_from_rpeaks_dalia(
        rpeaks, float(row["t_start_s"]), float(row["t_end_s"])
    )
    out["sdnn_gt"] = sdnn_gt
    out["rmssd_gt"] = rmssd_gt
    out["gt_failed"] = bool(np.isnan(sdnn_gt))

    _set_skip_reason_methods(out, "ecg_rpeaks_too_few_in_window", enabled_methods)
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


def summarize_by_tier_method(
    df: pd.DataFrame,
    tier_col: str,
    label_source: str,
    enabled_methods: set[str],
) -> pd.DataFrame:
    """Build summary: rows = (label_source, dataset, tier, method, metric,
    comparison_mode).

    Two comparison modes per (tier, method, metric):
      - "per_method_success": MAE on each method's own success set. Each
        method's success set may differ (each may fail on different hardest
        windows). Useful to show practical Method-X output range but unfair
        direct comparison across methods.
      - "matched": MAE on the all-ok intersection (gt_ok AND every enabled
        method ok). PRIMARY honest comparison: all enabled methods see the
        same windows.
    """
    rows: list[dict] = []
    df_gt_ok = df[~df["gt_failed"]].copy()
    enabled_letters = [METHOD_LETTERS[m] for m in METHODS if m in enabled_methods]

    def _stats_block(errs: np.ndarray, mode: str, ds: str, tier: str,
                     method: str, metric: str) -> dict:
        if len(errs) == 0:
            return {
                "label_source": label_source,
                "dataset": ds,
                "tier": tier,
                "method": method,
                "metric": metric,
                "comparison_mode": mode,
                "n_windows": 0,
                "mae_mean": float("nan"),
                "mae_median": float("nan"),
                "mae_p95": float("nan"),
                "mae_ci95_lo": float("nan"),
                "mae_ci95_hi": float("nan"),
            }
        ci_lo, ci_hi = bootstrap_mae_ci(errs)
        return {
            "label_source": label_source,
            "dataset": ds,
            "tier": tier,
            "method": method,
            "metric": metric,
            "comparison_mode": mode,
            "n_windows": int(len(errs)),
            "mae_mean": float(np.mean(errs)),
            "mae_median": float(np.median(errs)),
            "mae_p95": float(np.percentile(errs, 95)),
            "mae_ci95_lo": ci_lo,
            "mae_ci95_hi": ci_hi,
        }

    for ds, ds_grp in df_gt_ok.groupby("dataset"):
        for tier in TIERS:
            tier_grp = ds_grp[ds_grp[tier_col] == tier]

            # ---- per_method_success rows (kept for transparency) ----
            for method in METHODS:
                if method not in enabled_methods:
                    continue
                letter = METHOD_LETTERS[method]
                fail_col = f"method_{letter}_failed"
                method_ok = tier_grp[~tier_grp[fail_col]]
                for metric in METRICS:
                    err_col = f"err_{metric}_{letter}"
                    errs = method_ok[err_col].dropna().to_numpy(dtype=float)
                    rows.append(
                        _stats_block(errs, "per_method_success", ds, tier, method, metric)
                    )

            # ---- matched (all_ok) rows: PRIMARY honest comparison ----
            # all_ok: every enabled method must have succeeded.
            if len(enabled_letters) == 0:
                continue
            all_ok_mask = pd.Series([True] * len(tier_grp), index=tier_grp.index)
            for letter in enabled_letters:
                all_ok_mask &= ~tier_grp[f"method_{letter}_failed"]
            all_ok_grp = tier_grp[all_ok_mask]
            for method in METHODS:
                if method not in enabled_methods:
                    continue
                letter = METHOD_LETTERS[method]
                for metric in METRICS:
                    err_col = f"err_{metric}_{letter}"
                    errs = all_ok_grp[err_col].dropna().to_numpy(dtype=float)
                    rows.append(
                        _stats_block(errs, "matched", ds, tier, method, metric)
                    )
    return pd.DataFrame(rows)


def compute_window_failure_stats(
    df: pd.DataFrame, enabled_methods: set[str]
) -> pd.DataFrame:
    """Per (dataset, tier_col, tier): count windows where each enabled
    method succeeded vs failed, plus per-method kill rates relative to
    Method A (when A is enabled). all_ok = every enabled method
    succeeded on the same window.
    """
    rows: list[dict] = []
    df_gt_ok = df[~df["gt_failed"]].copy()
    enabled_letters = [METHOD_LETTERS[m] for m in METHODS if m in enabled_methods]
    a_enabled = "A_no_filter" in enabled_methods

    for tier_col, label_source in [
        ("sqa_tier_pseudo", "pseudo"),
        ("sqa_tier_lr_pred", "lr_pred"),
    ]:
        for ds, ds_grp in df_gt_ok.groupby("dataset"):
            for tier in TIERS:
                tier_grp = ds_grp[ds_grp[tier_col] == tier]
                rec: dict = {
                    "label_source": label_source,
                    "dataset": ds,
                    "tier": tier,
                    "n_total": int(len(tier_grp)),
                }
                if len(tier_grp) == 0:
                    for letter in enabled_letters:
                        rec[f"n_{letter}_ok"] = 0
                        if a_enabled and letter != "a":
                            rec[f"n_a_ok_{letter}_failed"] = 0
                            rec[f"pct_{letter}_filter_kill"] = float("nan")
                    rec["n_all_ok"] = 0
                    rows.append(rec)
                    continue

                ok_series: dict[str, pd.Series] = {
                    letter: ~tier_grp[f"method_{letter}_failed"]
                    for letter in enabled_letters
                }
                for letter, ok in ok_series.items():
                    rec[f"n_{letter}_ok"] = int(ok.sum())

                # all_ok across enabled methods.
                if enabled_letters:
                    all_ok_mask = pd.Series(
                        [True] * len(tier_grp), index=tier_grp.index
                    )
                    for ok in ok_series.values():
                        all_ok_mask &= ok
                    rec["n_all_ok"] = int(all_ok_mask.sum())
                else:
                    rec["n_all_ok"] = 0

                # Filter-kill rates: A_ok AND letter_failed (vs A_ok total).
                if a_enabled:
                    a_ok = ok_series["a"]
                    n_a_ok = int(a_ok.sum())
                    for letter in enabled_letters:
                        if letter == "a":
                            continue
                        ok = ok_series[letter]
                        n_a_ok_letter_failed = int((a_ok & ~ok).sum())
                        rec[f"n_a_ok_{letter}_failed"] = n_a_ok_letter_failed
                        rec[f"pct_{letter}_filter_kill"] = (
                            float(n_a_ok_letter_failed) / float(n_a_ok) * 100.0
                            if n_a_ok > 0
                            else float("nan")
                        )
                rows.append(rec)
    return pd.DataFrame(rows)


def compute_improvement_table(
    summary: pd.DataFrame,
    enabled_methods: set[str],
    comparison_mode: str = "per_method_success",
) -> pd.DataFrame:
    """Pivot summary so each row has every enabled method's MAE side-by-side,
    plus % improvement of each non-A method vs Method A baseline:
      improvement_X_vs_a = (mae_A - mae_X) / mae_A * 100   (X over A)

    comparison_mode:
      - "per_method_success" (older interface): per-method success sets (different
        N per method; unfair across methods).
      - "matched": all-ok intersection (same N across enabled methods;
        primary honest).
    """
    rows: list[dict] = []
    src = summary[summary["comparison_mode"] == comparison_mode]
    grouped = src.groupby(["label_source", "dataset", "tier", "metric"])
    enabled_in_order = [m for m in METHODS if m in enabled_methods]
    a_enabled = "A_no_filter" in enabled_methods

    def _pct(base: float, target: float, base_n: int, tgt_n: int) -> float:
        if base_n == 0 or tgt_n == 0:
            return float("nan")
        if not (base > 0):
            return float("nan")
        return (base - target) / base * 100.0

    for key, grp in grouped:
        per_method = {m: grp[grp["method"] == m] for m in enabled_in_order}
        # Skip group if any enabled method has no row in summary.
        if any(df.empty for df in per_method.values()):
            continue
        rec: dict = {
            "label_source": key[0],
            "dataset": key[1],
            "tier": key[2],
            "metric": key[3],
            "comparison_mode": comparison_mode,
        }
        for m in enabled_in_order:
            letter = METHOD_LETTERS[m]
            r = per_method[m].iloc[0]
            rec[f"n_{letter}"] = int(r["n_windows"])
            rec[f"mae_{letter}"] = float(r["mae_mean"])
        if a_enabled:
            mae_a = rec["mae_a"]
            n_a = rec["n_a"]
            for m in enabled_in_order:
                if m == "A_no_filter":
                    continue
                letter = METHOD_LETTERS[m]
                rec[f"improvement_{letter}_vs_a"] = _pct(
                    mae_a, rec[f"mae_{letter}"], n_a, rec[f"n_{letter}"]
                )
        rows.append(rec)
    return pd.DataFrame(rows)


# ============================================================
# Console reporting
# ============================================================
def print_summary_table(
    summary: pd.DataFrame, comparison_mode: str = "per_method_success"
) -> None:
    src = summary[summary["comparison_mode"] == comparison_mode]
    print()
    print("=" * 116)
    print(
        f"HRV ABLATION — MAE per (label_source x dataset x tier x method x metric)"
        f"  [comparison_mode={comparison_mode}]"
    )
    print("=" * 116)
    print(
        f"{'label_src':<10}{'dataset':<12}{'tier':<12}{'metric':<8}"
        f"{'method':<16}{'N':>6}  {'mae_mean':>9}{'mae_med':>9}"
        f"{'p95':>9}{'CI95':>22}"
    )
    print("-" * 116)
    for label_source in src["label_source"].unique():
        for ds in sorted(src[src.label_source == label_source]["dataset"].unique()):
            for tier in TIERS:
                for metric in METRICS:
                    for method in METHODS:
                        sub = src[
                            (src.label_source == label_source)
                            & (src.dataset == ds)
                            & (src.tier == tier)
                            & (src.method == method)
                            & (src.metric == metric)
                        ]
                        if sub.empty:
                            continue
                        r = sub.iloc[0]
                        if r["n_windows"] == 0:
                            print(
                                f"{label_source:<10}{ds:<12}{tier:<12}"
                                f"{metric:<8}{method:<16}{r['n_windows']:>6}  "
                                f"{'-':>9}{'-':>9}{'-':>9}{'-':>22}"
                            )
                            continue
                        ci = f"[{r['mae_ci95_lo']:.2f}, {r['mae_ci95_hi']:.2f}]"
                        print(
                            f"{label_source:<10}{ds:<12}{tier:<12}"
                            f"{metric:<8}{method:<16}{r['n_windows']:>6}  "
                            f"{r['mae_mean']:>9.2f}{r['mae_median']:>9.2f}"
                            f"{r['mae_p95']:>9.2f}{ci:>22}"
                        )
            print("-" * 116)


def print_improvement_table(imp: pd.DataFrame, enabled_methods: set[str]) -> None:
    mode = imp["comparison_mode"].iloc[0] if not imp.empty else "?"
    enabled_in_order = [m for m in METHODS if m in enabled_methods]
    a_enabled = "A_no_filter" in enabled_methods
    non_a = [m for m in enabled_in_order if m != "A_no_filter"]

    print()
    print("=" * 140)
    print(
        f"METHOD vs METHOD-A baseline — % improvement (positive = "
        f"BETTER than baseline)  [comparison_mode={mode}]"
    )
    if a_enabled:
        for m in non_a:
            letter = METHOD_LETTERS[m].upper()
            print(f"  improv_{letter}_vs_A = (MAE_A - MAE_{letter}) / MAE_A * 100")
    else:
        print("  (Method A disabled — no improvement column available.)")
    print("=" * 140)

    # Header.
    header = f"{'label_src':<10}{'dataset':<12}{'tier':<12}{'metric':<7}"
    for m in enabled_in_order:
        L = METHOD_LETTERS[m].upper()
        header += f"{'N_'+L:>5}"
    header += "  "
    for m in enabled_in_order:
        L = METHOD_LETTERS[m].upper()
        header += f"{'MAE_'+L:>9}"
    if a_enabled:
        header += "  "
        for m in non_a:
            L = METHOD_LETTERS[m].upper()
            header += f"{L+'_vs_A%':>10}"
    print(header)
    print("-" * 140)

    def _fmt_pct(v: float) -> str:
        return f"{v:>9.1f}%" if not np.isnan(v) else f"{'-':>10}"

    def _fmt_mae(v: float) -> str:
        return f"{v:>9.2f}" if not np.isnan(v) else f"{'-':>9}"

    for label_source in ["pseudo", "lr_pred"]:
        for ds in sorted(imp[imp.label_source == label_source]["dataset"].unique()):
            for tier in TIERS:
                for metric in METRICS:
                    sub = imp[
                        (imp.label_source == label_source)
                        & (imp.dataset == ds)
                        & (imp.tier == tier)
                        & (imp.metric == metric)
                    ]
                    if sub.empty:
                        continue
                    r = sub.iloc[0]
                    line = f"{label_source:<10}{ds:<12}{tier:<12}{metric:<7}"
                    for m in enabled_in_order:
                        letter = METHOD_LETTERS[m]
                        line += f"{int(r[f'n_{letter}']):>5}"
                    line += "  "
                    for m in enabled_in_order:
                        letter = METHOD_LETTERS[m]
                        line += _fmt_mae(r[f"mae_{letter}"])
                    if a_enabled:
                        line += "  "
                        for m in non_a:
                            letter = METHOD_LETTERS[m]
                            line += _fmt_pct(r[f"improvement_{letter}_vs_a"])
                    print(line)
            print("-" * 140)


def print_window_failures(fail_df: pd.DataFrame, enabled_methods: set[str]) -> None:
    enabled_letters = [METHOD_LETTERS[m] for m in METHODS if m in enabled_methods]
    a_enabled = "A_no_filter" in enabled_methods

    print()
    print("=" * 140)
    print(
        "WINDOW-FAILURE STATS — per-method success counts and kill rates "
        "(A succeeded but other method dropped <3 RR)"
    )
    print("=" * 140)
    header = f"{'label_src':<10}{'dataset':<12}{'tier':<12}{'n_total':>8}"
    for letter in enabled_letters:
        header += f"{'n_'+letter.upper()+'_ok':>9}"
    if a_enabled:
        for letter in enabled_letters:
            if letter == "a":
                continue
            header += f"{'%'+letter.upper()+'_kill':>9}"
    header += f"{'n_all_ok':>10}"
    print(header)
    print("-" * 140)
    for _, r in fail_df.iterrows():
        line = (
            f"{r['label_source']:<10}{r['dataset']:<12}{r['tier']:<12}"
            f"{int(r['n_total']):>8}"
        )
        for letter in enabled_letters:
            col = f"n_{letter}_ok"
            line += f"{int(r[col]) if col in r and not pd.isna(r[col]) else 0:>9}"
        if a_enabled:
            for letter in enabled_letters:
                if letter == "a":
                    continue
                col = f"pct_{letter}_filter_kill"
                v = r[col] if col in r else float("nan")
                line += (
                    f"{v:>8.1f}%" if not (isinstance(v, float) and np.isnan(v)) else f"{'-':>9}"
                )
        line += f"{int(r['n_all_ok']):>10}"
        print(line)


def print_skipped(df: pd.DataFrame) -> None:
    print()
    print("=" * 100)
    print("SKIPPED WINDOWS")
    print("=" * 100)
    skipped = df[df["skip_reason"] != ""]
    if skipped.empty:
        print("  (none)")
        return
    print(f"{'dataset':<14}{'reason':<40}{'count':>8}")
    print("-" * 100)
    for (ds, reason), grp in skipped.groupby(["dataset", "skip_reason"]):
        print(f"{ds:<14}{reason:<40}{len(grp):>8d}")
    print(f"{'TOTAL skipped':<14}{'':<40}{len(skipped):>8d}")
    print(f"{'TOTAL rows':<14}{'':<40}{len(df):>8d}")


def print_honest_framing() -> None:
    print()
    print("=" * 80)
    print(" HRV ABLATION — HONEST FRAMING NOTES (5-METHOD EXTENSION):")
    print("=" * 80)
    print(
        """
1. Window length: 10s -> SHORT-TERM HRV per Task Force 1996. Values
   won't match clinical 5-min HRV norms. Method comparisons within
   this fixed-length protocol are still apples-to-apples.

2. CIRCULAR DEPENDENCY: per-beat tSQI feature is rank #1 in  LR
   classifier (|coef|=6.07). LR-tier results may be confounded.
   Pseudo-tier (Orphanidou rules) is independent primary evidence.

3. Methods B/C/E with tSQI filter may over-aggressively prune beats
   on unfit windows. If filtered RR count < 3, the corresponding
   method is treated as failed (excluded from MAE comparison rather
   than counted as MAE=0). See WINDOW-FAILURE STATS for kill rates.

4. Ground truth uses ECG. BIDMC: re-derived in this script via
   prominence-based scipy.find_peaks (algorithmic-agreement caveat
   for HR/RR similar to ). PPG-DaLiA: pre-computed 'rpeaks'
   from RespiBAN chest ECG per Reiss 2019 — independent algorithm,
   strictly cleaner ablation benchmark.

5. SDNN ddof=0 (population std per Task Force 1996, matches Backend
   K. fix 2026-05-04). RMSSD = sqrt(mean(diff(RR)^2)) where the
   diffs are restricted to adjacent pairs that BOTH survive each
   method's filter (Methods C, D, E). For Methods A and B, RMSSD is
   the simple np.diff of the surviving RR series (which for Method B
   includes gap-spanning RRs by design — see note 7).

6. Bootstrap 95% CI: 1000 resamples, seed=42 (deterministic).

7. METHOD B (NAIVE) GAP-RR HANDLING —  baseline negative result:
   When tSQI >= 0.86 filter excises bad beats, surviving consecutive
   peaks are diff'ed directly -> longer "RR" intervals spanning gaps
   where bad beats lived. This is NOT the Task Force 1996 / HeartPy
   van Gent 2019 standard. We retain gap-spanning intervals to
   demonstrate the consequence of naive hard-threshold filtering.
   On motion-artifact signals where 35-53% of beats are excised,
   surviving "RRs" become 2-5x normal length, inflating SDNN/RMSSD
   catastrophically. Method B's regression is expected behavior of
   this design choice, not a bug.

8. METHOD C (LITERATURE-CORRECT tSQI FILTER):
   Source: heartpy/analysis.py L293 (van Gent et al. 2019):
       rr_list = [rr_source[i]
                  if b_peaklist[i] + b_peaklist[i+1] == 2]
   Vectorized: keep RR[i] iff peaks[i] AND peaks[i+1] are BOTH
   accepted (tSQI >= 0.86). Gap-spanning RRs CANNOT exist by
   construction. Plus a Task Force 1996 physiological gate
   [300, 2000] ms on survivors. Adjacency-aware RMSSD.

9. METHOD D (PRODUCTION HEARTPY 2-STAGE — no per-beat tSQI):
   Stage 1: keep RR in [300, 2000] ms (Task Force 1996
            = HR in [30, 200] BPM).
   Stage 2: keep RR with quotient in [0.8, 1.2] of running median
            (van Gent 2019 HeartPy default RR_OUTLIER_TOLERANCE=0.20).
   Mirrors backend/main.py:_reject_rr_outliers (post K. fix
   2026-05-04, ml/main.py parity preserved). RR-level
   filtering — does not depend on per-beat tSQI.

10. METHOD E (COMBINED — C then Stage 2):
    Apply Method C tSQI mask + physiological gate, then on survivors
    apply Stage 2 ±20% median quotient (van Gent 2019). Tests
    whether per-beat tSQI adds value over RR-level filtering alone.

11. MATCHED COMPARISON: rows in 'matched' mode require gt_ok AND
    every enabled method to have succeeded on the same window
    (all-ok intersection). Subsets are smaller than per-method
    success counts but apples-to-apples across all enabled methods.

12. PRODUCTION-PARITY CAVEAT FOR METHOD D:
    This script's Method D RMSSD uses adjacency-aware diffs (only
    pairs (i, i+1) where both RRs survive Stage 1+2). Production
    main.py applies np.diff to surviving RRs directly (potentially
    spanning gaps). The SDNN definitions match (ddof=0 of survivors).
    Choice is to keep apples-to-apples with Methods C and E rather
    than to reproduce the production RMSSD inflation pattern; it is
    the stricter Task Force 1996 / van Gent 2019 reading.

13. MATCHED-N DRIFT vs  BASELINE:
     reported Method A unfit DaLiA pseudo MAE = 142.85 ms on
    matched N=4539 (A+B intersection only). The + extension
    enforces 5-method matched intersection -> N=1871 (Methods D, E
    eliminate windows with extreme RR > 2000 ms via Stage 1
    physiological gate, which coincide with Method A's worst errors).
    The 5-method matched MAE_A drops mechanically to 132.84 ms.
    Per-method-success rows (per_method_success in summary CSV) keep
    A's full N=7061 -> 146.43 ms for full-population reference. When
    citing improvement %, always state N (matched_1871 vs
    per_method_success_7061 are NOT directly comparable).

14. <6 PEAKS ASYMMETRY:
    compute_tsqi_per_beat requires >= template_n+1 = 6 valid peaks
    to fit the running template. Windows with 3-5 detected peaks
    yield empty tsqi_list -> Methods B/C/E always skip (treated as
    method-failed). Method A still computes on raw peaks. This
    inflates Method A per_method_success N relative to B/C/E. In
    matched mode, these windows are excluded by construction (no
    asymmetry). In per_method_success mode, beware unequal denominators.
"""
    )
    print("=" * 80)


# ============================================================
# Figures
# ============================================================
def _plot_metric_bars(
    ax,
    summary: pd.DataFrame,
    label_source: str,
    metric: str,
    methods_to_plot: list[str],
    comparison_mode: str = "per_method_success",
) -> None:
    """Grouped bar chart: x = tier, hue = (dataset, method).

    `methods_to_plot` controls which methods appear (subset of METHODS).
    """
    sub = summary[
        (summary.label_source == label_source)
        & (summary.metric == metric)
        & (summary.comparison_mode == comparison_mode)
        & (summary.method.isin(methods_to_plot))
    ]
    datasets = sorted(sub["dataset"].unique())
    n_groups = len(datasets) * len(methods_to_plot)
    bar_w = 0.8 / max(n_groups, 1)

    for i, ds in enumerate(datasets):
        for j, method in enumerate(methods_to_plot):
            cell = sub[(sub.dataset == ds) & (sub.method == method)].set_index("tier").reindex(TIERS)
            means = cell["mae_mean"].to_numpy(dtype=float)
            ci_lo = cell["mae_ci95_lo"].to_numpy(dtype=float)
            ci_hi = cell["mae_ci95_hi"].to_numpy(dtype=float)
            ns = cell["n_windows"].to_numpy(dtype=float)
            offset = (i * len(methods_to_plot) + j) * bar_w - (n_groups - 1) * bar_w / 2
            xs = np.arange(len(TIERS)) + offset
            yerr_lo = np.clip(np.nan_to_num(means - ci_lo, nan=0.0), 0, None)
            yerr_hi = np.clip(np.nan_to_num(ci_hi - means, nan=0.0), 0, None)
            method_letter = METHOD_LETTERS[method].upper()
            color = METHOD_COLORS[method]
            hatch = METHOD_HATCHES[method]
            edge = "black" if i == 0 else "#444444"
            ax.bar(
                xs,
                np.nan_to_num(means, nan=0.0),
                bar_w * 0.95,
                yerr=[yerr_lo, yerr_hi],
                color=color,
                alpha=0.85 if i == 0 else 0.55,
                edgecolor=edge,
                linewidth=0.5,
                capsize=2,
                hatch=hatch,
                label=f"{ds} M{method_letter}",
            )
            for x, m, n in zip(xs, means, ns):
                if not np.isnan(m) and n > 0:
                    ax.text(x, m, f"{int(n)}", ha="center", va="bottom", fontsize=5)

    ax.set_xticks(np.arange(len(TIERS)))
    ax.set_xticklabels(TIERS)
    ax.grid(axis="y", linestyle=":", alpha=0.4)


def fig_hrv_ablation_metric(
    summary: pd.DataFrame,
    metric: str,
    methods_to_plot: list[str],
    out_name: str,
    title_methods_label: str,
) -> Path:
    """One figure per metric: 2 subplots (pseudo + LR), grouped bars."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharey=True)
    titles = ["Pseudo-label tier (Orphanidou rules)", "LR-predicted tier ()"]
    label_sources = ["pseudo", "lr_pred"]
    for ax, ls, title in zip(axes, label_sources, titles):
        _plot_metric_bars(ax, summary, ls, metric, methods_to_plot)
        ax.set_title(title)
        ax.set_xlabel("SQA tier")
    axes[0].set_ylabel(f"{metric.upper()} MAE (ms)")
    axes[0].legend(loc="upper left", fontsize=7, framealpha=0.9, ncol=3)
    fig.suptitle(
        f"{metric.upper()} estimation MAE — {title_methods_label} (95% bootstrap CI)"
    )
    plt.tight_layout()
    out = FIG_DIR / out_name
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_hrv_ablation_improvement(
    imp: pd.DataFrame,
    methods_to_plot: list[str],
    out_name: str,
    title_methods_label: str,
) -> Path:
    """% improvement bar chart vs Method A baseline. 2 rows (metric) x 2
    cols (label_source). Each panel: x = tier, grouped bars per dataset
    with one sub-bar per non-A method. Positive = green (target wins),
    negative = red (target loses).
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharey=True)
    label_sources = ["pseudo", "lr_pred"]
    datasets = sorted(imp["dataset"].unique())
    non_a_methods = [m for m in methods_to_plot if m != "A_no_filter"]
    comparisons = [
        (f"improvement_{METHOD_LETTERS[m]}_vs_a",
         f"M{METHOD_LETTERS[m].upper()} vs A",
         METHOD_HATCHES[m] or "//")
        for m in non_a_methods
    ]
    n_groups = len(datasets) * max(len(comparisons), 1)
    bar_w = 0.85 / max(n_groups, 1)

    for col, ls in enumerate(label_sources):
        for row, metric in enumerate(METRICS):
            ax = axes[row, col]
            sub = imp[(imp.label_source == ls) & (imp.metric == metric)]
            for di, ds in enumerate(datasets):
                ds_sub = sub[sub.dataset == ds].set_index("tier").reindex(TIERS)
                for ci, (col_name, comp_label, hatch) in enumerate(comparisons):
                    if col_name not in ds_sub.columns:
                        continue
                    vals = ds_sub[col_name].to_numpy(dtype=float)
                    offset = (
                        di * len(comparisons) + ci
                    ) * bar_w - (n_groups - 1) * bar_w / 2
                    xs = np.arange(len(TIERS)) + offset
                    colors = [
                        "#27ae60" if (not np.isnan(v) and v >= 0) else "#c0392b"
                        for v in vals
                    ]
                    ax.bar(
                        xs,
                        np.nan_to_num(vals, nan=0.0),
                        bar_w * 0.95,
                        color=colors,
                        edgecolor="black",
                        linewidth=0.3,
                        hatch=hatch,
                        alpha=0.85 if di == 0 else 0.55,
                        label=f"{ds} {comp_label}" if (row == 0 and col == 0) else None,
                    )
            ax.axhline(0, color="black", linewidth=0.6)
            ax.set_xticks(np.arange(len(TIERS)))
            ax.set_xticklabels(TIERS)
            ax.set_title(
                f"{metric.upper()} — "
                f"{'pseudo' if ls == 'pseudo' else 'LR-predicted'} tier"
            )
            if col == 0:
                ax.set_ylabel("% improvement vs Method A")
            ax.grid(axis="y", linestyle=":", alpha=0.4)
            if row == 0 and col == 0:
                ax.legend(loc="upper right", fontsize=7, framealpha=0.9, ncol=2)

    fig.suptitle(
        f"% improvement vs Method A — {title_methods_label} "
        "(positive = wins, negative = loses)"
    )
    plt.tight_layout()
    out = FIG_DIR / out_name
    fig.savefig(out)
    plt.close(fig)
    return out


# ============================================================
# CLI parsing
# ============================================================
def parse_methods_arg(arg: str) -> set[str]:
    """Parse --methods 'A,B,C,D,E' into the canonical method-name set."""
    out: set[str] = set()
    for tok in arg.split(","):
        tok = tok.strip().upper()
        if not tok:
            continue
        if tok not in LETTER_TO_METHOD:
            raise SystemExit(
                f"ERROR: unknown method letter '{tok}'. Valid: A, B, C, D, E."
            )
        out.add(LETTER_TO_METHOD[tok])
    if not out:
        raise SystemExit("ERROR: --methods produced empty set.")
    return out


def parse_argv(argv: list[str]) -> set[str]:
    """Tiny argparse-free parser. Default: all 5 methods."""
    if "--help" in argv or "-h" in argv:
        print(__doc__)
        sys.exit(0)
    enabled = set(METHODS)
    if "--methods" in argv:
        i = argv.index("--methods")
        if i + 1 >= len(argv):
            raise SystemExit("ERROR: --methods requires a value (e.g. A,B,C,D,E).")
        enabled = parse_methods_arg(argv[i + 1])
    return enabled


# ============================================================
# Console: 5-method headline summary (per task spec)
# ============================================================
def print_headline_table(
    summary: pd.DataFrame, enabled_methods: set[str]
) -> None:
    """Headline table requested in task spec, MATCHED comparison only:
        Method | Dataset | Tier | N_matched | MAE_SDNN | MAE_RMSSD | Improvement_vs_A
    """
    src = summary[summary.comparison_mode == "matched"]
    if src.empty:
        print("\n(No matched-mode rows to print.)")
        return
    print()
    print("=" * 110)
    print(
        "HEADLINE — 5-method matched (all-ok intersection) MAE comparison"
    )
    print("=" * 110)
    print(
        f"{'Method':<8}{'Dataset':<14}{'Tier':<12}{'N':>8}"
        f"{'MAE_SDNN':>12}{'MAE_RMSSD':>12}{'Diff_SDNN_vs_A':>16}{'Diff_RMSSD_vs_A':>17}"
    )
    print("-" * 110)
    for label_source in ["pseudo", "lr_pred"]:
        ls_src = src[src.label_source == label_source]
        if ls_src.empty:
            continue
        print(f"  [label_source = {label_source}]")
        for ds in sorted(ls_src["dataset"].unique()):
            for tier in TIERS:
                # MAE_A baseline (per metric).
                base_sdnn = ls_src[
                    (ls_src.dataset == ds)
                    & (ls_src.tier == tier)
                    & (ls_src.method == "A_no_filter")
                    & (ls_src.metric == "sdnn")
                ]
                base_rmssd = ls_src[
                    (ls_src.dataset == ds)
                    & (ls_src.tier == tier)
                    & (ls_src.method == "A_no_filter")
                    & (ls_src.metric == "rmssd")
                ]
                base_sdnn_mae = (
                    float(base_sdnn["mae_mean"].iloc[0])
                    if not base_sdnn.empty
                    else float("nan")
                )
                base_rmssd_mae = (
                    float(base_rmssd["mae_mean"].iloc[0])
                    if not base_rmssd.empty
                    else float("nan")
                )
                for method in METHODS:
                    if method not in enabled_methods:
                        continue
                    sdnn_row = ls_src[
                        (ls_src.dataset == ds)
                        & (ls_src.tier == tier)
                        & (ls_src.method == method)
                        & (ls_src.metric == "sdnn")
                    ]
                    rmssd_row = ls_src[
                        (ls_src.dataset == ds)
                        & (ls_src.tier == tier)
                        & (ls_src.method == method)
                        & (ls_src.metric == "rmssd")
                    ]
                    if sdnn_row.empty:
                        continue
                    n = int(sdnn_row["n_windows"].iloc[0])
                    if n == 0:
                        continue
                    mae_sdnn = float(sdnn_row["mae_mean"].iloc[0])
                    mae_rmssd = (
                        float(rmssd_row["mae_mean"].iloc[0])
                        if not rmssd_row.empty
                        else float("nan")
                    )
                    method_letter = METHOD_LETTERS[method].upper()
                    if method == "A_no_filter":
                        ds_str = "(baseline)"
                        rs_str = "(baseline)"
                    else:
                        if base_sdnn_mae > 0 and not np.isnan(mae_sdnn):
                            ds_str = (
                                f"{(base_sdnn_mae - mae_sdnn) / base_sdnn_mae * 100:+.1f}%"
                            )
                        else:
                            ds_str = "-"
                        if base_rmssd_mae > 0 and not np.isnan(mae_rmssd):
                            rs_str = (
                                f"{(base_rmssd_mae - mae_rmssd) / base_rmssd_mae * 100:+.1f}%"
                            )
                        else:
                            rs_str = "-"
                    print(
                        f"  {method_letter:<6}{ds:<14}{tier:<12}{n:>8}"
                        f"{mae_sdnn:>12.2f}{mae_rmssd:>12.2f}"
                        f"{ds_str:>16}{rs_str:>17}"
                    )
                print("  " + "-" * 106)


# ============================================================
# Main
# ============================================================
def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    enabled_methods = parse_argv(argv)

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

    work = cls_df[cls_df["dataset"].isin(["BIDMC", "PPG-DaLiA"])].copy()
    print(
        f"  Filtered to BIDMC + PPG-DaLiA: {len(work)} windows "
        f"(BIDMC={int((work.dataset=='BIDMC').sum())}, "
        f"PPG-DaLiA={int((work.dataset=='PPG-DaLiA').sum())})"
    )

    enabled_letters = sorted(METHOD_LETTERS[m] for m in enabled_methods)
    print(
        f"  Enabled methods: {[m.upper() for m in enabled_letters]} "
        f"(use --methods A,B,C,D,E to override)"
    )

    print("\nDeriving SDNN/RMSSD per window for enabled methods + ground truth...")
    t0 = time.time()
    out_rows: list[dict] = []
    n = len(work)
    for i, (_, row) in enumerate(work.iterrows()):
        if i % 500 == 0 and i > 0:
            print(f"  {i}/{n} ({i/n*100:.0f}%) elapsed {time.time()-t0:.0f}s")
        if row["dataset"] == "BIDMC":
            info = eval_bidmc_row(row, enabled_methods)
        else:
            info = eval_dalia_row(row, enabled_methods)

        # Build error columns (only valid where method ok AND gt ok).
        def _err(method_letter: str, metric: str) -> float:
            method_failed = info[f"method_{method_letter}_failed"]
            if method_failed or info["gt_failed"]:
                return float("nan")
            return abs(info[f"{metric}_{method_letter}"] - info[f"{metric}_gt"])

        rec = {
            "file": row["file"],
            "dataset": row["dataset"],
            "subject_id": row["subject_id"],
            "window_idx": int(row["window_idx"]),
            "t_start_s": float(row["t_start_s"]),
            "t_end_s": float(row["t_end_s"]),
            "fs": int(row["fs"]),
            "n_peaks_total": int(info["n_peaks_total"]),
            "n_peaks_kept_b": int(info["n_peaks_kept_b"]),
            "n_rr_kept_c": int(info["n_rr_kept_c"]),
            "n_rr_kept_d": int(info["n_rr_kept_d"]),
            "n_rr_kept_e": int(info["n_rr_kept_e"]),
            "n_dropped_c": int(info["n_dropped_c"]),
            "n_dropped_d": int(info["n_dropped_d"]),
            "n_dropped_e": int(info["n_dropped_e"]),
            "sdnn_gt": info["sdnn_gt"],
            "rmssd_gt": info["rmssd_gt"],
            "gt_failed": bool(info["gt_failed"]),
            "skip_reason": info["skip_reason"],
            "sqa_tier_pseudo": row["label"],
            "sqa_tier_lr_pred": row["label_pred_lr"],
        }
        for letter in ["a", "b", "c", "d", "e"]:
            rec[f"sdnn_{letter}"] = info[f"sdnn_{letter}"]
            rec[f"rmssd_{letter}"] = info[f"rmssd_{letter}"]
            rec[f"err_sdnn_{letter}"] = _err(letter, "sdnn")
            rec[f"err_rmssd_{letter}"] = _err(letter, "rmssd")
            rec[f"method_{letter}_failed"] = bool(info[f"method_{letter}_failed"])
        out_rows.append(rec)
    elapsed = time.time() - t0
    print(f"  Done {n}/{n} in {elapsed:.1f}s")

    df = pd.DataFrame(out_rows)
    df.to_csv(OUT_RESULTS_CSV, index=False)
    print(f"\nSaved per-window results: {OUT_RESULTS_CSV.name} ({len(df)} rows)")

    print_skipped(df)

    summary_pseudo = summarize_by_tier_method(
        df, "sqa_tier_pseudo", "pseudo", enabled_methods
    )
    summary_lr = summarize_by_tier_method(
        df, "sqa_tier_lr_pred", "lr_pred", enabled_methods
    )
    summary = pd.concat([summary_pseudo, summary_lr], ignore_index=True)
    summary.to_csv(OUT_SUMMARY_CSV, index=False)
    print(f"Saved summary table: {OUT_SUMMARY_CSV.name} ({len(summary)} rows)")

    # PRIMARY honest comparison: matched (all-ok intersection) subset.
    print_summary_table(summary, comparison_mode="matched")
    # Transparency: per-method success (different N per method).
    print_summary_table(summary, comparison_mode="per_method_success")

    fail_df = compute_window_failure_stats(df, enabled_methods)
    print_window_failures(fail_df, enabled_methods)

    # PRIMARY: matched comparison.
    imp_matched = compute_improvement_table(
        summary, enabled_methods, comparison_mode="matched"
    )
    print_improvement_table(imp_matched, enabled_methods)
    # Transparency: per-method-success comparison.
    imp_per_method = compute_improvement_table(
        summary, enabled_methods, comparison_mode="per_method_success"
    )
    print_improvement_table(imp_per_method, enabled_methods)

    # Headline table per task spec.
    print_headline_table(summary, enabled_methods)

    print("\nGenerating figures...")
    #  reproduction figures: A, B, D (where D = production HeartPy
    # 2-stage, was previously labeled "C_heartpy_rr" in  file). Only
    # plot if the relevant methods are enabled.
    day7_methods = [
        m for m in ["A_no_filter", "B_tsqi_naive", "D_heartpy_2stage"]
        if m in enabled_methods
    ]
    if len(day7_methods) >= 2:
        f1 = fig_hrv_ablation_metric(
            summary, "sdnn", day7_methods,
            "fig_hrv_ablation_sdnn.pdf",
            "A (no filter) vs B (naive tSQI) vs D (production HeartPy 2-stage)",
        )
        print(f"  {f1.name}")
        f2 = fig_hrv_ablation_metric(
            summary, "rmssd", day7_methods,
            "fig_hrv_ablation_rmssd.pdf",
            "A (no filter) vs B (naive tSQI) vs D (production HeartPy 2-stage)",
        )
        print(f"  {f2.name}")
        f3 = fig_hrv_ablation_improvement(
            imp_per_method, day7_methods,
            "fig_hrv_ablation_improvement.pdf",
            " reproduction (A/B/D)",
        )
        print(f"  {f3.name}")

    # NEW 5-method figures (per task spec). Plot all enabled methods.
    enabled_in_order = [m for m in METHODS if m in enabled_methods]
    if len(enabled_in_order) >= 2:
        method_letters_label = "+".join(METHOD_LETTERS[m].upper() for m in enabled_in_order)
        f4 = fig_hrv_ablation_metric(
            summary, "sdnn", enabled_in_order,
            "fig_hrv_ablation_methods_sdnn.pdf",
            f"5-method comparison ({method_letters_label})",
        )
        print(f"  {f4.name}")
        f5 = fig_hrv_ablation_metric(
            summary, "rmssd", enabled_in_order,
            "fig_hrv_ablation_methods_rmssd.pdf",
            f"5-method comparison ({method_letters_label})",
        )
        print(f"  {f5.name}")
        f6 = fig_hrv_ablation_improvement(
            imp_per_method, enabled_in_order,
            "fig_hrv_ablation_methods_improvement.pdf",
            f"5-method comparison ({method_letters_label})",
        )
        print(f"  {f6.name}")

    print_honest_framing()
    return 0


if __name__ == "__main__":
    sys.exit(main())
