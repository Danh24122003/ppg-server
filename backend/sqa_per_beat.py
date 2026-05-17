"""
Per-beat template-matching Signal Quality Index for PPG.

Reference: Li, Q. & Clifford, G.D. (2012).
  "Dynamic time warping and machine learning for signal quality assessment
   of pulsatile signals." Physiol Meas 33(9):1491-1501. PMID 22902950.
  DOI 10.1088/0967-3334/33/9/1491

Threshold 0.86 cho PPG per Orphanidou et al. (2015) IEEE JBHI 19(3):832-838.

This is a STANDALONE module — NOT integrated into main.py pipeline yet
(deferred to W2 of Deep SQA plan after LR classifier validation).
"""
from __future__ import annotations

import numpy as np


def compute_tsqi_per_beat(
    filtered_signal: np.ndarray,
    peak_indices: np.ndarray | list,
    fs: int,
    window_ms: int = 600,
    template_n: int = 5,
) -> tuple[list[float], np.ndarray | None]:
    """Per-beat template-matching SQI via Pearson correlation.

    Algorithm:
    1. Extract beat windows ±(window_ms/2) around each peak.
    2. Build robust running template = median of first `template_n` beats.
    3. Per-beat: Pearson r vs template → tSQI score 0-1.

    Args:
        filtered_signal: 1D array, bandpass-filtered PPG (Butterworth [0.5, 4] Hz).
        peak_indices: peak indices in filtered_signal (from HeartPy or scipy.find_peaks).
        fs: sample rate Hz.
        window_ms: half-window total = window_ms ms (default 600 → ±300ms each side).
        template_n: number of first beats for median template (default 5).

    Returns:
        (tsqi_per_beat, template):
            tsqi_per_beat: list[float] — Pearson r per beat, or 0.0 if invalid.
            template: np.ndarray of length window_samples, or None if insufficient beats.

    Edge cases handled:
        - peak_indices empty or < template_n + 1: return ([], None)
        - peaks too close to signal edges: SKIP those beats (not in output list)
        - beat with zero variance: r = 0.0
        - filtered_signal length < window_ms × 2 / 1000 × fs: return ([], None)
    """
    half_w = int(window_ms / 1000.0 * fs / 2)
    n = len(filtered_signal)

    if n < 2 * half_w + 1 or len(peak_indices) < template_n + 1:
        return [], None

    peaks = np.asarray(peak_indices, dtype=int)

    # Extract valid beats (not too close to edges).
    # Window symmetric ±half_w around peak: total length = 2*half_w + 1.
    # Peak sample is at index half_w (center). Fixed P0 from code review 8/5.
    valid_beats = []
    for p in peaks:
        if p - half_w >= 0 and p + half_w < n:
            beat = filtered_signal[p - half_w:p + half_w + 1]
            valid_beats.append(beat)

    if len(valid_beats) < template_n + 1:
        return [], None

    valid_beats_arr = np.asarray(valid_beats, dtype=float)

    # Build template from first template_n valid beats (robust median)
    template = np.median(valid_beats_arr[:template_n], axis=0)

    template_std = float(np.std(template))
    if template_std < 1e-9:
        return [0.0] * len(valid_beats_arr), template

    # Per-beat Pearson r vs template. Clip negative correlations to 0 to
    # match documented range [0, 1] — anti-phase beats are bad-quality.
    # (Fixed P1 from code review 8/5.)
    tsqi_list: list[float] = []
    for beat in valid_beats_arr:
        beat_std = float(np.std(beat))
        if beat_std < 1e-9:
            tsqi_list.append(0.0)
            continue
        r = float(np.corrcoef(beat, template)[0, 1])
        if np.isnan(r):
            r = 0.0
        tsqi_list.append(max(0.0, r))

    return tsqi_list, template


def aggregate_tsqi(tsqi_per_beat: list[float]) -> dict:
    """Aggregate per-beat tSQI into batch-level summary.

    Returns dict with:
        - tsqi_median: median Pearson r
        - tsqi_mean: mean
        - tsqi_min: min r (worst beat)
        - tsqi_max: max r
        - tsqi_std: std deviation
        - bad_beat_ratio: fraction of beats with r < 0.86 (Orphanidou threshold)
        - n_beats: total valid beats
    """
    if not tsqi_per_beat:
        return {
            "tsqi_median": 0.0,
            "tsqi_mean": 0.0,
            "tsqi_min": 0.0,
            "tsqi_max": 0.0,
            "tsqi_std": 0.0,
            "bad_beat_ratio": 1.0,
            "n_beats": 0,
        }

    arr = np.asarray(tsqi_per_beat, dtype=float)
    return {
        "tsqi_median": float(np.median(arr)),
        "tsqi_mean": float(np.mean(arr)),
        "tsqi_min": float(np.min(arr)),
        "tsqi_max": float(np.max(arr)),
        "tsqi_std": float(np.std(arr)),
        "bad_beat_ratio": float(np.sum(arr < 0.86) / len(arr)),
        "n_beats": int(len(arr)),
    }


def filter_beats_by_tsqi(
    tsqi_per_beat: list[float],
    threshold: float = 0.86,
) -> list[int]:
    """Return indices of beats with tSQI >= threshold.

    Use this to filter peak_indices before HRV computation:
    >>> good_indices = filter_beats_by_tsqi(tsqi_list, threshold=0.86)
    >>> good_peaks = [peaks[i] for i in good_indices]
    >>> good_rr = np.diff(good_peaks) / fs * 1000  # ms
    """
    return [i for i, r in enumerate(tsqi_per_beat) if r >= threshold]
