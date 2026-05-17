"""
test_sqa_per_beat.py —  Deep SQA tests for sqa_per_beat module.

Independent QA — does NOT touch main.py pipeline.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pytest
from scipy.signal import butter, filtfilt, find_peaks

_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from sqa_per_beat import (  # noqa: E402
    aggregate_tsqi,
    compute_tsqi_per_beat,
    filter_beats_by_tsqi,
)


FS = 100
N = FS * 10  # 10s — long enough to yield > template_n+1 valid (non-edge) beats


def _make_clean_sine_with_peaks(hr_bpm: int = 75, fs: int = FS, duration: int = 10):
    """Generate clean sine + return (signal, peak_indices)."""
    t = np.linspace(0, duration, fs * duration, endpoint=False)
    f = hr_bpm / 60.0
    sig = np.sin(2 * np.pi * f * t).astype(float)
    peaks, _ = find_peaks(sig, distance=int(fs * 0.3))
    return sig, peaks


class TestPerBeatTSQI:
    # ---------------------------------------------------------------- 1
    def test_clean_sine_high_correlation(self):
        sig, peaks = _make_clean_sine_with_peaks(hr_bpm=75, fs=FS, duration=10)
        tsqi_list, template = compute_tsqi_per_beat(sig, peaks, fs=FS)
        assert template is not None
        assert len(tsqi_list) > 0
        median_r = float(np.median(tsqi_list))
        assert median_r > 0.95, f"clean sine median tSQI {median_r} <= 0.95"

    # ---------------------------------------------------------------- 2
    def test_short_signal_returns_empty(self):
        # window_ms=600 → 2*half_w = 60 samples at fs=100. Make signal shorter.
        sig = np.zeros(30, dtype=float)
        peaks = np.array([10, 20])
        tsqi_list, template = compute_tsqi_per_beat(sig, peaks, fs=FS)
        assert tsqi_list == []
        assert template is None

    # ---------------------------------------------------------------- 3
    def test_few_peaks_returns_empty(self):
        sig, _ = _make_clean_sine_with_peaks(hr_bpm=75, fs=FS, duration=10)
        # < template_n + 1 = 6 peaks
        peaks = np.array([100, 200, 300, 400, 500])  # exactly 5
        tsqi_list, template = compute_tsqi_per_beat(sig, peaks, fs=FS)
        assert tsqi_list == []
        assert template is None

    # ---------------------------------------------------------------- 4
    def test_peaks_at_edges_skipped(self):
        sig, peaks = _make_clean_sine_with_peaks(hr_bpm=75, fs=FS, duration=10)
        # half_w = 30 samples for default window_ms=600 at fs=100
        half_w = int(600 / 1000.0 * FS / 2)
        n = len(sig)
        # Count peaks that already sit within [half_w, n - half_w) (interior).
        interior_count = int(
            np.sum((peaks - half_w >= 0) & (peaks + half_w < n))
        )
        # Inject edge peaks at 0 and n-1; both must be skipped by compute().
        peaks_with_edges = np.concatenate(([0], peaks, [n - 1]))
        tsqi_list, template = compute_tsqi_per_beat(sig, peaks_with_edges, fs=FS)
        assert len(tsqi_list) == interior_count, (
            f"edge peaks not skipped: got {len(tsqi_list)} vs interior {interior_count}"
        )
        assert template is not None

    # ---------------------------------------------------------------- 5
    def test_zero_variance_beat_returns_zero(self):
        sig, peaks = _make_clean_sine_with_peaks(hr_bpm=75, fs=FS, duration=10)
        # half_w samples for window_ms=600 default: 600/1000 * 100 / 2 = 30
        half_w = int(600 / 1000.0 * FS / 2)
        # Pick an interior peak well away from edges
        target_idx_in_peaks = 5  # 6th peak
        target_peak = int(peaks[target_idx_in_peaks])
        sig_modified = sig.copy()
        sig_modified[target_peak - half_w:target_peak + half_w + 1] = 5.0  # constant (window inclusive)
        tsqi_list, _ = compute_tsqi_per_beat(sig_modified, peaks, fs=FS)
        # That peak corresponds to the same index in valid_beats list (no peaks
        # were skipped — all interior). Verify r == 0 for the modified beat.
        # Find which valid index corresponds to target_peak (skipping edge peaks).
        # Simple: count interior peaks before target.
        valid_idx = sum(
            1 for p in peaks[:target_idx_in_peaks]
            if (p - half_w >= 0) and (p + half_w < len(sig_modified))
        )
        assert tsqi_list[valid_idx] == 0.0, (
            f"zero-variance beat got r={tsqi_list[valid_idx]}"
        )

    # ---------------------------------------------------------------- 6
    def test_aggregate_tsqi_returns_all_keys(self):
        out = aggregate_tsqi([0.95, 0.5, 0.9, 0.3])
        expected = {
            "tsqi_median",
            "tsqi_mean",
            "tsqi_min",
            "tsqi_max",
            "tsqi_std",
            "bad_beat_ratio",
            "n_beats",
        }
        assert set(out.keys()) == expected
        assert out["n_beats"] == 4
        assert out["tsqi_max"] == 0.95
        assert out["tsqi_min"] == 0.3

    # ---------------------------------------------------------------- 7
    def test_aggregate_empty_returns_zeros(self):
        out = aggregate_tsqi([])
        assert out["tsqi_median"] == 0.0
        assert out["tsqi_mean"] == 0.0
        assert out["tsqi_min"] == 0.0
        assert out["tsqi_max"] == 0.0
        assert out["tsqi_std"] == 0.0
        assert out["bad_beat_ratio"] == 1.0
        assert out["n_beats"] == 0

    # ---------------------------------------------------------------- 8
    def test_filter_beats_threshold_086(self):
        out = filter_beats_by_tsqi([0.95, 0.5, 0.9, 0.3], threshold=0.86)
        assert out == [0, 2]

    # ---------------------------------------------------------------- 9
    def test_realistic_quoc_p1(self):
        csv_path = os.path.normpath(
            os.path.join(
                _BACKEND_DIR,
                "..",
                "collected_data",
                "sqa_experiment",
                "S09_p1_20260507_153619.csv",
            )
        )
        if not os.path.exists(csv_path):
            pytest.skip(f"S09_p1 CSV not available: {csv_path}")

        # Load: skip 5 metadata comment lines + 1 column header row = 6
        data = np.genfromtxt(
            csv_path,
            delimiter=",",
            skip_header=6,
            usecols=(0, 1, 2),
            dtype=float,
        )
        ir = data[:, 1]
        # fs from metadata header line 4: fs=104. Use median timestamp diff for robust fs.
        ts_ms = data[:, 0]
        diffs = np.diff(ts_ms)
        diffs = diffs[diffs > 0]
        fs = int(round(1000.0 / float(np.median(diffs)))) if len(diffs) else 104

        # Bandpass [0.5, 4.0] Hz Butterworth order=4 (matches main.py convention)
        b, a = butter(4, [0.5, 4.0], btype="band", fs=fs)
        ir_filt = filtfilt(b, a, ir)

        # Peak detection: distance based on max HR 200 BPM → 0.3 s
        peaks, _ = find_peaks(ir_filt, distance=int(fs * 0.3))

        tsqi_list, template = compute_tsqi_per_beat(ir_filt, peaks, fs=fs)
        agg = aggregate_tsqi(tsqi_list)

        # Stash for report
        TestPerBeatTSQI._quoc_p1_n_beats = agg["n_beats"]
        TestPerBeatTSQI._quoc_p1_median = agg["tsqi_median"]

        assert agg["n_beats"] >= 100, (
            f"expected ~150-200 beats, got {agg['n_beats']}"
        )
        assert agg["tsqi_median"] > 0.6, (
            f"clean resting expected median > 0.6, got {agg['tsqi_median']}"
        )

    # ---------------------------------------------------------------- 10
    def test_performance_under_5ms(self):
        sig, peaks = _make_clean_sine_with_peaks(hr_bpm=75, fs=FS, duration=5)
        assert len(sig) == 500  # 500-sample synthetic (per spec)

        iters = 100
        start = time.perf_counter()
        for _ in range(iters):
            compute_tsqi_per_beat(sig, peaks, fs=FS)
        elapsed = time.perf_counter() - start
        mean_ms = (elapsed / iters) * 1000.0

        # Stash for report
        TestPerBeatTSQI._mean_latency_ms = mean_ms

        assert mean_ms < 5.0, f"mean latency {mean_ms:.3f} ms >= 5 ms"
