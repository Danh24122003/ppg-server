"""
test_sqa_edge_cases.py — Independent QA: edge-case & cross-server tests for
compute_sqi_features() / assess_signal_quality()

 Deep SQA implementation verification (2026-05-07).
DO NOT modify backend/main.py or ml/main.py.
"""

import sys
import os
import math
import time
import importlib

import numpy as np
import pytest
from scipy.signal import butter, filtfilt, find_peaks as _find_peaks

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_BACKEND_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)))
)
_ML_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ML code", "ml")
)

if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

# Import Backend symbols
from main import compute_sqi_features as backend_sqi, assess_signal_quality as backend_asg  # noqa: E402

# Try importing ML server symbols (may fail if ml deps absent)
_ML_AVAILABLE = False
try:
    if _ML_DIR not in sys.path:
        sys.path.insert(0, _ML_DIR)
    # Avoid re-running FastAPI startup by importing the module object directly
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location("ml_main", os.path.join(_ML_DIR, "main.py"))
    _ml_mod = _ilu.module_from_spec(_spec)
    # Suppress startup side-effects by pre-stubbing app-level globals minimally
    import unittest.mock as _mock
    with _mock.patch("fastapi.FastAPI", return_value=_mock.MagicMock()):
        with _mock.patch("slowapi.Limiter", return_value=_mock.MagicMock()):
            try:
                _spec.loader.exec_module(_ml_mod)
                ml_sqi = _ml_mod.compute_sqi_features
                _ML_AVAILABLE = True
            except Exception:
                _ML_AVAILABLE = False
except Exception:
    _ML_AVAILABLE = False

# ---------------------------------------------------------------------------
# Shared constants & helpers
# ---------------------------------------------------------------------------
FS = 100
N = 500
DC = 50_000.0


def _make_bandpass(signal: np.ndarray, fs: int = FS) -> np.ndarray:
    b, a = butter(4, [0.5, 4.0], btype="band", fs=fs)
    return filtfilt(b, a, signal)


def _make_peaks(filtered: np.ndarray, fs: int = FS) -> list:
    peaks, _ = _find_peaks(filtered, distance=int(fs * 0.3))
    return peaks.tolist()


def _sine_signal(freq_hz: float = 1.25, n: int = N, fs: int = FS,
                 dc: float = DC, amp: float = 2000.0) -> np.ndarray:
    t = np.arange(n) / fs
    return dc + amp * np.sin(2 * math.pi * freq_hz * t)


# ---------------------------------------------------------------------------
# TC1: Empty / very short signal — must NOT crash
# ---------------------------------------------------------------------------
class TestShortSignals:

    def test_single_sample_no_crash(self):
        """compute_sqi_features must not raise on length-1 arrays."""
        raw = np.array([50000.0])
        filt = np.array([0.0])
        feats = backend_sqi(raw, filt, [], fs=FS)
        assert isinstance(feats, dict)
        assert set(feats.keys()) == {
            "spectral_purity", "hr_bpm", "amplitude_ratio",
            "ssqi", "ksqi", "entropy_amp_dist"
        }

    def test_short_signal_below_8_ssqi_ksqi_entropy_are_zero(self):
        """When len(filtered) < 8, sSQI / kSQI / entropy_amp_dist must stay at 0.0."""
        raw = np.ones(7) * DC
        filt = np.zeros(7)
        feats = backend_sqi(raw, filt, [], fs=FS)
        assert feats["ssqi"] == 0.0, f"expected 0.0, got {feats['ssqi']}"
        assert feats["ksqi"] == 0.0, f"expected 0.0, got {feats['ksqi']}"
        assert feats["entropy_amp_dist"] == 0.0, f"expected 0.0, got {feats['entropy_amp_dist']}"

    def test_short_signal_spectral_purity_is_zero_when_nperseg_too_small(self):
        """nperseg = len(filt)//2 < 16 → spectral_purity = 0.0."""
        raw = np.ones(20) * DC
        filt = np.zeros(20)  # 20 // 2 = 10 < 16
        feats = backend_sqi(raw, filt, [], fs=FS)
        assert feats["spectral_purity"] == 0.0

    def test_exactly_8_samples_computes_ssqi(self):
        """Length-8 filtered signal with non-zero std should compute sSQI != 0.0."""
        rng = np.random.default_rng(0)
        raw = np.ones(8) * DC
        filt = rng.standard_normal(8) * 100.0
        feats = backend_sqi(raw, filt, [], fs=FS)
        # std > 1e-9 → sSQI/kSQI computed; they may be any finite value
        assert np.isfinite(feats["ssqi"])
        assert np.isfinite(feats["ksqi"])

    def test_empty_arrays_no_crash(self):
        """Completely empty arrays must not raise."""
        raw = np.array([], dtype=float)
        filt = np.array([], dtype=float)
        feats = backend_sqi(raw, filt, [], fs=FS)
        assert isinstance(feats, dict)


# ---------------------------------------------------------------------------
# TC2: All-zero filtered signal — divide-by-zero guard
# ---------------------------------------------------------------------------
class TestZeroFilteredSignal:

    def test_zero_filtered_no_divide_by_zero(self):
        """std(filtered)==0 → sSQI/kSQI/entropy_amp_dist must stay 0.0, not NaN/Inf."""
        raw = np.ones(N) * DC
        filt = np.zeros(N)
        feats = backend_sqi(raw, filt, [], fs=FS)
        assert feats["ssqi"] == 0.0
        assert feats["ksqi"] == 0.0
        assert feats["entropy_amp_dist"] == 0.0
        assert not np.isnan(feats["spectral_purity"])
        assert not np.isinf(feats["amplitude_ratio"])

    def test_near_zero_std_guard(self):
        """std = 5e-10 (< 1e-9 threshold) → sSQI branch skipped."""
        filt = np.ones(N) * 1e-10  # non-zero but std effectively 0
        feats = backend_sqi(np.ones(N) * DC, filt, [], fs=FS)
        assert feats["ssqi"] == 0.0

    def test_amplitude_ratio_nonzero_when_raw_positive(self):
        """amplitude_ratio = ptp(filtered) / (mean(|raw|) + 1e-12) must be finite."""
        raw = np.ones(N) * DC
        filt = np.zeros(N)
        feats = backend_sqi(raw, filt, [], fs=FS)
        assert np.isfinite(feats["amplitude_ratio"])
        assert feats["amplitude_ratio"] >= 0.0


# ---------------------------------------------------------------------------
# TC3: Negative skewness signal (left-tail)
# ---------------------------------------------------------------------------
class TestSkewnessKurtosis:

    def test_negative_skewness_returned_as_is(self):
        """A left-skewed signal (clipped sines) should yield ssqi < 0."""
        rng = np.random.default_rng(7)
        t = np.arange(N) / FS
        # Right-clipping creates left tail → negative skew
        sig = np.clip(2000.0 * np.sin(2 * math.pi * 1.2 * t), -np.inf, 500.0)
        sig += rng.standard_normal(N) * 50.0
        feats = backend_sqi(np.ones(N) * DC, sig, [], fs=FS)
        assert feats["ssqi"] < 0.0, f"expected negative sSQI, got {feats['ssqi']}"

    def test_positive_skewness_returned_as_is(self):
        """A right-skewed signal (left-clipped) should yield ssqi > 0."""
        t = np.arange(N) / FS
        sig = np.clip(2000.0 * np.sin(2 * math.pi * 1.2 * t), -500.0, np.inf)
        feats = backend_sqi(np.ones(N) * DC, sig, [], fs=FS)
        assert feats["ssqi"] > 0.0, f"expected positive sSQI, got {feats['ssqi']}"

    def test_high_kurtosis_signal_returns_positive_ksqi(self):
        """Signal with spikes → Fisher excess kurtosis > 0 (heavy tails)."""
        sig = np.zeros(N)
        # Insert impulses: heavy-tail distribution
        sig[::50] = 10000.0
        feats = backend_sqi(np.ones(N) * DC, sig, [], fs=FS)
        assert feats["ksqi"] > 0.0, f"expected kSQI > 0 for impulsive signal, got {feats['ksqi']}"

    def test_gaussian_noise_kurtosis_near_three(self):
        """Gaussian noise → raw Pearson kurtosis ≈ 3 (Elgendi 2016 formula).
        Fisher=False per cross-server kSQI fix  (8/5).
        """
        rng = np.random.default_rng(42)
        sig = rng.standard_normal(2000) * 500.0
        feats = backend_sqi(np.ones(2000) * DC, sig, [], fs=FS)
        assert 1.5 < feats["ksqi"] < 4.5, \
            f"Gaussian raw kSQI should be near 3, got {feats['ksqi']}"


# ---------------------------------------------------------------------------
# TC4: Entropy SQI bounds
# ---------------------------------------------------------------------------
class TestEntropySQI:

    def test_entropy_amp_dist_always_nonnegative(self):
        """Shannon entropy must be >= 0 for any valid signal."""
        for seed in [0, 1, 99]:
            rng = np.random.default_rng(seed)
            sig = rng.standard_normal(N) * 500.0
            feats = backend_sqi(np.ones(N) * DC, sig, [], fs=FS)
            assert feats["entropy_amp_dist"] >= 0.0, \
                f"entropy negative (seed={seed}): {feats['entropy_amp_dist']}"

    def test_entropy_amp_dist_finite(self):
        """Entropy must be a finite float for all typical inputs."""
        raw = _sine_signal()
        filt = _make_bandpass(raw)
        feats = backend_sqi(raw, filt, _make_peaks(filt), fs=FS)
        assert np.isfinite(feats["entropy_amp_dist"])


# ---------------------------------------------------------------------------
# TC5: Cross-server consistency — Backend vs ML main.py
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not _ML_AVAILABLE, reason="ML server module not importable in this env")
class TestCrossServerConsistency:

    def test_same_input_same_output_clean_sine(self):
        """Backend and ML compute_sqi_features must return identical floats."""
        raw = _sine_signal(freq_hz=1.25)
        filt = _make_bandpass(raw)
        peaks = _make_peaks(filt)
        b_feats = backend_sqi(raw, filt, peaks, fs=FS)
        m_feats = ml_sqi(raw, filt, peaks, fs=FS)
        for key in b_feats:
            assert key in m_feats, f"ML missing key '{key}'"
            assert np.isclose(b_feats[key], m_feats[key], atol=1e-9), \
                f"Mismatch on '{key}': backend={b_feats[key]}, ml={m_feats[key]}"

    def test_same_input_same_output_noisy_signal(self):
        """Cross-server consistency on noisy signal."""
        rng = np.random.default_rng(13)
        raw = _sine_signal() + rng.standard_normal(N) * 200
        filt = _make_bandpass(raw)
        peaks = _make_peaks(filt)
        b_feats = backend_sqi(raw, filt, peaks, fs=FS)
        m_feats = ml_sqi(raw, filt, peaks, fs=FS)
        assert set(b_feats.keys()) == set(m_feats.keys())
        for key in b_feats:
            assert np.isclose(b_feats[key], m_feats[key], atol=1e-9), \
                f"Noisy signal mismatch on '{key}': {b_feats[key]} vs {m_feats[key]}"


# ---------------------------------------------------------------------------
# TC6: Backward compat with realistic synthetic PPG (no CSV available)
# ---------------------------------------------------------------------------
class TestRealisticPPG:

    def _make_realistic_ppg(self):
        """
        Synthetic finger PPG: DC=50000 + 1.2 Hz fundamental + 2nd harmonic.
        Mimics typical MAX30102 waveform morphology.
        """
        t = np.arange(N) / FS
        sig = (DC
               + 2000.0 * np.sin(2 * math.pi * 1.2 * t)
               + 400.0 * np.sin(2 * math.pi * 2.4 * t + 0.3))
        return sig.astype(np.float64)

    def test_assess_signal_quality_returns_good_for_clean_ppg(self):
        """Clean synthetic PPG must be rated 'good'."""
        raw = self._make_realistic_ppg()
        filt = _make_bandpass(raw)
        peaks = _make_peaks(filt)
        quality = backend_asg(raw, filt, peaks, FS)
        assert quality == "good", f"Expected 'good', got '{quality}'"

    def test_compute_sqi_all_keys_finite_for_realistic_ppg(self):
        """All 6 SQI features must be finite floats on a realistic signal."""
        raw = self._make_realistic_ppg()
        filt = _make_bandpass(raw)
        peaks = _make_peaks(filt)
        feats = backend_sqi(raw, filt, peaks, FS)
        for key, val in feats.items():
            assert isinstance(val, float), f"{key} not float"
            assert np.isfinite(val), f"{key} not finite: {val}"

    def test_spectral_purity_high_for_clean_ppg(self):
        """Spectral purity > 0.60 for a clean PPG-like sine."""
        raw = self._make_realistic_ppg()
        filt = _make_bandpass(raw)
        feats = backend_sqi(raw, filt, _make_peaks(filt), FS)
        assert feats["spectral_purity"] > 0.60, \
            f"Expected purity > 0.60, got {feats['spectral_purity']}"

    def test_hr_bpm_within_physiological_range(self):
        """HR estimate from peak count must fall in [40, 200] for 72-BPM signal."""
        raw = self._make_realistic_ppg()
        filt = _make_bandpass(raw)
        peaks = _make_peaks(filt)
        feats = backend_sqi(raw, filt, peaks, FS)
        assert 40 <= feats["hr_bpm"] <= 200, f"HR out of range: {feats['hr_bpm']}"

    def test_pure_noise_rated_poor(self):
        """White noise with no HR signal should be rated 'poor' or 'fair'."""
        rng = np.random.default_rng(5)
        raw = rng.standard_normal(N) * 100.0 + DC
        filt = _make_bandpass(raw)
        quality = backend_asg(raw, filt, [], FS)
        assert quality in ("poor", "fair"), \
            f"Noise signal with no peaks should not be 'good', got '{quality}'"


# ---------------------------------------------------------------------------
# TC7: Determinism — identical output across repeated calls
# ---------------------------------------------------------------------------
class TestDeterminism:

    def test_same_input_three_calls_identical(self):
        """compute_sqi_features must return identical floats on repeated calls."""
        raw = _sine_signal(freq_hz=1.1)
        filt = _make_bandpass(raw)
        peaks = _make_peaks(filt)
        results = [backend_sqi(raw, filt, peaks, FS) for _ in range(3)]
        for key in results[0]:
            vals = [r[key] for r in results]
            assert vals[0] == vals[1] == vals[2], \
                f"Non-deterministic output on '{key}': {vals}"

    def test_determinism_with_noisy_signal(self):
        """Noisy-signal results are stable across 3 calls."""
        rng = np.random.default_rng(99)
        raw = _sine_signal() + rng.standard_normal(N) * 300
        filt = _make_bandpass(raw)
        peaks = _make_peaks(filt)
        r1 = backend_sqi(raw, filt, peaks, FS)
        r2 = backend_sqi(raw, filt, peaks, FS)
        for key in r1:
            assert r1[key] == r2[key], f"Non-determinism on '{key}'"


# ---------------------------------------------------------------------------
# TC8: assess_signal_quality return-value contract
# ---------------------------------------------------------------------------
class TestAssessSignalQualityContract:

    def test_returns_string_only(self):
        raw = _sine_signal()
        filt = _make_bandpass(raw)
        result = backend_asg(raw, filt, _make_peaks(filt), FS)
        assert isinstance(result, str)

    def test_return_value_in_allowed_set(self):
        """Return value must be exactly one of 'good', 'fair', 'poor'."""
        for freq in [0.8, 1.2, 2.0]:
            raw = _sine_signal(freq_hz=freq)
            filt = _make_bandpass(raw)
            result = backend_asg(raw, filt, _make_peaks(filt), FS)
            assert result in ("good", "fair", "poor"), \
                f"Unexpected quality label '{result}' for freq={freq}"

    def test_flat_dc_only_signal_is_poor(self):
        """A flat DC-only signal (no AC) has amplitude_ratio ~ 0 → 'poor'."""
        raw = np.ones(N) * DC
        filt = np.zeros(N)   # bandpass of flat = zero
        result = backend_asg(raw, filt, [], FS)
        assert result == "poor", f"DC-only signal should be 'poor', got '{result}'"

    def test_high_freq_noise_no_hr_peaks_fair_or_poor(self):
        """High-frequency noise has no detectable HR peaks → score <= 2."""
        rng = np.random.default_rng(21)
        raw = rng.standard_normal(N) * 500.0 + DC
        # Inject a large high-freq component (10 Hz) that is outside HR band
        t = np.arange(N) / FS
        filt = 5000.0 * np.sin(2 * math.pi * 10.0 * t)
        result = backend_asg(raw, filt, [], FS)
        assert result in ("fair", "poor"), \
            f"High-freq noise with no peaks should not be 'good', got '{result}'"


# ---------------------------------------------------------------------------
# TC9: Performance benchmark
# ---------------------------------------------------------------------------
class TestPerformance:

    def test_latency_under_5ms_per_call(self):
        """Mean latency for 100 iterations on 500-sample signal must be < 5 ms."""
        raw = _sine_signal()
        filt = _make_bandpass(raw)
        peaks = _make_peaks(filt)
        # Warm up
        backend_sqi(raw, filt, peaks, FS)

        iterations = 100
        t0 = time.perf_counter()
        for _ in range(iterations):
            backend_sqi(raw, filt, peaks, FS)
        t1 = time.perf_counter()

        mean_ms = (t1 - t0) / iterations * 1000.0
        print(f"\nMean latency: {mean_ms:.3f} ms over {iterations} iterations")
        assert mean_ms < 5.0, f"compute_sqi_features too slow: {mean_ms:.3f} ms (limit 5 ms)"
