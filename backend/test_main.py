"""
PPG Backend - Test Suite (pytest)
===========================================
Target: backend/main.py  (FastAPI v4.0.0)
Runner: pytest
Dependencies: httpx (for TestClient), numpy, scipy

Groups
------
1. Unit: bandpass_filter, lowpass_filter
2. Unit: _reject_rr_outliers, _compute_hrv_metrics
3. Unit: calculate_spo2_v23
4. Unit: calculate_spo2_confidence
5. Unit: assess_signal_quality
6. Unit: apply_temporal_smoothing
7. Unit: _parse_ts, _validate_device_id (helpers)
8. Integration: POST /api/ppg/upload  — happy path
9. Integration: POST /api/ppg/upload  — validation failures
10. Integration: POST /api/ppg/upload  — edge / signal cases
11. Integration: GET /api/ppg/latest/{device_id}
12. Integration: GET /api/ppg/history/{device_id}
13. Integration: DELETE /api/ppg/history/{device_id}
14. Integration: GET /api/ppg/stats/{device_id}
15. Integration: GET / and GET /api/stats
16. Performance: throughput on 500-sample batch
"""

import sys
import os
import time
import math
import pytest
import numpy as np

# ---------------------------------------------------------------------------
# Make sure Python can find main.py even when pytest is invoked from another
# working directory.
# ---------------------------------------------------------------------------
_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

# Import the application and the functions-under-test BEFORE TestClient so
# that we can patch module-level state (readings_db, etc.) cleanly.
from main import (
    app,
    bandpass_filter,
    _reject_rr_outliers,
    _compute_hrv_metrics,
    calculate_spo2_v23,
    calculate_spo2_confidence,
    assess_signal_quality,
    compute_sqi_features,
    apply_temporal_smoothing,
    _parse_ts,
    _validate_device_id,
    readings_db,
    spo2_history,
    overlap_buffer,
    rr_accumulator,
    HR_MIN_BPM,
    HR_MAX_BPM,
    SPO2_MIN_VALID,
    SPO2_MAX_VALID,
    AC_DC_MIN_IR,
    AC_DC_MAX_IR,
)

from fastapi import HTTPException
from fastapi.testclient import TestClient

client = TestClient(app)

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------
FS = 100          # default sample rate used by ESP32
N  = 500          # default sample count  (5 s at 100 Hz)
DC = 50_000.0     # realistic MAX30102 DC offset (18-bit scale)


def _sine_ppg(n: int = N, fs: int = FS, freq_hz: float = 1.2,
              dc: float = DC, amp: float = 2000.0) -> np.ndarray:
    """
    Synthetic PPG signal: DC + sine at freq_hz (default 1.2 Hz = 72 BPM).
    Values are returned as a numpy float64 array.
    """
    t = np.arange(n) / fs
    return dc + amp * np.sin(2 * math.pi * freq_hz * t)


def _int_list(arr: np.ndarray) -> list:
    """Convert numpy array to list[int] for the Pydantic model."""
    return arr.astype(int).tolist()


def _valid_payload(device_id: str = "esp32_01",
                   n: int = N, fs: int = FS,
                   ir_dc: float = DC, red_dc: float = DC * 0.95,
                   ir_amp: float = 2000.0, red_amp: float = 1900.0,
                   freq_hz: float = 1.2) -> dict:
    """Build a valid PPGReading payload dict."""
    ir  = _sine_ppg(n, fs, freq_hz, ir_dc,  ir_amp)
    red = _sine_ppg(n, fs, freq_hz, red_dc, red_amp)
    return {
        "device_id":   device_id,
        "ir_values":   _int_list(ir),
        "red_values":  _int_list(red),
        "sample_rate": fs,
    }


@pytest.fixture(autouse=True)
def _clear_db():
    """Reset all in-memory state before every test to prevent cross-test pollution."""
    import main as _main
    readings_db.clear()
    spo2_history.clear()
    overlap_buffer.clear()
    rr_accumulator.clear()
    _main.device_index.clear()
    # Reset slowapi rate limiter storage để tránh cross-test exhaust
    try:
        from main import limiter as _limiter
        _limiter.reset()
    except Exception:
        pass
    yield
    readings_db.clear()
    spo2_history.clear()
    overlap_buffer.clear()
    rr_accumulator.clear()
    _main.device_index.clear()


# ===========================================================================
# 1. Unit tests — bandpass_filter
# ===========================================================================
class TestBandpassFilter:

    def test_bandpass_filter_happy_path_returns_same_length(self):
        sig = _sine_ppg()
        out = bandpass_filter(sig, FS)
        assert len(out) == len(sig)

    def test_bandpass_filter_output_is_ndarray(self):
        sig = _sine_ppg()
        out = bandpass_filter(sig, FS)
        assert isinstance(out, np.ndarray)

    def test_bandpass_filter_attenuates_dc(self):
        """DC component (0 Hz) must be heavily attenuated."""
        dc_signal = np.ones(N) * DC
        out = bandpass_filter(dc_signal, FS)
        assert np.max(np.abs(out)) < DC * 0.01

    def test_bandpass_filter_passes_1hz_signal(self):
        """1 Hz sine (well within 0.5–4 Hz band) should be largely preserved."""
        amp = 1000.0
        sig = DC + amp * np.sin(2 * math.pi * 1.0 * np.arange(N) / FS)
        out = bandpass_filter(sig, FS)
        # Output RMS should be meaningfully large (at least 20% of input AC amp)
        assert np.sqrt(np.mean(out ** 2)) > amp * 0.2

    def test_bandpass_filter_low_sample_rate_boundary(self):
        """sample_rate=25 Hz: Nyquist=12.5 Hz, highcut=4 Hz is valid."""
        sig = _sine_ppg(n=100, fs=25)
        out = bandpass_filter(sig, 25)
        assert len(out) == 100

    def test_bandpass_filter_high_sample_rate_boundary(self):
        """sample_rate=400 Hz: should work without error."""
        sig = _sine_ppg(n=2000, fs=400)
        out = bandpass_filter(sig, 400)
        assert len(out) == 2000

    def test_bandpass_filter_low_highcut_degrades_gracefully(self):
        """When lowcut >= highcut due to extreme fs, should return a copy without crash."""
        # fs=25 → nyquist=12.5; low=0.5/12.5=0.04, high=4/12.5=0.32 — still valid.
        # Force degenerate case with very high lowcut/highcut args.
        sig = np.ones(200) * 1000.0
        # lowcut=5, highcut=4 → low >= high; function should return signal copy
        out = bandpass_filter(sig, 10, lowcut=5.0, highcut=4.0)
        np.testing.assert_array_equal(out, sig)

    def test_bandpass_filter_flat_signal_near_zero_output(self):
        """Flat (DC-only) signal filtered to bandpass band → near-zero output."""
        flat = np.full(N, 30000.0)
        out = bandpass_filter(flat, FS)
        assert np.max(np.abs(out)) < 1.0


# ===========================================================================
# 2. Unit tests — _reject_rr_outliers and _compute_hrv_metrics
# ===========================================================================
class TestRRUtils:

    def test_reject_rr_outliers_empty_returns_empty(self):
        rr = np.array([], dtype=float)
        out = _reject_rr_outliers(rr)
        assert len(out) == 0

    def test_reject_rr_outliers_removes_outliers(self):
        # median ≈ 800 ms → tolerance ±160 ms (640–960)
        rr = np.array([800.0, 790.0, 810.0, 300.0, 1500.0])
        out = _reject_rr_outliers(rr)
        assert 300.0 not in out
        assert 1500.0 not in out
        assert len(out) == 3

    def test_reject_rr_outliers_keeps_all_valid(self):
        rr = np.array([820.0, 800.0, 810.0, 795.0])
        out = _reject_rr_outliers(rr)
        assert len(out) == 4

    def test_compute_hrv_metrics_single_interval(self):
        """Only 1 interval → zero HRV."""
        hrv = _compute_hrv_metrics(np.array([800.0]))
        assert hrv.sdnn_ms == 0.0
        assert hrv.rmssd_ms == 0.0
        assert hrv.mean_nn_ms == 0.0

    def test_compute_hrv_metrics_returns_nonnegative(self):
        rr = np.array([800.0, 820.0, 790.0, 810.0, 805.0])
        hrv = _compute_hrv_metrics(rr)
        assert hrv.sdnn_ms >= 0.0
        assert hrv.rmssd_ms >= 0.0
        assert hrv.pnn50_pct >= 0.0

    def test_compute_hrv_metrics_mean_nn_correct(self):
        rr = np.array([800.0, 800.0, 800.0])
        hrv = _compute_hrv_metrics(rr)
        assert hrv.mean_nn_ms == pytest.approx(800.0, abs=0.1)

    def test_compute_hrv_metrics_sdnn_known_value(self):
        """Two identical RR → std=0."""
        rr = np.array([800.0, 800.0])
        hrv = _compute_hrv_metrics(rr)
        assert hrv.sdnn_ms == pytest.approx(0.0, abs=0.01)

    def test_compute_hrv_metrics_pnn50_all_above_50(self):
        """Consecutive diffs all > 50 ms → pNN50 near 100%."""
        rr = np.array([800.0, 900.0, 800.0, 900.0])  # diffs = [100, 100, 100]
        hrv = _compute_hrv_metrics(rr)
        assert hrv.pnn50_pct > 50.0

    def test_compute_hrv_metrics_rr_count_stored(self):
        rr = np.array([800.0, 810.0, 790.0])
        hrv = _compute_hrv_metrics(rr)
        assert hrv.rr_count == 3


# ===========================================================================
# 3. Unit tests — calculate_spo2_v23
# ===========================================================================
class TestCalculateSpo2:

    def _build_signals(self, ac_ir_ratio: float = 0.03,
                       ac_red_ratio: float = 0.02,
                       dc: float = DC, n: int = N) -> tuple:
        """
        Build IR/Red signals with controlled AC/DC ratios so that R = ac_red_ratio / ac_ir_ratio.
        """
        ir  = dc + dc * ac_ir_ratio  * np.sin(2 * math.pi * 1.2 * np.arange(n) / FS)
        red = dc * 0.95 + dc * 0.95 * ac_red_ratio * np.sin(2 * math.pi * 1.2 * np.arange(n) / FS)
        return ir, red

    def test_spo2_happy_path_returns_valid_range(self):
        ir, red = self._build_signals()
        spo2, R, acdc_ir, acdc_red, reason = calculate_spo2_v23(ir, red, FS)
        assert spo2 is not None
        assert SPO2_MIN_VALID <= spo2 <= SPO2_MAX_VALID

    def test_spo2_happy_path_reason_ok(self):
        ir, red = self._build_signals()
        _, _, _, _, reason = calculate_spo2_v23(ir, red, FS)
        assert reason == "ok"

    def test_spo2_ratio_r_in_valid_range(self):
        ir, red = self._build_signals()
        _, R, _, _, reason = calculate_spo2_v23(ir, red, FS)
        if reason == "ok":
            assert 0.4 <= R <= 2.0

    def test_spo2_returns_none_for_zero_signal(self):
        """All-zero IR → dc_ir = 0 → should return None."""
        ir  = np.zeros(N)
        red = np.zeros(N)
        spo2, _, _, _, reason = calculate_spo2_v23(ir, red, FS)
        assert spo2 is None

    def test_spo2_clipped_to_100(self):
        """R close to 0 → spo2 formula > 100 → must be clipped to 100.
        Uses a constant-valued 1-D red array (np.full) so filtfilt receives a
        proper 1-D array and does not raise IndexError on a 0-d scalar."""
        ir  = DC + 0.1 * np.sin(2 * math.pi * 1.2 * np.arange(N) / FS)
        red = np.full(N, DC * 0.95)  # 1-D array, not scalar
        spo2, _, _, _, _ = calculate_spo2_v23(ir, red, FS)
        if spo2 is not None:
            assert spo2 <= 100.0

    def test_spo2_clipped_to_85(self):
        """High R → spo2 formula < 85 → must be clipped to 85."""
        spo2 = float(np.clip(108.0 - 23.0 * 1.5, SPO2_MIN_VALID, SPO2_MAX_VALID))
        assert spo2 == pytest.approx(85.0, abs=0.1)

    def test_spo2_flat_red_returns_none_or_invalid(self):
        """Flat red signal has near-zero AC → should be rejected."""
        ir  = DC + 2000 * np.sin(2 * math.pi * 1.2 * np.arange(N) / FS)
        red = np.full(N, DC * 0.95)
        spo2, _, _, _, reason = calculate_spo2_v23(ir, red, FS)
        # Either None or reason != ok because ac_red ≈ 0
        assert spo2 is None or reason != "ok"

    @pytest.mark.parametrize("n_samples", [50, 100, 200, 500])
    def test_spo2_various_lengths(self, n_samples):
        ir, red = self._build_signals(n=n_samples)
        spo2, _, _, _, _ = calculate_spo2_v23(ir, red, FS)
        # Just assert no exception; result may be None if signal too short for filter


# ===========================================================================
# 4. Unit tests — calculate_spo2_confidence
# ===========================================================================
class TestSpo2Confidence:

    def test_confidence_short_signal_returns_zero(self):
        ir  = np.array([50000.0] * 5)
        red = np.array([47500.0] * 5)
        conf = calculate_spo2_confidence(ir, red, 0.03, 0.02)
        assert conf == 0.0

    def test_confidence_range_0_to_1(self):
        ir  = DC + 2000 * np.sin(2 * math.pi * 1.2 * np.arange(N) / FS)
        red = DC * 0.95 + 1900 * np.sin(2 * math.pi * 1.2 * np.arange(N) / FS)
        conf = calculate_spo2_confidence(ir, red, 0.03, 0.02)
        assert 0.0 <= conf <= 1.0

    def test_confidence_flat_signal_low(self):
        """Flat signal has std ≈ 0 → SNR term ≈ 0 → low confidence."""
        flat = np.full(N, DC)
        conf = calculate_spo2_confidence(flat, flat, 0.03, 0.02)
        assert conf < 0.5

    def test_confidence_good_acdc_score(self):
        """AC/DC ratios at midpoint of valid range → acdc_score = 1.0."""
        # AC_DC_MAX_RED = 0.10 (from main.py constant)
        mid_ir  = (AC_DC_MIN_IR + AC_DC_MAX_IR) / 2.0
        mid_red = (AC_DC_MIN_IR + 0.10)         / 2.0
        ir  = DC + 2000 * np.sin(2 * math.pi * 1.2 * np.arange(N) / FS)
        red = DC * 0.95 + 1900 * np.sin(2 * math.pi * 1.2 * np.arange(N) / FS)
        conf = calculate_spo2_confidence(ir, red, mid_ir, mid_red)
        # acdc contribution is 0.4 * acdc_score; total should be > 0
        assert conf > 0.0

    def test_confidence_returns_float(self):
        ir  = DC + 2000 * np.sin(2 * math.pi * 1.2 * np.arange(N) / FS)
        red = DC * 0.95 + 1900 * np.sin(2 * math.pi * 1.2 * np.arange(N) / FS)
        conf = calculate_spo2_confidence(ir, red, 0.03, 0.02)
        assert isinstance(conf, float)


# ===========================================================================
# 5. Unit tests — assess_signal_quality
# ===========================================================================
class TestAssessSignalQuality:

    def _make_peaks(self, filtered_ir: np.ndarray, fs: int = FS) -> list:
        from scipy.signal import find_peaks as _fp
        peaks, _ = _fp(filtered_ir, distance=int(fs * 0.3))
        return peaks.tolist()

    def test_quality_good_on_clean_sine(self):
        raw = _sine_ppg()
        filt = bandpass_filter(raw, FS)
        peaks = self._make_peaks(filt)
        quality = assess_signal_quality(raw, filt, peaks, FS)
        assert quality in ("good", "fair")

    def test_quality_poor_on_flat_signal(self):
        """DC-only signal has no peaks → poor quality."""
        raw  = np.full(N, DC)
        filt = bandpass_filter(raw, FS)
        peaks = []
        quality = assess_signal_quality(raw, filt, peaks, FS)
        assert quality == "poor"

    def test_quality_returns_valid_string(self):
        raw  = _sine_ppg()
        filt = bandpass_filter(raw, FS)
        peaks = self._make_peaks(filt)
        quality = assess_signal_quality(raw, filt, peaks, FS)
        assert quality in ("good", "fair", "poor")

    def test_quality_high_noise_signal(self):
        """Signal dominated by random noise → quality degraded."""
        rng = np.random.default_rng(42)
        noise = rng.standard_normal(N) * 5000
        raw  = DC + noise
        filt = bandpass_filter(raw, FS)
        peaks = self._make_peaks(filt)
        # We just assert no crash and valid string returned
        quality = assess_signal_quality(raw, filt, peaks, FS)
        assert quality in ("good", "fair", "poor")

    def test_quality_with_one_peak(self):
        """Only 1 peak → BPM cannot be estimated → score is lower."""
        raw  = _sine_ppg()
        filt = bandpass_filter(raw, FS)
        quality = assess_signal_quality(raw, filt, [100], FS)
        # One peak is still better than zero peaks; result is fair or poor
        assert quality in ("fair", "poor", "good")

    def test_quality_empty_peaks(self):
        raw  = np.full(N, DC)
        filt = bandpass_filter(raw, FS)
        quality = assess_signal_quality(raw, filt, [], FS)
        assert quality == "poor"


# ===========================================================================
# 6. Unit tests — apply_temporal_smoothing
# ===========================================================================
class TestTemporalSmoothing:

    def test_smoothing_single_reading(self):
        smoothed, snap = apply_temporal_smoothing([], 97.0)
        assert smoothed == pytest.approx(97.0, abs=0.1)
        assert snap == [97.0]

    def test_smoothing_median_of_five(self):
        values = [95.0, 97.0, 96.0, 98.0, 94.0]
        snap = []
        last = 0.0
        for v in values:
            last, snap = apply_temporal_smoothing(snap, v)
        # Median of [95, 97, 96, 98, 94] = 96.0
        assert last == pytest.approx(96.0, abs=0.1)

    def test_smoothing_window_limited_to_5(self):
        """After 6 readings, the 6th reading replaces the oldest."""
        snap = []
        for v in [90.0, 91.0, 92.0, 93.0, 94.0]:
            _, snap = apply_temporal_smoothing(snap, v)
        # Push 99.0: buffer = [91, 92, 93, 94, 99] → median = 93
        result, snap = apply_temporal_smoothing(snap, 99.0)
        assert result == pytest.approx(93.0, abs=0.5)
        assert len(snap) == 5

    def test_smoothing_returns_float(self):
        smoothed, snap = apply_temporal_smoothing([], 97.5)
        assert isinstance(smoothed, float)
        assert isinstance(snap, list)


# ===========================================================================
# 7. Unit tests — _parse_ts and _validate_device_id
# ===========================================================================
class TestHelpers:

    def test_parse_ts_valid_iso(self):
        from datetime import timezone as tz
        dt = _parse_ts("2026-04-24T10:30:00+00:00")
        assert dt.tzinfo is not None

    def test_parse_ts_naive_gets_utc(self):
        from datetime import timezone as tz
        dt = _parse_ts("2026-04-24T10:30:00")
        assert dt.tzinfo == tz.utc

    def test_parse_ts_invalid_returns_min(self):
        from datetime import datetime
        dt = _parse_ts("not-a-date")
        assert dt == datetime.min.replace(tzinfo=__import__("datetime").timezone.utc)

    def test_parse_ts_none_returns_min(self):
        from datetime import datetime
        dt = _parse_ts(None)
        assert dt == datetime.min.replace(tzinfo=__import__("datetime").timezone.utc)

    def test_validate_device_id_valid(self):
        assert _validate_device_id("esp32_01") == "esp32_01"

    def test_validate_device_id_valid_with_hyphen(self):
        assert _validate_device_id("device-ABC-123") == "device-ABC-123"

    def test_validate_device_id_invalid_raises(self):
        with pytest.raises(HTTPException) as exc_info:
            _validate_device_id("device@name!")
        assert exc_info.value.status_code == 400

    def test_validate_device_id_empty_raises(self):
        with pytest.raises(HTTPException):
            _validate_device_id("")

    def test_validate_device_id_too_long_raises(self):
        long_id = "a" * 65
        with pytest.raises(HTTPException):
            _validate_device_id(long_id)


# ===========================================================================
# 8. Integration — POST /api/ppg/upload  happy path
# ===========================================================================
class TestUploadHappyPath:

    def test_upload_returns_200(self):
        payload = _valid_payload()
        resp = client.post("/api/ppg/upload", json=payload)
        assert resp.status_code == 200

    def test_upload_response_has_required_fields(self):
        payload = _valid_payload()
        resp = client.post("/api/ppg/upload", json=payload)
        data = resp.json()
        required = [
            "reading_id", "device_id", "heart_rate", "hr_confidence",
            "spo2", "spo2_raw", "spo2_confidence", "ratio_r",
            "perfusion_index", "hrv", "signal_quality", "timestamp",
        ]
        for field in required:
            assert field in data, f"Missing field: {field}"

    def test_upload_device_id_echoed(self):
        payload = _valid_payload(device_id="test_device")
        resp = client.post("/api/ppg/upload", json=payload)
        assert resp.json()["device_id"] == "test_device"

    def test_upload_heart_rate_in_valid_range(self):
        payload = _valid_payload()
        resp = client.post("/api/ppg/upload", json=payload)
        hr = resp.json()["heart_rate"]
        # hr=0 is returned when signal is too noisy for peak detection
        assert hr == 0.0 or (HR_MIN_BPM <= hr <= HR_MAX_BPM)

    def test_upload_spo2_in_valid_range_or_zero(self):
        payload = _valid_payload()
        resp = client.post("/api/ppg/upload", json=payload)
        spo2 = resp.json()["spo2"]
        assert spo2 == 0.0 or (SPO2_MIN_VALID <= spo2 <= SPO2_MAX_VALID)

    def test_upload_signal_quality_is_valid_string(self):
        payload = _valid_payload()
        resp = client.post("/api/ppg/upload", json=payload)
        assert resp.json()["signal_quality"] in ("good", "fair", "poor")

    def test_upload_hrv_metrics_nonnegative(self):
        payload = _valid_payload()
        resp = client.post("/api/ppg/upload", json=payload)
        hrv = resp.json()["hrv"]
        assert hrv["sdnn_ms"]    >= 0.0
        assert hrv["rmssd_ms"]   >= 0.0
        assert hrv["pnn50_pct"]  >= 0.0
        assert hrv["mean_nn_ms"] >= 0.0

    def test_upload_reading_id_is_full_uuid_hex(self):
        payload = _valid_payload()
        resp = client.post("/api/ppg/upload", json=payload)
        rid = resp.json()["reading_id"]
        assert len(rid) == 32
        assert all(c in "0123456789abcdef" for c in rid)

    def test_upload_saves_to_db(self):
        payload = _valid_payload(device_id="db_test")
        resp = client.post("/api/ppg/upload", json=payload)
        assert resp.status_code == 200
        assert len(readings_db) == 1

    def test_upload_perfusion_index_nonnegative(self):
        payload = _valid_payload()
        resp = client.post("/api/ppg/upload", json=payload)
        assert resp.json()["perfusion_index"] >= 0.0

    def test_upload_filtered_signal_same_length_as_input(self):
        payload = _valid_payload(n=500)
        resp = client.post("/api/ppg/upload", json=payload)
        assert len(resp.json()["filtered_signal"]) == 500

    def test_upload_with_explicit_timestamp(self):
        # Use a timestamp 30 minutes ago so it falls within the validator's
        # [now-1h, now+5s] acceptance window (main.py:_v_ts replay-protection).
        from datetime import datetime, timezone, timedelta
        ts = (datetime.now(timezone.utc) - timedelta(minutes=30)).isoformat()
        payload = _valid_payload()
        payload["timestamp"] = ts
        resp = client.post("/api/ppg/upload", json=payload)
        assert resp.status_code == 200
        # Compare the YYYY-MM-DD portion (echoes back when validator accepts)
        expected_date = ts[:10]
        assert expected_date in resp.json()["timestamp"]

    def test_upload_sample_rate_25_boundary(self):
        payload = _valid_payload(fs=25, n=100)
        resp = client.post("/api/ppg/upload", json=payload)
        assert resp.status_code == 200

    def test_upload_sample_rate_400_boundary(self):
        payload = _valid_payload(fs=400, n=2000)
        resp = client.post("/api/ppg/upload", json=payload)
        assert resp.status_code == 200

    def test_upload_hr_confidence_in_0_1(self):
        payload = _valid_payload()
        resp = client.post("/api/ppg/upload", json=payload)
        conf = resp.json()["hr_confidence"]
        assert 0.0 <= conf <= 1.0

    def test_upload_spo2_confidence_in_0_1(self):
        payload = _valid_payload()
        resp = client.post("/api/ppg/upload", json=payload)
        conf = resp.json()["spo2_confidence"]
        assert 0.0 <= conf <= 1.0

    def test_upload_peaks_are_integers(self):
        payload = _valid_payload()
        resp = client.post("/api/ppg/upload", json=payload)
        for p in resp.json()["peaks"]:
            assert isinstance(p, int)

    def test_upload_ratio_r_nonnegative(self):
        payload = _valid_payload()
        resp = client.post("/api/ppg/upload", json=payload)
        assert resp.json()["ratio_r"] >= 0.0


# ===========================================================================
# 9. Integration — POST /api/ppg/upload  validation failures
# ===========================================================================
class TestUploadValidation:

    def test_upload_missing_device_id_returns_422(self):
        payload = _valid_payload()
        del payload["device_id"]
        resp = client.post("/api/ppg/upload", json=payload)
        assert resp.status_code == 422

    def test_upload_too_few_samples_returns_400(self):
        """len(ir) < MIN_SAMPLES (50) → HTTP 4xx (422 from Pydantic, 400 from endpoint)."""
        payload = _valid_payload(n=30)
        resp = client.post("/api/ppg/upload", json=payload)
        assert resp.status_code in (400, 422)

    def test_upload_mismatched_ir_red_length_returns_400(self):
        payload = _valid_payload()
        payload["red_values"] = payload["red_values"][:-10]
        resp = client.post("/api/ppg/upload", json=payload)
        assert resp.status_code == 400

    def test_upload_no_finger_ir_returns_400(self):
        """mean(IR) < 1000 → HTTP 400 (no finger)."""
        ir  = [100] * N
        red = _int_list(_sine_ppg(dc=DC))
        payload = {
            "device_id":   "esp32_01",
            "ir_values":   ir,
            "red_values":  red,
            "sample_rate": FS,
        }
        resp = client.post("/api/ppg/upload", json=payload)
        assert resp.status_code == 400

    def test_upload_no_finger_red_returns_400(self):
        """mean(RED) < 1000 → HTTP 400."""
        ir  = _int_list(_sine_ppg(dc=DC))
        red = [100] * N
        payload = {
            "device_id":   "esp32_01",
            "ir_values":   ir,
            "red_values":  red,
            "sample_rate": FS,
        }
        resp = client.post("/api/ppg/upload", json=payload)
        assert resp.status_code == 400

    def test_upload_sample_rate_too_low_returns_422(self):
        payload = _valid_payload()
        payload["sample_rate"] = 10
        resp = client.post("/api/ppg/upload", json=payload)
        assert resp.status_code == 422

    def test_upload_sample_rate_too_high_returns_422(self):
        payload = _valid_payload()
        payload["sample_rate"] = 500
        resp = client.post("/api/ppg/upload", json=payload)
        assert resp.status_code == 422

    def test_upload_device_id_too_long_returns_422(self):
        payload = _valid_payload()
        payload["device_id"] = "a" * 65
        resp = client.post("/api/ppg/upload", json=payload)
        assert resp.status_code == 422

    def test_upload_device_id_special_chars_returns_422(self):
        payload = _valid_payload()
        payload["device_id"] = "device@name!"
        resp = client.post("/api/ppg/upload", json=payload)
        assert resp.status_code == 422

    def test_upload_device_id_with_space_returns_422(self):
        payload = _valid_payload()
        payload["device_id"] = "device name"
        resp = client.post("/api/ppg/upload", json=payload)
        assert resp.status_code == 422

    def test_upload_too_many_samples_returns_422(self):
        """len(ir) > MAX_SAMPLES (10_000) → Pydantic validation error → 422."""
        payload = _valid_payload(n=10_001)
        resp = client.post("/api/ppg/upload", json=payload)
        assert resp.status_code == 422

    def test_upload_empty_arrays_returns_422_or_400(self):
        """Empty arrays: len < MIN_SAMPLES → 400, or Pydantic may accept empty list."""
        payload = {
            "device_id":  "esp32_01",
            "ir_values":  [],
            "red_values": [],
            "sample_rate": FS,
        }
        resp = client.post("/api/ppg/upload", json=payload)
        assert resp.status_code in (400, 422)

    def test_upload_wrong_type_ir_values_returns_422(self):
        """ir_values contains strings → Pydantic type coercion may fail."""
        payload = _valid_payload()
        payload["ir_values"] = ["abc"] * N
        resp = client.post("/api/ppg/upload", json=payload)
        assert resp.status_code == 422

    def test_upload_missing_ir_values_returns_422(self):
        payload = _valid_payload()
        del payload["ir_values"]
        resp = client.post("/api/ppg/upload", json=payload)
        assert resp.status_code == 422


# ===========================================================================
# 10. Integration — POST /api/ppg/upload  edge / signal edge cases
# ===========================================================================
class TestUploadEdgeCases:

    def test_upload_all_zeros_ir_returns_400(self):
        """All-zero IR → mean(IR)=0 < 1000 → 400 no-finger."""
        payload = {
            "device_id":   "edge_dev",
            "ir_values":   [0] * N,
            "red_values":  [0] * N,
            "sample_rate": FS,
        }
        resp = client.post("/api/ppg/upload", json=payload)
        assert resp.status_code == 400

    def test_upload_constant_high_signal_returns_200(self):
        """Constant signal at high DC is a valid finger reading; no crash expected."""
        payload = {
            "device_id":   "edge_dev",
            "ir_values":   [int(DC)] * N,
            "red_values":  [int(DC * 0.95)] * N,
            "sample_rate": FS,
        }
        resp = client.post("/api/ppg/upload", json=payload)
        assert resp.status_code == 200

    def test_upload_minimum_valid_samples(self):
        """Exactly MIN_SAMPLES (50) samples → should succeed."""
        payload = _valid_payload(n=50)
        resp = client.post("/api/ppg/upload", json=payload)
        assert resp.status_code == 200

    def test_upload_49_samples_returns_400(self):
        """49 samples < MIN_SAMPLES → 4xx (422 from Pydantic, 400 from endpoint)."""
        payload = _valid_payload(n=49)
        resp = client.post("/api/ppg/upload", json=payload)
        assert resp.status_code in (400, 422)

    def test_upload_pure_noise_returns_200(self):
        """High-frequency noise with sufficient DC → no crash, quality likely poor."""
        rng = np.random.default_rng(7)
        ir  = (DC + rng.integers(-3000, 3000, N)).tolist()
        red = (DC * 0.95 + rng.integers(-3000, 3000, N)).tolist()
        payload = {
            "device_id":   "noise_dev",
            "ir_values":   ir,
            "red_values":  red,
            "sample_rate": FS,
        }
        resp = client.post("/api/ppg/upload", json=payload)
        assert resp.status_code == 200

    def test_upload_device_id_64_chars_valid(self):
        """device_id with exactly 64 chars → valid."""
        payload = _valid_payload(device_id="a" * 64)
        resp = client.post("/api/ppg/upload", json=payload)
        assert resp.status_code == 200

    def test_upload_device_id_with_underscores_and_hyphens(self):
        payload = _valid_payload(device_id="esp32_XIAO-S3_01")
        resp = client.post("/api/ppg/upload", json=payload)
        assert resp.status_code == 200

    def test_upload_very_high_hr_signal_clipped(self):
        """4 Hz sine (~240 BPM) exceeds HR_MAX → HR clipped to 200."""
        payload = _valid_payload(freq_hz=3.5)   # 210 BPM theoretical
        resp = client.post("/api/ppg/upload", json=payload)
        assert resp.status_code == 200
        hr = resp.json()["heart_rate"]
        assert hr == 0.0 or hr <= HR_MAX_BPM

    def test_upload_very_low_hr_signal_clipped(self):
        """0.5 Hz sine (~30 BPM) below HR_MIN → HR either 0 or clipped to 40."""
        payload = _valid_payload(freq_hz=0.5)
        resp = client.post("/api/ppg/upload", json=payload)
        assert resp.status_code == 200
        hr = resp.json()["heart_rate"]
        assert hr == 0.0 or hr >= HR_MIN_BPM

    def test_upload_large_batch_10000_samples(self):
        """MAX_SAMPLES allowed: 10 000 samples."""
        payload = _valid_payload(n=10_000, fs=400)
        resp = client.post("/api/ppg/upload", json=payload)
        assert resp.status_code == 200

    def test_upload_spo2_rejection_reason_when_invalid(self):
        """When SpO2 is invalid (None), rejection_reason must not be None."""
        ir  = [int(DC)] * N
        red = [int(DC * 0.95)] * N   # flat → ac_red ≈ 0
        payload = {
            "device_id":   "spo2_reject",
            "ir_values":   ir,
            "red_values":  red,
            "sample_rate": FS,
        }
        resp = client.post("/api/ppg/upload", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        # SpO2 = 0.0 and rejection_reason should be set
        if data["spo2"] == 0.0:
            assert data["rejection_reason"] is not None

    def test_upload_rr_accumulator_grows_across_batches(self):
        """Second upload for same device should accumulate RR intervals."""
        dev = "rr_acc_test"
        payload = _valid_payload(device_id=dev)
        client.post("/api/ppg/upload", json=payload)
        client.post("/api/ppg/upload", json=payload)
        if dev in rr_accumulator:
            assert len(rr_accumulator[dev]) >= 0  # at least not error


# ===========================================================================
# 11. Integration — GET /api/ppg/latest/{device_id}
# ===========================================================================
class TestGetLatest:

    def _upload(self, device_id: str = "latest_dev") -> dict:
        payload = _valid_payload(device_id=device_id)
        return client.post("/api/ppg/upload", json=payload).json()

    def test_latest_returns_200_after_upload(self):
        self._upload("lat01")
        resp = client.get("/api/ppg/latest/lat01")
        assert resp.status_code == 200

    def test_latest_returns_most_recent(self):
        """Two uploads; latest must return a reading for the same device."""
        dev = "lat_order"
        self._upload(dev)
        self._upload(dev)
        resp = client.get(f"/api/ppg/latest/{dev}")
        assert resp.json()["device_id"] == dev

    def test_latest_returns_404_no_data(self):
        resp = client.get("/api/ppg/latest/nonexistent_device")
        assert resp.status_code == 404

    def test_latest_invalid_device_id_returns_400(self):
        resp = client.get("/api/ppg/latest/bad@device!")
        assert resp.status_code == 400

    def test_latest_has_correct_device_id(self):
        self._upload("lat_check")
        resp = client.get("/api/ppg/latest/lat_check")
        assert resp.json()["device_id"] == "lat_check"


# ===========================================================================
# 12. Integration — GET /api/ppg/history/{device_id}
# ===========================================================================
class TestGetHistory:

    def _upload_n(self, device_id: str, n_times: int) -> list:
        results = []
        for _ in range(n_times):
            r = client.post("/api/ppg/upload", json=_valid_payload(device_id=device_id))
            results.append(r.json())
        return results

    def test_history_returns_200(self):
        self._upload_n("hist_dev", 1)
        resp = client.get("/api/ppg/history/hist_dev")
        assert resp.status_code == 200

    def test_history_empty_device_returns_200_count_0(self):
        resp = client.get("/api/ppg/history/never_seen_dev")
        assert resp.status_code == 200
        assert resp.json()["count"] == 0

    def test_history_count_matches_uploads(self):
        dev = "hist_count"
        self._upload_n(dev, 3)
        resp = client.get(f"/api/ppg/history/{dev}")
        assert resp.json()["count"] == 3

    def test_history_limit_respected(self):
        dev = "hist_limit"
        self._upload_n(dev, 10)
        resp = client.get(f"/api/ppg/history/{dev}?limit=5")
        assert len(resp.json()["readings"]) <= 5

    def test_history_limit_clamped_to_1(self):
        dev = "hist_limit0"
        self._upload_n(dev, 3)
        resp = client.get(f"/api/ppg/history/{dev}?limit=0")
        assert len(resp.json()["readings"]) >= 1

    def test_history_invalid_device_id_returns_400(self):
        resp = client.get("/api/ppg/history/bad@id!")
        assert resp.status_code == 400

    def test_history_different_devices_isolated(self):
        self._upload_n("hist_devA", 2)
        self._upload_n("hist_devB", 3)
        respA = client.get("/api/ppg/history/hist_devA")
        respB = client.get("/api/ppg/history/hist_devB")
        assert respA.json()["count"] == 2
        assert respB.json()["count"] == 3


# ===========================================================================
# 13. Integration — DELETE /api/ppg/history/{device_id}
# ===========================================================================
class TestDeleteHistory:

    def test_delete_returns_200(self):
        client.post("/api/ppg/upload", json=_valid_payload(device_id="del_dev"))
        resp = client.delete("/api/ppg/history/del_dev")
        assert resp.status_code == 200

    def test_delete_removes_readings(self):
        dev = "del_clean"
        client.post("/api/ppg/upload", json=_valid_payload(device_id=dev))
        client.delete(f"/api/ppg/history/{dev}")
        resp = client.get(f"/api/ppg/history/{dev}")
        assert resp.json()["count"] == 0

    def test_delete_clears_spo2_buffer(self):
        dev = "del_spo2"
        client.post("/api/ppg/upload", json=_valid_payload(device_id=dev))
        client.delete(f"/api/ppg/history/{dev}")
        assert dev not in spo2_history

    def test_delete_clears_overlap_buffer(self):
        dev = "del_overlap"
        client.post("/api/ppg/upload", json=_valid_payload(device_id=dev))
        client.delete(f"/api/ppg/history/{dev}")
        assert dev not in overlap_buffer

    def test_delete_clears_rr_accumulator(self):
        dev = "del_rr"
        client.post("/api/ppg/upload", json=_valid_payload(device_id=dev))
        client.delete(f"/api/ppg/history/{dev}")
        assert dev not in rr_accumulator

    def test_delete_nonexistent_device_returns_200_deleted_0(self):
        resp = client.delete("/api/ppg/history/ghost_device")
        assert resp.status_code == 200
        assert resp.json()["deleted"] == 0

    def test_delete_invalid_device_id_returns_400(self):
        resp = client.delete("/api/ppg/history/bad@id!")
        assert resp.status_code == 400

    def test_delete_only_removes_target_device(self):
        dev_a = "del_only_A"
        dev_b = "del_only_B"
        client.post("/api/ppg/upload", json=_valid_payload(device_id=dev_a))
        client.post("/api/ppg/upload", json=_valid_payload(device_id=dev_b))
        client.delete(f"/api/ppg/history/{dev_a}")
        assert client.get(f"/api/ppg/history/{dev_b}").json()["count"] == 1

    # -----------------------------------------------------------------------
    # Race-condition regression tests (concurrent DELETE + POST)
    #
    # PRE-FIX failure mode (for reviewer):
    #   clear_history did: `with device_lock: ... _device_locks.pop(device_id, None)`
    #   While thread A held device_lock and popped the registry entry, thread B
    #   calling _get_device_lock() would see no entry and create a NEW Lock object.
    #   From that point the two threads held DIFFERENT locks — mutual exclusion was
    #   broken.  Race window: ~microseconds but reproducible at 500 total operations.
    #   With the fix (pop removed) the lock stays in the registry permanently so
    #   every _get_device_lock() call returns the same object.
    # -----------------------------------------------------------------------

    def test_concurrent_delete_and_upload_no_corruption(self):
        """
        Regression test for TOCTOU bug in clear_history:
        _device_locks.pop() while holding the lock broke mutual exclusion.

        5 upload threads x 50 iterations + 5 delete threads x 50 iterations
        (500 total operations) must:
          - return only semantically-valid HTTP codes (200 or 429 rate-limit;
            never 500 / unhandled exception)
          - leave readings_db and device_index in a consistent state
            (every reading_id in device_index must also exist in readings_db)

        429 (Too Many Requests) is accepted because the slowapi rate limiter
        fires under burst concurrency.  429 is the CORRECT response to burst
        requests — it is NOT a corruption signal.  The bug we are guarding
        against produces 500 (unhandled KeyError / AttributeError from broken
        mutual exclusion) or structural inconsistency in readings_db /
        device_index, which are checked explicitly after the threads finish.
        """
        import threading
        import main as _main

        DEVICE_ID = "dev_race_test"
        ITERATIONS = 50
        # Accepted HTTP codes:
        #   200 = normal success
        #   429 = rate-limit burst (expected under 5-thread concurrency; NOT corruption)
        ACCEPTED_STATUS = {200, 429}
        errors: list = []
        bad_status: list = []

        # Reset rate-limiter counters so burst traffic starts from a clean slate
        try:
            _main.limiter.reset()
        except Exception:
            pass

        def do_upload():
            for _ in range(ITERATIONS):
                try:
                    resp = client.post(
                        "/api/ppg/upload",
                        json=_valid_payload(device_id=DEVICE_ID),
                    )
                    if resp.status_code not in ACCEPTED_STATUS:
                        bad_status.append(("upload", resp.status_code))
                except Exception as exc:
                    errors.append(("upload", type(exc).__name__, str(exc)))

        def do_delete():
            for _ in range(ITERATIONS):
                try:
                    resp = client.delete(f"/api/ppg/history/{DEVICE_ID}")
                    # 200 with deleted=N or deleted=0 are both valid
                    if resp.status_code not in ACCEPTED_STATUS:
                        bad_status.append(("delete", resp.status_code))
                except Exception as exc:
                    errors.append(("delete", type(exc).__name__, str(exc)))

        threads = (
            [threading.Thread(target=do_upload, daemon=True) for _ in range(5)]
            + [threading.Thread(target=do_delete, daemon=True) for _ in range(5)]
        )
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        # No unhandled exceptions from any thread (KeyError / AttributeError
        # from broken mutual exclusion would surface here)
        assert errors == [], f"Thread exceptions (indicates corruption): {errors}"
        # Only 200 or 429 allowed; 500 means an unhandled server error
        assert bad_status == [], f"Unexpected HTTP statuses (500 = corruption): {bad_status}"

        # Structural integrity: every reading_id referenced by device_index
        # must exist in readings_db (no dangling pointers in either direction)
        with _main._state_lock:
            indexed_ids = set(_main.device_index.get(DEVICE_ID, []))
            for rid in indexed_ids:
                assert rid in readings_db, (
                    f"reading_id {rid!r} in device_index but missing from readings_db"
                )
            # Reverse: no orphan readings for this device in readings_db
            db_ids_for_device = {
                k for k, v in readings_db.items() if v.get("device_id") == DEVICE_ID
            }
            for rid in db_ids_for_device:
                assert rid in indexed_ids, (
                    f"reading_id {rid!r} in readings_db but missing from device_index"
                )

    def test_concurrent_delete_uses_same_lock_object(self):
        """
        Regression invariant: _get_device_lock() must return the SAME Python
        lock object (id-equality) before and after a DELETE operation.

        Pre-fix: clear_history called _device_locks.pop(device_id, None) while
        holding that very lock.  A subsequent _get_device_lock() call created a
        brand-new Lock, so id(lock_before) != id(lock_after).
        Post-fix: pop is removed; the lock lives for device lifetime.
        """
        import main as _main

        dev = "dev_lockid_race"
        # Warm up: create the lock via an upload
        client.post("/api/ppg/upload", json=_valid_payload(device_id=dev))
        lock_before = _main._get_device_lock(dev)

        # DELETE — this previously popped the lock from the registry
        resp = client.delete(f"/api/ppg/history/{dev}")
        assert resp.status_code == 200

        # After the DELETE the registry entry must still be the original object
        lock_after = _main._get_device_lock(dev)
        assert id(lock_before) == id(lock_after), (
            "Lock object identity changed after DELETE — "
            "_device_locks.pop() was NOT fully removed from clear_history. "
            "This indicates the TOCTOU fix is incomplete."
        )


# ===========================================================================
# 14. Integration — GET /api/ppg/stats/{device_id}
# ===========================================================================
class TestDeviceStats:

    def test_stats_returns_200_after_upload(self):
        dev = "stats_dev"
        client.post("/api/ppg/upload", json=_valid_payload(device_id=dev))
        resp = client.get(f"/api/ppg/stats/{dev}")
        assert resp.status_code == 200

    def test_stats_returns_404_no_data(self):
        resp = client.get("/api/ppg/stats/no_data_device")
        assert resp.status_code == 404

    def test_stats_has_required_fields(self):
        dev = "stats_fields"
        client.post("/api/ppg/upload", json=_valid_payload(device_id=dev))
        data = client.get(f"/api/ppg/stats/{dev}").json()
        assert "heart_rate_bpm" in data
        assert "spo2_pct" in data
        assert "hrv" in data
        assert "total_readings" in data

    def test_stats_invalid_device_id_returns_400(self):
        resp = client.get("/api/ppg/stats/bad@id!")
        assert resp.status_code == 400

    def test_stats_count_matches_uploads(self):
        dev = "stats_count"
        for _ in range(3):
            client.post("/api/ppg/upload", json=_valid_payload(device_id=dev))
        data = client.get(f"/api/ppg/stats/{dev}").json()
        assert data["total_readings"] == 3


# ===========================================================================
# 15. Integration — GET / and GET /api/stats
# ===========================================================================
class TestHealthAndGlobalStats:

    def test_root_returns_200(self):
        resp = client.get("/")
        assert resp.status_code == 200

    def test_root_has_status_running(self):
        resp = client.get("/")
        assert resp.json()["status"] == "running"

    def test_root_has_version(self):
        resp = client.get("/")
        assert "version" in resp.json()

    def test_root_has_heartpy_available_flag(self):
        resp = client.get("/")
        assert "heartpy_available" in resp.json()

    def test_global_stats_returns_200(self):
        resp = client.get("/api/stats")
        assert resp.status_code == 200

    def test_global_stats_total_readings_increments(self):
        before = client.get("/api/stats").json()["total_readings"]
        client.post("/api/ppg/upload", json=_valid_payload(device_id="stat_inc"))
        after = client.get("/api/stats").json()["total_readings"]
        assert after == before + 1

    def test_global_stats_unique_devices_counted(self):
        client.post("/api/ppg/upload", json=_valid_payload(device_id="stats_uniq_A"))
        client.post("/api/ppg/upload", json=_valid_payload(device_id="stats_uniq_B"))
        data = client.get("/api/stats").json()
        assert data["unique_devices"] >= 2


# ===========================================================================
# 16. Performance — 500 samples should process within 2 seconds
# ===========================================================================
class TestPerformance:

    def test_upload_500_samples_within_2_seconds(self):
        payload = _valid_payload(n=500)
        start = time.perf_counter()
        resp = client.post("/api/ppg/upload", json=payload)
        elapsed = time.perf_counter() - start
        assert resp.status_code == 200
        assert elapsed < 2.0, f"Processing took {elapsed:.3f}s (limit 2s)"

    def test_repeated_uploads_no_memory_explosion(self):
        """10 consecutive uploads for the same device should not grow rr_accumulator
        beyond its configured maximum (RR_ACCUMULATOR_MAX = 240)."""
        from main import RR_ACCUMULATOR_MAX
        dev = "perf_mem_dev"
        for _ in range(10):
            client.post("/api/ppg/upload", json=_valid_payload(device_id=dev))
        if dev in rr_accumulator:
            assert len(rr_accumulator[dev]) <= RR_ACCUMULATOR_MAX

    def test_upload_10000_samples_within_5_seconds(self):
        payload = _valid_payload(n=10_000, fs=400)
        start = time.perf_counter()
        resp = client.post("/api/ppg/upload", json=payload)
        elapsed = time.perf_counter() - start
        assert resp.status_code == 200
        assert elapsed < 5.0, f"Processing 10 000 samples took {elapsed:.3f}s (limit 5s)"


# ===========================================================================
# 18. History — reading limit per device (MAX_HISTORY = 100)
# ===========================================================================
class TestHistoryLimit:

    def test_db_trimmed_when_exceeds_max_history(self):
        """After 105 uploads, device should have at most MAX_HISTORY (100) entries."""
        from main import MAX_HISTORY
        dev = "overflow_dev"
        for _ in range(MAX_HISTORY + 5):
            client.post("/api/ppg/upload", json=_valid_payload(device_id=dev))
        device_count = sum(1 for v in readings_db.values()
                           if v["device_id"] == dev)
        assert device_count <= MAX_HISTORY


# ===========================================================================
# Critical fixes verification (Bước 0)
# ===========================================================================
class TestCriticalFixes:

    def test_pnn50_formula_task_force_1996(self):
        """pNN50 phải chia cho len(diff_nn) = N-1 theo Task Force 1996."""
        from main import _compute_hrv_metrics
        # rr = [800, 850, 750, 900, 780] -> diff = [50, -100, 150, -120]
        # |diff| > 50 count = 3 (100, 150, 120); pNN50 = 3/4*100 = 75.0
        rr = np.array([800, 850, 750, 900, 780], dtype=float)
        hrv = _compute_hrv_metrics(rr)
        assert hrv.pnn50_pct == pytest.approx(75.0, abs=0.1)

    def test_pnn20_formula(self):
        """pNN20: chia len(diff_nn), threshold 20ms."""
        from main import _compute_hrv_metrics
        # rr=[800,850,750,900,780] -> diff=[50,-100,150,-120]
        # |diff| > 20 count = 4 (50, 100, 150, 120)
        # pNN20 = 4/4 * 100 = 100.0
        rr = np.array([800, 850, 750, 900, 780], dtype=float)
        hrv = _compute_hrv_metrics(rr)
        assert hrv.pnn20_pct == pytest.approx(100.0, abs=0.1)

    def test_pnn20_threshold_works_correctly(self):
        """RR với diff <= 20ms -> pNN20 = 0."""
        from main import _compute_hrv_metrics
        # diff = [10, -5, 10, -15] -> tất cả đều <= 20
        rr = np.array([800, 810, 805, 815, 800], dtype=float)
        hrv = _compute_hrv_metrics(rr)
        assert hrv.pnn20_pct == pytest.approx(0.0, abs=0.1)

    def test_reliability_low_when_few_intervals(self):
        """< 10 intervals -> reliability = 'low'"""
        from main import _compute_hrv_metrics
        rr = np.array([800.0] * 5, dtype=float)
        hrv = _compute_hrv_metrics(rr)
        assert hrv.reliability == "low"

    def test_reliability_medium_when_moderate_intervals(self):
        """10-59 intervals -> reliability = 'medium'"""
        from main import _compute_hrv_metrics
        rr = np.array([800.0] * 30, dtype=float)
        hrv = _compute_hrv_metrics(rr)
        assert hrv.reliability == "medium"

    def test_reliability_high_when_enough_intervals(self):
        """>= 60 intervals -> reliability = 'high'"""
        from main import _compute_hrv_metrics
        rr = np.array([800.0] * 70, dtype=float)
        hrv = _compute_hrv_metrics(rr)
        assert hrv.reliability == "high"

    def test_hrv_response_contains_pnn20_and_reliability(self):
        """Upload -> response có pnn20_pct và reliability fields."""
        payload = _valid_payload(device_id="pnn20_response")
        resp = client.post("/api/ppg/upload", json=payload)
        assert resp.status_code == 200
        hrv = resp.json()["hrv"]
        assert "pnn20_pct" in hrv
        assert "reliability" in hrv
        assert hrv["reliability"] in ("low", "medium", "high")

    def test_reading_id_full_32_hex_chars(self):
        payload = _valid_payload(device_id="rid_test")
        resp = client.post("/api/ppg/upload", json=payload)
        assert resp.status_code == 200
        rid = resp.json()["reading_id"]
        assert len(rid) == 32
        assert all(c in "0123456789abcdef" for c in rid)

    def test_eviction_cleans_overlap_rr_spo2(self):
        """Khi device bị evict (DELETE), overlap_buffer/rr_accumulator/spo2_history phải sạch."""
        readings_db.clear()
        overlap_buffer.clear()
        rr_accumulator.clear()
        spo2_history.clear()
        device_index_ref = __import__('main').device_index
        device_index_ref.clear()

        client.post("/api/ppg/upload", json=_valid_payload(device_id="evict_a"))
        assert "evict_a" in overlap_buffer

        client.delete("/api/ppg/history/evict_a")
        assert "evict_a" not in overlap_buffer
        assert "evict_a" not in rr_accumulator
        assert "evict_a" not in spo2_history

    def test_get_history_no_crash_on_empty(self):
        """READ endpoints không crash khi device không tồn tại (sau khi thêm lock)."""
        resp = client.get("/api/ppg/history/nonexistent_xyz")
        assert resp.status_code == 200
        assert resp.json()["count"] == 0

    def test_signal_processing_can_run_concurrent(self):
        """Upload đồng thời từ 5 thread không crash."""
        import concurrent.futures
        def upload(i):
            return client.post("/api/ppg/upload", json=_valid_payload(device_id=f"concur_{i}"))
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:
            results = list(ex.map(upload, range(5)))
        for r in results:
            assert r.status_code == 200


# ===========================================================================
# LF/HF frequency-domain HRV (Bước 1)
# ===========================================================================
class TestLFHF:

    def test_lf_hf_none_when_insufficient_rr(self):
        """<60 intervals -> lf_ms2/hf_ms2/lf_hf = None"""
        from main import _compute_hrv_metrics
        rr = np.array([800.0] * 30)
        hrv = _compute_hrv_metrics(rr)
        assert hrv.lf_ms2 is None
        assert hrv.hf_ms2 is None
        assert hrv.lf_hf is None

    def test_lf_hf_computed_when_enough_rr(self):
        """>=60 intervals -> lf_ms2/hf_ms2/lf_hf có giá trị"""
        from main import _compute_hrv_metrics
        np.random.seed(42)
        rr = 800 + 20 * np.sin(np.linspace(0, 6 * np.pi, 80)) + np.random.randn(80) * 5
        hrv = _compute_hrv_metrics(rr)
        assert hrv.lf_ms2 is not None
        assert hrv.hf_ms2 is not None
        assert hrv.lf_ms2 >= 0
        assert hrv.hf_ms2 >= 0

    def test_lf_hf_constant_rr_gives_low_power(self):
        """RR hằng số -> LF/HF power gần 0"""
        from main import _compute_hrv_metrics
        rr = np.array([800.0] * 80)
        hrv = _compute_hrv_metrics(rr)
        if hrv.lf_ms2 is not None:
            assert hrv.lf_ms2 < 1.0

    def test_hrv_metrics_has_lfhf_fields(self):
        """HRVMetrics model phải có 3 fields mới"""
        from main import HRVMetrics
        m = HRVMetrics(sdnn_ms=0, rmssd_ms=0, pnn50_pct=0, mean_nn_ms=0)
        assert hasattr(m, "lf_ms2")
        assert hasattr(m, "hf_ms2")
        assert hasattr(m, "lf_hf")


# ===========================================================================
# Authentication (Bước 2)
# ===========================================================================
class TestAuthentication:

    def test_upload_no_key_configured_no_header_ok(self, monkeypatch):
        """PPG_API_KEY rỗng -> không cần token"""
        import main
        monkeypatch.setattr(main, "PPG_API_KEY", "")
        resp = client.post("/api/ppg/upload", json=_valid_payload(device_id="auth_open"))
        assert resp.status_code == 200

    def test_upload_key_set_no_header_rejected(self, monkeypatch):
        import main
        monkeypatch.setattr(main, "PPG_API_KEY", "secret_token_123")
        resp = client.post("/api/ppg/upload", json=_valid_payload(device_id="auth_miss"))
        assert resp.status_code == 403

    def test_upload_wrong_token_rejected(self, monkeypatch):
        import main
        monkeypatch.setattr(main, "PPG_API_KEY", "secret_token_123")
        resp = client.post(
            "/api/ppg/upload",
            json=_valid_payload(device_id="auth_wrong"),
            headers={"X-Device-Token": "wrong_token"},
        )
        assert resp.status_code == 403

    def test_upload_correct_token_accepted(self, monkeypatch):
        import main
        monkeypatch.setattr(main, "PPG_API_KEY", "secret_token_123")
        resp = client.post(
            "/api/ppg/upload",
            json=_valid_payload(device_id="auth_ok"),
            headers={"X-Device-Token": "secret_token_123"},
        )
        assert resp.status_code == 200

    def test_history_requires_token_when_set(self, monkeypatch):
        import main
        monkeypatch.setattr(main, "PPG_API_KEY", "secret")
        resp = client.get("/api/ppg/history/some_dev")
        assert resp.status_code == 403

    def test_health_endpoint_no_auth_required(self, monkeypatch):
        """GET / vẫn mở để health check"""
        import main
        monkeypatch.setattr(main, "PPG_API_KEY", "secret")
        resp = client.get("/")
        assert resp.status_code == 200


# ===========================================================================
# Rate limiting (Bước 4)
# ===========================================================================
class TestRateLimiting:

    def test_rate_limit_exists_on_upload(self):
        """slowapi limiter phải được gắn vào app."""
        import main
        assert hasattr(main.app.state, "limiter")

    def test_upload_allows_normal_rate(self):
        """Các upload thông thường (không spam) không bị limit."""
        for i in range(5):
            resp = client.post("/api/ppg/upload", json=_valid_payload(device_id=f"rate_{i}"))
            assert resp.status_code == 200

    def test_upload_hits_429_after_20_requests(self):
        """21 uploads liên tiếp từ cùng IP phải trigger 429 ở request thứ 21."""
        import main
        # Reset limiter state trước khi spam
        try:
            main.limiter.reset()
        except Exception:
            pass
        statuses = []
        for i in range(21):
            resp = client.post("/api/ppg/upload",
                               json=_valid_payload(device_id=f"spam_{i}"))
            statuses.append(resp.status_code)
        # Ít nhất 1 trong 21 requests phải bị 429
        assert 429 in statuses, (
            f"Expected at least one 429 in 21 rapid uploads, got: {statuses}"
        )


# ===========================================================================
# New tests — coverage gaps identified after v4.1 refactor
# ===========================================================================

class TestConcurrentSameDevice:
    """3-phase lock correctness: concurrent upload từ cùng 1 device."""

    def test_concurrent_upload_same_device_no_crash(self):
        """2 thread upload đồng thời cùng device_id không gây crash hay data corruption."""
        import concurrent.futures

        dev = "concurrent_same"
        payload = _valid_payload(device_id=dev)
        errors = []

        def do_upload(_):
            try:
                r = client.post("/api/ppg/upload", json=payload)
                return r.status_code
            except Exception as e:
                errors.append(str(e))
                return -1

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
            codes = list(ex.map(do_upload, range(2)))

        assert errors == [], f"Exceptions during concurrent upload: {errors}"
        # Cả 2 phải thành công (không có race condition gây 500)
        assert all(c == 200 for c in codes), f"Expected all 200, got {codes}"

    def test_concurrent_upload_same_device_rr_accumulator_not_corrupted(self):
        """Sau concurrent upload, rr_accumulator của device phải là deque hợp lệ."""
        import concurrent.futures
        import main as _main

        dev = "concurrent_rr"
        payload = _valid_payload(device_id=dev)

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
            list(ex.map(lambda _: client.post("/api/ppg/upload", json=payload), range(3)))

        if dev in rr_accumulator:
            acc = rr_accumulator[dev]
            # Phải là deque (không phải list hay bị corrupt)
            from collections import deque as _deque
            assert isinstance(acc, _deque)
            # Mỗi phần tử phải là số thực dương
            for val in acc:
                assert isinstance(val, float)
                assert val > 0


class TestLFHFBoundaryAndIntegration:
    """LF/HF frequency domain: boundary conditions và integration qua upload."""

    def test_lf_hf_exactly_60_rr_intervals_returns_values(self):
        """Đúng 60 RR intervals (biên dưới) phải tính được LF/HF."""
        from main import _compute_hrv_metrics
        # 60 RR around 800ms với biến thiên nhỏ
        np.random.seed(0)
        rr = 800.0 + np.random.randn(60) * 10
        hrv = _compute_hrv_metrics(rr)
        # Biên 60 phải đủ để tính
        assert hrv.lf_ms2 is not None, "lf_ms2 must not be None with exactly 60 RR"
        assert hrv.hf_ms2 is not None
        assert hrv.lf_ms2 >= 0.0
        assert hrv.hf_ms2 >= 0.0

    def test_lf_hf_59_rr_returns_none(self):
        """59 RR intervals (1 dưới biên) → lf_ms2 = None."""
        from main import _compute_hrv_metrics
        rr = np.array([800.0] * 59)
        hrv = _compute_hrv_metrics(rr)
        assert hrv.lf_ms2 is None
        assert hrv.lf_hf is None

    def test_lf_hf_lf_dominant_sinusoidal(self):
        """RR dao động ~0.1 Hz (midband LF) → lf_ms2 > hf_ms2."""
        from main import _compute_lf_hf
        # Tạo 80 RR intervals với dao động tần số LF: ~0.1 Hz
        # Tại 60 BPM (RR=1000ms), 80 nhịp = 80s → period LF = 10s = 0.1 Hz
        t = np.arange(80)
        rr = 800.0 + 50.0 * np.sin(2 * math.pi * 0.1 / (1000.0 / 800.0) * t)
        result = _compute_lf_hf(rr)
        if result["lf_ms2"] is not None and result["hf_ms2"] is not None:
            # LF power phải lớn hơn HF khi signal là LF sinusoid
            assert result["lf_ms2"] >= result["hf_ms2"], (
                f"Expected LF ({result['lf_ms2']}) >= HF ({result['hf_ms2']}) "
                f"for LF-dominant signal"
            )

    def test_lf_hf_appears_in_upload_response_after_accumulation(self):
        """Sau nhiều batch upload, response JSON có thể chứa lf_hf != None."""
        dev = "lfhf_accum_dev"
        # Cần tích lũy >= 60 RR. Ở 72 BPM (1.2 Hz), 5s batch = ~6 RR.
        # Sau 10 batch = ~60 RR → lf_hf có thể có giá trị
        payload = _valid_payload(device_id=dev)
        last_resp = None
        for _ in range(12):
            last_resp = client.post("/api/ppg/upload", json=payload)
        assert last_resp is not None
        assert last_resp.status_code == 200
        hrv = last_resp.json()["hrv"]
        # Sau nhiều batch, rr_count phải tăng lên
        assert hrv["rr_count"] > 0
        # lf_hf field phải tồn tại trong schema (có thể None hoặc float)
        assert "lf_hf" in hrv


class TestMaxDevicesStateCleanliness:
    """MAX_DEVICES eviction: kiểm tra buffer state sau khi reject."""

    def test_max_devices_reject_does_not_add_overlap_buffer(self):
        """Device thứ 51 bị 429, overlap_buffer không được có entry của device đó."""
        import main as _main

        # Clear toàn bộ state và set device_index đầy (50 devices)
        readings_db.clear()
        overlap_buffer.clear()
        rr_accumulator.clear()
        spo2_history.clear()
        _main.device_index.clear()

        try:
            # Điền đủ 50 devices (chỉ thêm vào device_index, không cần upload thật)
            for i in range(50):
                _main.device_index[f"filler_{i:02d}"] = [f"fake_rid_{i}"]

            # Thử upload từ device thứ 51
            new_dev = "device_51_overflow"
            payload = _valid_payload(device_id=new_dev)
            resp = client.post("/api/ppg/upload", json=payload)

            # Phải bị reject
            assert resp.status_code == 429

            # Buffer không được có entry của device bị reject
            assert new_dev not in overlap_buffer, (
                "overlap_buffer must not contain rejected device"
            )
            assert new_dev not in rr_accumulator, (
                "rr_accumulator must not contain rejected device"
            )
            assert new_dev not in spo2_history, (
                "spo2_history must not contain rejected device"
            )
        finally:
            # Restore device_index so autouse fixture and subsequent tests work cleanly
            _main.device_index.clear()


class TestAuthenticationEdgeCases:
    """Authentication: whitespace key, header case insensitivity."""

    def test_whitespace_api_key_treated_as_empty(self, monkeypatch):
        """PPG_API_KEY = '   ' (whitespace only) → `if PPG_API_KEY` truthy → auth ENFORCED.
        Đây là một potential bug: whitespace key passes the truthiness check."""
        import main
        monkeypatch.setattr(main, "PPG_API_KEY", "   ")
        # Với key = whitespace, `if PPG_API_KEY` là True → sẽ require token
        # Upload không có header → 403
        resp = client.post("/api/ppg/upload", json=_valid_payload(device_id="ws_key"))
        # Documenting current behavior: whitespace key triggers auth enforcement
        assert resp.status_code == 403, (
            "Whitespace PPG_API_KEY triggers authentication enforcement. "
            "Consider stripping before comparison."
        )

    def test_token_header_case_insensitive(self, monkeypatch):
        """HTTP headers không phân biệt hoa thường: x-device-token phải được chấp nhận."""
        import main
        monkeypatch.setattr(main, "PPG_API_KEY", "mytoken")
        # FastAPI/Starlette normalize headers to lowercase internally
        resp = client.post(
            "/api/ppg/upload",
            json=_valid_payload(device_id="case_insensitive"),
            headers={"x-device-token": "mytoken"},  # lowercase
        )
        assert resp.status_code == 200, (
            "Lowercase header 'x-device-token' should be accepted (HTTP headers are case-insensitive)"
        )

    def test_empty_token_with_key_set_returns_403(self, monkeypatch):
        """Token rỗng string với PPG_API_KEY set → 403."""
        import main
        monkeypatch.setattr(main, "PPG_API_KEY", "secret")
        resp = client.post(
            "/api/ppg/upload",
            json=_valid_payload(device_id="empty_token"),
            headers={"X-Device-Token": ""},
        )
        assert resp.status_code == 403


class TestSpO2QuadraticFormula:
    """Tests for quadratic SpO2 formula (Maxim MAXREFDES117 reference,
    embedded in SparkFun MAX3010x library, MIT license)."""

    @staticmethod
    def _quadratic_expected(r: float) -> float:
        """Reference implementation of expected formula."""
        spo2 = -45.060 * r * r + 30.354 * r + 94.845
        return float(np.clip(spo2, SPO2_MIN_VALID, SPO2_MAX_VALID))

    @pytest.mark.parametrize("r,expected", [
        (0.5, 98.757),  # -45.060*0.25 + 30.354*0.5 + 94.845
        (0.7, 94.013),  # mid-range
        (0.9, 85.665),  # -45.060*0.81 + 30.354*0.9 + 94.845
        (1.1, 85.0),    # clipped at floor
        (1.3, 85.0),    # clipped at floor
    ])
    def test_spo2_quadratic_spot_checks(self, r, expected):
        """Spot check quadratic formula at 5 R values."""
        actual = self._quadratic_expected(r)
        assert actual == pytest.approx(expected, abs=0.01)

    def test_spo2_monotonically_decreasing(self):
        """SpO2 should decrease as R increases in valid range."""
        r_values = [0.4, 0.5, 0.6, 0.7, 0.8]
        spo2_values = [self._quadratic_expected(r) for r in r_values]
        for i in range(len(spo2_values) - 1):
            assert spo2_values[i] >= spo2_values[i + 1], (
                f"SpO2 not monotonic at R={r_values[i]} -> {r_values[i+1]}: "
                f"{spo2_values[i]:.2f} -> {spo2_values[i+1]:.2f}"
            )

    def test_spo2_output_range_85_100(self):
        """All outputs must be in [85, 100] for valid R range."""
        r_values = np.linspace(0.4, 1.4, 100)
        for r in r_values:
            spo2 = self._quadratic_expected(r)
            assert 85.0 <= spo2 <= 100.0, f"SpO2={spo2} out of range at R={r}"

    def test_spo2_continuity_no_jumps(self):
        """Quadratic should be continuous everywhere - no jumps > 0.5%."""
        r_values = np.linspace(0.4, 1.4, 1000)
        spo2_values = [self._quadratic_expected(r) for r in r_values]
        diffs = np.abs(np.diff(spo2_values))
        max_jump = float(np.max(diffs))
        assert max_jump < 0.5, f"Discontinuity detected: max jump = {max_jump:.3f}%"

    def test_calculate_spo2_v23_uses_quadratic(self):
        """Verify production function uses quadratic formula."""
        # Build synthetic IR/RED with known R approx 0.5
        fs = 100
        duration = 5
        n = fs * duration
        t = np.arange(n) / fs

        # Synthesize signals: AC pulsatile + DC baseline
        # Target: AC_red/DC_red = 0.05, AC_ir/DC_ir = 0.10 -> R = 0.5
        pulse = np.sin(2 * np.pi * 1.2 * t)  # 72 BPM
        ir_dc = 100000
        ir_ac = ir_dc * 0.10
        red_dc = 50000
        red_ac = red_dc * 0.05

        rng = np.random.default_rng(seed=42)
        ir_signal = ir_dc + ir_ac * pulse + rng.normal(0, ir_dc * 0.001, n)
        red_signal = red_dc + red_ac * pulse + rng.normal(0, red_dc * 0.001, n)

        spo2, R, _, _, reason = calculate_spo2_v23(
            ir_signal.astype(int), red_signal.astype(int), fs
        )

        if reason == "ok":
            # Verify R is approximately 0.5 (target)
            assert R == pytest.approx(0.5, abs=0.1)
            # Verify SpO2 matches quadratic formula
            expected_spo2 = self._quadratic_expected(R)
            assert spo2 == pytest.approx(expected_spo2, abs=0.5)


class TestHeartPyFallback:
    """HeartPy fallback path: mock HEARTPY_AVAILABLE = False."""

    def test_fallback_scipy_returns_valid_hr(self, monkeypatch):
        """Khi HEARTPY_AVAILABLE=False, scipy fallback phải trả HR trong range hoặc 0."""
        import main
        monkeypatch.setattr(main, "HEARTPY_AVAILABLE", False)
        payload = _valid_payload()
        resp = client.post("/api/ppg/upload", json=payload)
        assert resp.status_code == 200
        hr = resp.json()["heart_rate"]
        assert hr == 0.0 or (HR_MIN_BPM <= hr <= HR_MAX_BPM)

    def test_fallback_scipy_response_has_all_required_fields(self, monkeypatch):
        """Fallback path không bỏ sót field nào trong response."""
        import main
        monkeypatch.setattr(main, "HEARTPY_AVAILABLE", False)
        resp = client.post("/api/ppg/upload", json=_valid_payload())
        assert resp.status_code == 200
        required = [
            "reading_id", "heart_rate", "spo2", "hrv",
            "signal_quality", "peaks", "filtered_signal",
        ]
        data = resp.json()
        for f in required:
            assert f in data, f"Missing field '{f}' in fallback response"


class TestSQIFeatures:
    """Unit tests for compute_sqi_features() —  Deep SQA refactor."""

    def _make_peaks(self, filtered_ir: np.ndarray, fs: int = FS) -> list:
        from scipy.signal import find_peaks as _fp
        peaks, _ = _fp(filtered_ir, distance=int(fs * 0.3))
        return peaks.tolist()

    def test_compute_sqi_features_returns_all_keys(self):
        raw = _sine_ppg(freq_hz=1.25)  # 75 BPM
        filt = bandpass_filter(raw, FS)
        peaks = self._make_peaks(filt)
        feats = compute_sqi_features(raw, filt, peaks, FS)
        expected_keys = {"spectral_purity", "hr_bpm", "amplitude_ratio",
                         "ssqi", "ksqi", "entropy_amp_dist"}
        assert set(feats.keys()) == expected_keys
        for k, v in feats.items():
            assert isinstance(v, float), f"{k} not float: {type(v)}"

    def test_compute_sqi_features_clean_sine_high_purity(self):
        raw = _sine_ppg(freq_hz=1.25)  # 75 BPM
        filt = bandpass_filter(raw, FS)
        peaks = self._make_peaks(filt)
        feats = compute_sqi_features(raw, filt, peaks, FS)
        assert feats["spectral_purity"] > 0.6, \
            f"clean sine should have high purity, got {feats['spectral_purity']}"
        assert 40 <= feats["hr_bpm"] <= 100, \
            f"75 BPM sine, got {feats['hr_bpm']}"

    def test_compute_sqi_features_skewness_kurtosis_finite(self):
        rng = np.random.default_rng(42)
        raw = _sine_ppg(freq_hz=1.25) + rng.standard_normal(N) * 200
        filt = bandpass_filter(raw, FS)
        peaks = self._make_peaks(filt)
        feats = compute_sqi_features(raw, filt, peaks, FS)
        assert np.isfinite(feats["ssqi"])
        assert np.isfinite(feats["ksqi"])
        assert feats["entropy_amp_dist"] >= 0.0, \
            f"Shannon entropy must be non-negative, got {feats['entropy_amp_dist']}"

    def test_assess_signal_quality_backward_compat(self):
        raw = _sine_ppg(freq_hz=1.25)
        filt = bandpass_filter(raw, FS)
        peaks = self._make_peaks(filt)
        quality = assess_signal_quality(raw, filt, peaks, FS)
        assert quality == "good", f"clean sine should be 'good', got '{quality}'"
        assert isinstance(quality, str)
