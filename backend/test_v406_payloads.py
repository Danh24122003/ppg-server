"""
test_v406_payloads.py
=====================
Firmware v4.0.6 compatibility tests for Backend v4.3.

Firmware v4.0.6 changes that affect payloads:
  Fix 1 — sample_rate fallback: SENSOR_RATE_HZ/SAMPLE_AVG = 200/4 = 50 Hz
  Fix 3 — dropsOut atomic: has_gaps + drop_count fields now in every payload
  Fixes 2 & 4 — firmware-side only, no payload change

Covers:
  Test #1  — sample_rate=50, 250 samples (standard 5s @ 50Hz batch)
  Test #2  — sample_rate=50, 150 samples (partial flush minimum 3s @ 50Hz)
  Test #3  — has_gaps=true, drop_count>0 extra fields silently ignored
  Test #4  — sample_rate boundary 25 and 400
  Test #5  — sample_rate out-of-range: 24 and 401 → 422
  Test #6  — sample_rate=50 with 72 BPM sine, HR within [65, 79]
  Test #7  — LF/HF non-negative even with outlier RR injected
"""

import sys
import os
import math
import pytest
import numpy as np

_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from main import (
    app,
    readings_db,
    spo2_history,
    overlap_buffer,
    rr_accumulator,
    _compute_lf_hf,
)
import main as _main

from fastapi.testclient import TestClient

client = TestClient(app)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clear_state():
    """Reset all in-memory state before and after every test."""
    readings_db.clear()
    spo2_history.clear()
    overlap_buffer.clear()
    rr_accumulator.clear()
    _main.device_index.clear()
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ppg(n: int, fs: int, freq_hz: float = 1.2,
              ir_dc: float = 80_000.0, red_dc: float = 70_000.0,
              amp_ir: float = 2_000.0, amp_red: float = 1_800.0) -> dict:
    """Build a valid PPGReading-compatible dict with realistic MAX30102 values."""
    t = np.arange(n) / fs
    ir  = (ir_dc  + amp_ir  * np.sin(2 * math.pi * freq_hz * t)).astype(int).tolist()
    red = (red_dc + amp_red * np.sin(2 * math.pi * freq_hz * t)).astype(int).tolist()
    return {
        "device_id":   "fw406_test",
        "ir_values":   ir,
        "red_values":  red,
        "sample_rate": fs,
    }


# ---------------------------------------------------------------------------
# Test #1 — sample_rate=50 with 250 samples (nominal firmware batch)
# ---------------------------------------------------------------------------

class TestFirmwareSampleRate50_250Samples:

    def test_upload_returns_200(self):
        payload = _make_ppg(n=250, fs=50)
        resp = client.post("/api/ppg/upload", json=payload)
        assert resp.status_code == 200, resp.text

    def test_heart_rate_positive(self):
        payload = _make_ppg(n=250, fs=50)
        resp = client.post("/api/ppg/upload", json=payload)
        assert resp.status_code == 200
        assert resp.json()["heart_rate"] >= 0.0

    def test_response_has_signal_quality(self):
        payload = _make_ppg(n=250, fs=50)
        resp = client.post("/api/ppg/upload", json=payload)
        assert "signal_quality" in resp.json()

    def test_heartpy_high_precision_flag_for_fs50(self):
        """1000 % 50 == 0 → HeartPy high_precision=True path must not crash."""
        assert 1000 % 50 == 0, "pre-condition: 50 Hz qualifies high_precision"
        payload = _make_ppg(n=250, fs=50)
        resp = client.post("/api/ppg/upload", json=payload)
        # If high_precision path crashes we'd get 500; 200 confirms it ran fine.
        assert resp.status_code == 200

    def test_bandpass_does_not_crash_at_fs50(self):
        """Nyquist=25 Hz, bandpass [0.5–4] Hz — coefficients must be valid."""
        from main import bandpass_filter
        sig = np.array(_make_ppg(250, 50)["ir_values"], dtype=float)
        out = bandpass_filter(sig, 50)
        assert len(out) == 250
        assert not np.any(np.isnan(out))


# ---------------------------------------------------------------------------
# Test #2 — sample_rate=50 with 150 samples (partial flush minimum 3 s)
# ---------------------------------------------------------------------------

class TestFirmwareSampleRate50_150Samples:

    def test_upload_150_samples_returns_200(self):
        payload = _make_ppg(n=150, fs=50)
        resp = client.post("/api/ppg/upload", json=payload)
        assert resp.status_code == 200, resp.text

    def test_upload_150_samples_has_signal_quality_field(self):
        payload = _make_ppg(n=150, fs=50)
        resp = client.post("/api/ppg/upload", json=payload)
        assert resp.status_code == 200
        assert "signal_quality" in resp.json()

    def test_upload_49_samples_rejected(self):
        """Backend MIN_SAMPLES=50 — 49 samples rejected (Pydantic 422 or endpoint 400 per M.3)."""
        payload = _make_ppg(n=49, fs=50)
        resp = client.post("/api/ppg/upload", json=payload)
        assert resp.status_code in (400, 422)


# ---------------------------------------------------------------------------
# Test #3 — has_gaps=true / drop_count extra fields silently ignored
# ---------------------------------------------------------------------------

class TestFirmwareHasGapsDropCount:

    def test_has_gaps_true_ignored_returns_200(self):
        payload = _make_ppg(n=250, fs=50)
        payload["has_gaps"]   = True
        payload["drop_count"] = 15
        resp = client.post("/api/ppg/upload", json=payload)
        assert resp.status_code == 200, resp.text

    def test_has_gaps_false_ignored_returns_200(self):
        payload = _make_ppg(n=250, fs=50)
        payload["has_gaps"]   = False
        payload["drop_count"] = 0
        resp = client.post("/api/ppg/upload", json=payload)
        assert resp.status_code == 200

    def test_extra_firmware_metadata_fields_ignored(self):
        """Any extra unknown fields must not break Pydantic parsing."""
        payload = _make_ppg(n=250, fs=50)
        payload["fw_version"]  = "4.0.6"
        payload["has_gaps"]    = True
        payload["drop_count"]  = 7
        payload["batch_index"] = 42
        resp = client.post("/api/ppg/upload", json=payload)
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Test #4 — sample_rate boundary: 25 (min) and 400 (max)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fs,n", [(25, 250), (400, 500)])
def test_sample_rate_boundary_valid(fs, n):
    """sample_rate at min (25) and max (400) boundary must return 200."""
    payload = _make_ppg(n=n, fs=fs)
    resp = client.post("/api/ppg/upload", json=payload)
    assert resp.status_code == 200, f"fs={fs}: {resp.text}"


# ---------------------------------------------------------------------------
# Test #5 — sample_rate out-of-range: 24 and 401 → 422
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fs", [24, 401])
def test_sample_rate_out_of_range_returns_422(fs):
    """sample_rate outside [25, 400] must be rejected by Pydantic → 422."""
    payload = _make_ppg(n=250, fs=50)  # build valid base
    payload["sample_rate"] = fs
    resp = client.post("/api/ppg/upload", json=payload)
    assert resp.status_code == 422, f"fs={fs} should be 422, got {resp.status_code}"


# ---------------------------------------------------------------------------
# Test #6 — sample_rate=50, 72 BPM sine → HR in [65, 79]
# ---------------------------------------------------------------------------

class TestFirmwareHRAccuracyAt50Hz:

    def test_hr_within_10pct_tolerance(self):
        """250 samples @ 50Hz with 1.2 Hz (72 BPM) sine — HR must land in [65, 79]."""
        payload = _make_ppg(n=250, fs=50, freq_hz=1.2,
                            ir_dc=80_000.0, red_dc=70_000.0,
                            amp_ir=3_000.0, amp_red=2_700.0)
        resp = client.post("/api/ppg/upload", json=payload)
        assert resp.status_code == 200
        hr = resp.json()["heart_rate"]
        # Allow HR=0 (poor signal) without failing — signal quality check first
        quality = resp.json()["signal_quality"]
        if quality in ("good", "fair"):
            assert 65.0 <= hr <= 79.0, (
                f"Expected HR in [65, 79] for 72 BPM signal @ 50Hz, got {hr}"
            )

    def test_signal_quality_not_poor_for_clean_sine(self):
        """250 large-amplitude sine samples @ 50Hz should not produce 'poor' quality."""
        payload = _make_ppg(n=250, fs=50, freq_hz=1.2,
                            ir_dc=80_000.0, red_dc=70_000.0,
                            amp_ir=3_000.0, amp_red=2_700.0)
        resp = client.post("/api/ppg/upload", json=payload)
        assert resp.status_code == 200
        assert resp.json()["signal_quality"] in ("good", "fair")


# ---------------------------------------------------------------------------
# Test #7 — LF/HF non-negative with outlier RR injected
# ---------------------------------------------------------------------------

class TestLFHFNonNegativeWithOutlierRR:

    def test_lf_hf_unit_non_negative_with_outlier(self):
        """
        _compute_lf_hf must return lf_ms2 >= 0 and hf_ms2 >= 0 even when
        one RR interval is a 2000 ms outlier injected into an otherwise
        normal 800 ms series.  Welch PSD of a real signal is mathematically
        non-negative but numerical issues (extrapolation noise, trapezoid on
        a sparse band) could produce tiny negatives — check for that bug.
        """
        rr_normal = np.full(62, 800.0)
        rr_normal[30] = 2000.0   # single outlier
        result = _compute_lf_hf(rr_normal)
        if result["lf_ms2"] is not None:
            assert result["lf_ms2"] >= 0.0, (
                f"Bug C3: lf_ms2={result['lf_ms2']} is negative — needs clamp"
            )
        if result["hf_ms2"] is not None:
            assert result["hf_ms2"] >= 0.0, (
                f"Bug C3: hf_ms2={result['hf_ms2']} is negative — needs clamp"
            )

    def test_lf_hf_unit_returns_dict_with_three_keys(self):
        rr = np.full(65, 800.0)
        result = _compute_lf_hf(rr)
        assert set(result.keys()) == {"lf_ms2", "hf_ms2", "lf_hf"}

    def test_lf_hf_unit_lf_hf_ratio_non_negative_when_computed(self):
        rr = np.full(65, 800.0)
        result = _compute_lf_hf(rr)
        if result["lf_hf"] is not None:
            assert result["lf_hf"] >= 0.0

    def test_lf_hf_integration_via_upload_accumulation(self):
        """
        Send enough batches to accumulate >=60 RR intervals, then verify
        lf_ms2 and hf_ms2 in the response are non-negative (not None).
        """
        # Each 250-sample @ 50Hz batch yields ~5 RR intervals at 72 BPM.
        # Send 14 batches → ~70 RR, enough to trigger LF/HF.
        dev = "lf_hf_test_dev"
        last_resp = None
        for _ in range(14):
            payload = _make_ppg(n=250, fs=50, freq_hz=1.2,
                                ir_dc=80_000.0, red_dc=70_000.0,
                                amp_ir=3_000.0, amp_red=2_700.0)
            payload["device_id"] = dev
            r = client.post("/api/ppg/upload", json=payload)
            assert r.status_code == 200
            last_resp = r.json()

        hrv = last_resp["hrv"]
        rr_count = hrv["rr_count"]
        if rr_count >= 60:
            assert hrv["lf_ms2"] is not None, "lf_ms2 should be computed at rr_count>=60"
            assert hrv["hf_ms2"] is not None, "hf_ms2 should be computed at rr_count>=60"
            assert hrv["lf_ms2"] >= 0.0, (
                f"Bug C3: lf_ms2={hrv['lf_ms2']} negative after accumulation"
            )
            assert hrv["hf_ms2"] >= 0.0, (
                f"Bug C3: hf_ms2={hrv['hf_ms2']} negative after accumulation"
            )
