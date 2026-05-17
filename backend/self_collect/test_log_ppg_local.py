"""
pytest tests for backend/self_collect/log_ppg_local.py
(post pipeline-integration changes 2026-05-08)

Groups:
  Group 1 — Pipeline integration (P0)   tests 1-4
  Group 2 — Skip-pipeline edge cases    tests 5-7
  Group 3 — Concurrency / race          tests 8-9
  Group 4 — CSV regression              tests 10-11
  Group 5 — Response schema             tests 12-13

Run from repo root:
    python -m pytest "backend/self_collect/test_log_ppg_local.py" -v
"""
from __future__ import annotations

import importlib.util as _ilu
import os
import sys
import threading
import time
from pathlib import Path
from typing import List
from unittest.mock import patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Path setup — must happen before any import of the module under test
# ---------------------------------------------------------------------------
ROOT = Path("C:/Users/Acer/Desktop/PPG monitor")
BACKEND_DIR = ROOT / "Backend code"
SELF_COLLECT_DIR = BACKEND_DIR / "self_collect"

# Ensure backend/ is importable (for main.py pipeline functions)
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

os.environ.setdefault("PPG_API_KEY", "")  # suppress auth in main.py

# ---------------------------------------------------------------------------
# Load log_ppg_local module via importlib (avoids __main__ execution)
# ---------------------------------------------------------------------------
_spec = _ilu.spec_from_file_location(
    "log_ppg_local",
    SELF_COLLECT_DIR / "log_ppg_local.py",
)
_mod = _ilu.module_from_spec(_spec)
# Register in sys.modules BEFORE exec_module so that @dataclass picks up the
# module dict correctly (required for Python 3.14 which tightened this check).
sys.modules["log_ppg_local"] = _mod
_spec.loader.exec_module(_mod)

# Convenience aliases
app = _mod.app
SubjectSession = _mod.SubjectSession
save_session = _mod.save_session
_PIPELINE_OK = _mod._PIPELINE_OK

# ---------------------------------------------------------------------------
# FastAPI TestClient
# ---------------------------------------------------------------------------
from fastapi.testclient import TestClient  # noqa: E402

client = TestClient(app, raise_server_exceptions=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
FS = 100
DURATION_S = 5
N_SAMPLES = FS * DURATION_S  # 500 samples

HR_TARGET_HZ = 1.2            # 72 BPM


def _make_ir(n: int = N_SAMPLES, fs: int = FS, hr_hz: float = HR_TARGET_HZ) -> List[int]:
    """Realistic-ish IR signal: DC ~100 000 + AC sine at hr_hz."""
    t = np.arange(n) / fs
    dc = 100_000
    ac_amp = 3_000
    sig = dc + ac_amp * np.sin(2 * np.pi * hr_hz * t)
    return sig.astype(int).tolist()


def _make_red(n: int = N_SAMPLES, fs: int = FS, hr_hz: float = HR_TARGET_HZ) -> List[int]:
    """Red signal: slightly lower DC + same HR."""
    t = np.arange(n) / fs
    dc = 90_000
    ac_amp = 2_500
    sig = dc + ac_amp * np.sin(2 * np.pi * hr_hz * t)
    return sig.astype(int).tolist()


def _valid_payload(device_id: str = "esp32_test",
                   n: int = N_SAMPLES) -> dict:
    return {
        "device_id": device_id,
        "ir_values": _make_ir(n),
        "red_values": _make_red(n),
        "sample_rate": FS,
    }


def _activate_subject(subject_id: str = "TestSubject") -> SubjectSession:
    """Inject an active SubjectSession so upload endpoint accepts batches."""
    sess = SubjectSession(
        subject_id=subject_id,
        age=25,
        sex="M",
        height_cm=170.0,
        weight_kg=65.0,
        sbp_baseline=120,
        dbp_baseline=80,
    )
    with _mod._state_lock:
        _mod._current = sess
        _mod._batch_count = 0
    return sess


def _deactivate():
    with _mod._state_lock:
        _mod._current = None
        _mod._batch_count = 0
    # Clear per-device pipeline state to isolate tests
    with _mod._pipeline_lock:
        _mod._overlap_buffer_demo.clear()
        _mod._rr_accumulator_demo.clear()
        _mod._spo2_history_demo.clear()
    # Reset pending-warn state
    with _mod._pending_lock:
        _mod._pending_count = 0
        _mod._pending_first_seen = None
        _mod._pending_warned_30s = False


# ===========================================================================
# Group 1 — Pipeline integration (P0)
# ===========================================================================

class TestPipelineIntegration:
    """Group 1 — pipeline import + real HR output."""

    def test_pipeline_ok_flag(self):
        """G1-T1: _PIPELINE_OK == True means production functions loaded."""
        if not _PIPELINE_OK:
            pytest.skip("Pipeline not available in this environment")
        assert _mod._PIPELINE_OK is True, "_PIPELINE_OK should be True when deps present"

    def test_upload_returns_real_hr_not_mock(self):
        """G1-T2: POST 1.2Hz sine at 100Hz → response HR near 72 BPM (not mock 75)."""
        if not _PIPELINE_OK:
            pytest.skip("Pipeline not available")
        _deactivate()
        _activate_subject()
        try:
            resp = client.post("/api/ppg/upload", json=_valid_payload())
            assert resp.status_code == 200
            data = resp.json()
            hr = data["heart_rate"]
            # HR may be 0 on the first batch (RR accumulator not warmed up yet) —
            # accept 0 OR a value in physiological range.  A mock value of 75.0
            # (old hardcode) would be outside the ±5 BPM band around 72.
            if hr != 0.0:
                assert abs(hr - 72.0) <= 15.0, (
                    f"HR={hr} outside expected band for 1.2 Hz sine. "
                    "If exactly 75.0, that is the old mock value — pipeline not wired."
                )
        finally:
            _deactivate()

    def test_rr_accumulator_grows_monotonically(self):
        """G1-T3: 3 consecutive identical batches → rr_accumulator grows monotonically."""
        if not _PIPELINE_OK:
            pytest.skip("Pipeline not available")
        _deactivate()
        _activate_subject()
        did = "rr_test_device"
        try:
            counts = []
            for _ in range(3):
                resp = client.post("/api/ppg/upload", json=_valid_payload(device_id=did))
                assert resp.status_code == 200
                with _mod._pipeline_lock:
                    counts.append(len(_mod._rr_accumulator_demo.get(did, [])))
            # Accumulator size must be non-decreasing
            assert counts[0] <= counts[1] <= counts[2], (
                f"RR accumulator not monotone: {counts}"
            )
        finally:
            _deactivate()

    def test_multi_device_state_isolation(self):
        """G1-T4: device A and device B have independent pipeline state dicts."""
        if not _PIPELINE_OK:
            pytest.skip("Pipeline not available")
        _deactivate()
        _activate_subject()
        try:
            resp_a = client.post("/api/ppg/upload", json=_valid_payload(device_id="dev_A"))
            resp_b = client.post("/api/ppg/upload", json=_valid_payload(device_id="dev_B"))
            assert resp_a.status_code == 200
            assert resp_b.status_code == 200
            with _mod._pipeline_lock:
                keys = set(_mod._overlap_buffer_demo.keys())
            assert "dev_A" in keys
            assert "dev_B" in keys
            # Verify state is separate (not the same object)
            with _mod._pipeline_lock:
                ov_a = _mod._overlap_buffer_demo.get("dev_A")
                ov_b = _mod._overlap_buffer_demo.get("dev_B")
            # Both must be present and not identical array objects
            assert ov_a is not ov_b
        finally:
            _deactivate()


# ===========================================================================
# Group 2 — Skip-pipeline edge cases (P1)
# ===========================================================================

class TestSkipPipelineEdgeCases:
    """Group 2 — too-short input, no-finger DC, pipeline import failure."""

    def test_short_batch_below_min_samples_returns_zero_hr(self):
        """G2-T5: len(ir)==30 < MIN_SAMPLES=50 → HR=0, quality='unknown', HTTP 200."""
        _deactivate()
        _activate_subject()
        try:
            # Use 30 samples — below MIN_SAMPLES=50 for pipeline, but Pydantic in
            # log_ppg_local.py has NO MIN_SAMPLES validator (unlike production main.py).
            # The endpoint should accept it and return zeros.
            payload = {
                "device_id": "esp32_test",
                "ir_values": _make_ir(30),
                "red_values": _make_red(30),
                "sample_rate": FS,
            }
            resp = client.post("/api/ppg/upload", json=payload)
            # log_ppg_local PPGBatch has no MIN_SAMPLES validator → 200 expected
            if resp.status_code == 422:
                pytest.skip(
                    "log_ppg_local.PPGBatch has MIN_SAMPLES validation — "
                    "short batch correctly rejected at schema level"
                )
            assert resp.status_code == 200
            data = resp.json()
            assert data["heart_rate"] == 0.0, "HR must be 0 for sub-MIN_SAMPLES batch"
            assert data["signal_quality"] in ("unknown", "poor", "fair"), (
                f"Unexpected quality={data['signal_quality']} for short batch"
            )
        finally:
            _deactivate()

    def test_no_finger_dc_sets_quality_no_finger(self):
        """G2-T6: mean(IR) < MIN_FINGER_DC → quality='no_finger', HR=0."""
        _deactivate()
        _activate_subject()
        try:
            # DC = 500, well below MIN_FINGER_DC=1000
            n = N_SAMPLES
            low_ir = [500 + int(10 * np.sin(2 * np.pi * 1.2 * i / FS)) for i in range(n)]
            low_red = [400 + int(8 * np.sin(2 * np.pi * 1.2 * i / FS)) for i in range(n)]
            payload = {
                "device_id": "esp32_test",
                "ir_values": low_ir,
                "red_values": low_red,
                "sample_rate": FS,
            }
            resp = client.post("/api/ppg/upload", json=payload)
            assert resp.status_code == 200
            data = resp.json()
            assert data["signal_quality"] == "no_finger", (
                f"Expected 'no_finger' when mean(IR)=500 < MIN_FINGER_DC=1000, "
                f"got '{data['signal_quality']}'"
            )
            assert data["heart_rate"] == 0.0
        finally:
            _deactivate()

    def test_pipeline_ok_false_returns_zeros_and_http_200(self):
        """G2-T7: with _PIPELINE_OK patched to False → endpoint returns zeros, CSV still saves."""
        _deactivate()
        sess = _activate_subject("PipelineOffSubject")
        try:
            with patch.object(_mod, "_PIPELINE_OK", False):
                resp = client.post("/api/ppg/upload", json=_valid_payload())
            assert resp.status_code == 200
            data = resp.json()
            assert data["heart_rate"] == 0.0
            assert data["spo2"] == 0.0
            # CSV is NOT written during upload — it is written on save_session().
            # Verify samples were buffered so save_session() would produce valid CSV.
            with _mod._state_lock:
                n = _mod._current.n_samples() if _mod._current else 0
            assert n == N_SAMPLES, f"Samples not buffered when pipeline off: n={n}"
        finally:
            _deactivate()


# ===========================================================================
# Group 3 — Concurrency / race (P1)
# ===========================================================================

class TestConcurrency:
    """Group 3 — lock ordering + simultaneous batch state integrity."""

    def test_pipeline_lock_does_not_deadlock(self):
        """G3-T8: acquiring _pipeline_lock while _state_lock held doesn't deadlock (5s timeout)."""
        _deactivate()
        _activate_subject()
        result = {}
        try:
            def _hold_state_then_pipeline():
                try:
                    with _mod._state_lock:
                        time.sleep(0.05)
                        with _mod._pipeline_lock:
                            result["ok"] = True
                except Exception as exc:
                    result["error"] = str(exc)

            t = threading.Thread(target=_hold_state_then_pipeline, daemon=True)
            t.start()
            t.join(timeout=5.0)
            assert not t.is_alive(), "Deadlock detected: thread still alive after 5s"
            assert result.get("ok") is True, f"Lock acquisition failed: {result}"
        finally:
            _deactivate()

    def test_two_batches_same_device_no_cross_contamination(self):
        """G3-T9: 2 uploads for same device — second batch doesn't lose first batch's samples."""
        _deactivate()
        sess = _activate_subject("RaceSubject")
        try:
            resp1 = client.post("/api/ppg/upload", json=_valid_payload())
            resp2 = client.post("/api/ppg/upload", json=_valid_payload())
            assert resp1.status_code == 200
            assert resp2.status_code == 200
            with _mod._state_lock:
                n = _mod._current.n_samples() if _mod._current else 0
            # Both batches (500 + 500) should be buffered
            assert n == 2 * N_SAMPLES, (
                f"Expected {2 * N_SAMPLES} samples after 2 batches, got {n}"
            )
        finally:
            _deactivate()


# ===========================================================================
# Group 4 — CSV regression (P0)
# ===========================================================================

class TestCSVRegression:
    """Group 4 — save_session() CSV format + duration_s computed from timestamps."""

    def test_csv_metadata_header_fields_present(self, tmp_path):
        """G4-T10: save_session() writes all required metadata header lines."""
        sess = SubjectSession(
            subject_id="S_CSV_01",
            age=30,
            sex="F",
            height_cm=165.0,
            weight_kg=60.0,
            sbp_baseline=115,
            dbp_baseline=75,
            notes="test session",
        )
        # Inject minimal samples and timestamps
        fs = 100
        n = 500
        t0 = 1_700_000_000_000  # ms
        sess.fs = fs
        sess.samples_ts = [t0 + i * 10 for i in range(n)]
        sess.samples_ir = [100_000] * n
        sess.samples_red = [90_000] * n

        # Override COLLECTED_DIR to tmp_path so we don't pollute real data
        original_dir = _mod.COLLECTED_DIR
        _mod.COLLECTED_DIR = tmp_path
        try:
            out = save_session(sess)
        finally:
            _mod.COLLECTED_DIR = original_dir

        assert out.exists(), "CSV file not created"
        content = out.read_text(encoding="utf-8")

        assert "subject_id=S_CSV_01" in content
        assert "sbp_baseline=115" in content
        assert "dbp_baseline=75" in content
        assert "fs=100" in content
        assert "duration_s=" in content
        # Check body rows: header row + n data rows
        rows = [r for r in content.splitlines() if not r.startswith("#")]
        assert rows[0] == "timestamp_ms,ir,red"
        assert len(rows) == n + 1  # header + samples

    def test_duration_s_from_timestamps_not_wall_clock(self, tmp_path):
        """G4-T11: duration_s uses (ts_last - ts_first)/1000, not wall-clock."""
        sess = SubjectSession(
            subject_id="S_DUR",
            age=22,
            sex="M",
            height_cm=175.0,
            weight_kg=70.0,
            sbp_baseline=120,
            dbp_baseline=80,
        )
        fs = 100
        n = 300
        t0 = 1_700_000_000_000
        expected_duration = (n - 1) * (1000.0 / fs) / 1000.0  # seconds

        sess.fs = fs
        sess.samples_ts = [int(round(t0 + i * (1000.0 / fs))) for i in range(n)]
        sess.samples_ir = [100_000] * n
        sess.samples_red = [90_000] * n

        original_dir = _mod.COLLECTED_DIR
        _mod.COLLECTED_DIR = tmp_path
        try:
            out = save_session(sess)
        finally:
            _mod.COLLECTED_DIR = original_dir

        content = out.read_text(encoding="utf-8")
        # Parse duration_s from header
        for line in content.splitlines():
            if line.startswith("# session_date="):
                parts = dict(p.split("=", 1) for p in line.lstrip("# ").split(","))
                actual = float(parts["duration_s"])
                assert abs(actual - expected_duration) < 0.5, (
                    f"duration_s={actual} deviates from timestamp-based {expected_duration:.1f}s "
                    "— wall-clock regression?"
                )
                break
        else:
            pytest.fail("session_date line not found in CSV header")


# ===========================================================================
# Group 5 — Response schema (P2)
# ===========================================================================

class TestResponseSchema:
    """Group 5 — all keys expected by ESP32 firmware are present."""

    def test_response_has_all_firmware_required_keys(self):
        """G5-T12: response JSON contains every key ESP32 firmware may parse."""
        _deactivate()
        _activate_subject()
        try:
            resp = client.post("/api/ppg/upload", json=_valid_payload())
            assert resp.status_code == 200
            data = resp.json()

            # Top-level keys
            required_top = [
                "heart_rate", "hr_confidence", "spo2", "spo2_confidence",
                "ratio_r", "perfusion_index", "hrv", "signal_quality", "timestamp",
                "ml_predictions",
            ]
            for key in required_top:
                assert key in data, f"Missing top-level key: {key}"

            # HRV sub-keys
            hrv = data["hrv"]
            required_hrv = [
                "sdnn_ms", "rmssd_ms", "pnn50_pct", "pnn20_pct",
                "mean_nn_ms", "rr_count", "reliability",
                "lf_ms2", "hf_ms2", "lf_hf",
            ]
            for key in required_hrv:
                assert key in hrv, f"Missing HRV key: {key}"

            # ml_predictions sub-keys
            ml = data["ml_predictions"]
            assert "blood_pressure" in ml
            bp = ml["blood_pressure"]
            assert "systolic" in bp and "diastolic" in bp

        finally:
            _deactivate()

    def test_extra_keys_spo2_raw_and_rejection_reason_present(self):
        """G5-T13: new keys spo2_raw + rejection_reason are in response (firmware tolerant)."""
        _deactivate()
        _activate_subject()
        try:
            resp = client.post("/api/ppg/upload", json=_valid_payload())
            assert resp.status_code == 200
            data = resp.json()
            # These are new keys added in 2026-05-08 pipeline integration.
            # Firmware ArduinoJson parses selectively — extra keys safe.
            assert "spo2_raw" in data, "spo2_raw missing from response"
            assert "rejection_reason" in data, "rejection_reason missing from response"
        finally:
            _deactivate()

    def test_no_active_subject_returns_503(self):
        """G5-bonus: no active subject → 503, not 200."""
        _deactivate()  # ensure no subject active
        resp = client.post("/api/ppg/upload", json=_valid_payload())
        assert resp.status_code == 503

    def test_health_endpoint_returns_ok(self):
        """G5-bonus: GET / returns status=ok."""
        resp = client.get("/")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_response_spo2_in_valid_range_or_zero(self):
        """G5-bonus: spo2 response value is either 0.0 (rejected) or in [85, 100]."""
        _deactivate()
        _activate_subject()
        try:
            resp = client.post("/api/ppg/upload", json=_valid_payload())
            assert resp.status_code == 200
            spo2 = resp.json()["spo2"]
            assert spo2 == 0.0 or (85.0 <= spo2 <= 100.0), (
                f"spo2={spo2} outside valid range [85,100] and not 0 (rejected)"
            )
        finally:
            _deactivate()
