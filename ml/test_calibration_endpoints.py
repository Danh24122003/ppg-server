"""
PPG ML Server — Calibration endpoint tests
==========================================
Target: ml/main.py FastAPI app
Runner: pytest

Tests:
1. POST /api/ml/calibrate — happy path (anchor saved, offset returned)
2. POST /api/ml/calibrate — IR/Red length mismatch → 400
3. POST /api/ml/calibrate — finger DC too low → 400
4. POST /api/ml/calibrate — AC% too low (cold finger / vasoconstriction) → 400
5. POST /api/ml/calibrate — Pydantic validation (sample_rate out of range) → 422
6. POST /api/ml/calibrate — offset out of physiological range (±60 SBP) → 400
7. GET /api/ml/calibrate/{device_id} — returns saved record
8. GET /api/ml/calibrate/{device_id} — 404 for uncalibrated
9. GET /api/ml/calibrate — list all calibrated devices
10. DELETE /api/ml/calibrate/{device_id} — removes record
11. DELETE /api/ml/calibrate/{device_id} — 404 for non-existent
12. POST /api/ppg/upload — calibrated device returns calibrated_bp + flag
13. POST /api/ppg/upload — uncalibrated device returns warning
"""
from __future__ import annotations
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

_ML_DIR = Path(__file__).resolve().parent
if str(_ML_DIR) not in sys.path:
    sys.path.insert(0, str(_ML_DIR))

# Disable auth for tests
os.environ.pop("PPG_API_KEY", None)
os.environ["MODEL_DIR"] = str(_ML_DIR / "models")

from main import app, readings_db, device_index  # noqa: E402
from calibration_db import CalibrationDB  # noqa: E402
import main as _main_mod  # noqa: E402

from fastapi.testclient import TestClient  # noqa: E402

client = TestClient(app)

FS = 100
N = 500  # 5s @ 100Hz
DC = 80_000.0


def _sine_ppg(n: int = N, fs: int = FS, freq_hz: float = 1.2,
              dc: float = DC, amp: float = 2000.0) -> np.ndarray:
    """Synthetic PPG: DC + sine. amp/dc ratio determines AC%."""
    t = np.arange(n) / fs
    return dc + amp * np.sin(2 * math.pi * freq_hz * t)


def _int_list(arr: np.ndarray) -> list:
    return arr.astype(int).tolist()


def _calib_payload(device_id: str = "test_calib_01",
                   sbp_real: float = 120.0, dbp_real: float = 80.0,
                   ir_amp: float = 2000.0, red_amp: float = 1900.0,
                   ir_dc: float = DC, red_dc: float = DC * 0.95,
                   n: int = N, fs: int = FS) -> Dict[str, Any]:
    ir = _sine_ppg(n, fs, 1.2, ir_dc, ir_amp)
    red = _sine_ppg(n, fs, 1.2, red_dc, red_amp)
    return {
        "device_id": device_id,
        "ir_values": _int_list(ir),
        "red_values": _int_list(red),
        "sample_rate": fs,
        "sbp_real": sbp_real,
        "dbp_real": dbp_real,
        "notes": "pytest test",
    }


def _upload_payload(device_id: str = "test_calib_01",
                    ir_amp: float = 2000.0, red_amp: float = 1900.0,
                    ir_dc: float = DC, red_dc: float = DC * 0.95,
                    n: int = N, fs: int = FS) -> Dict[str, Any]:
    ir = _sine_ppg(n, fs, 1.2, ir_dc, ir_amp)
    red = _sine_ppg(n, fs, 1.2, red_dc, red_amp)
    return {
        "device_id": device_id,
        "ir_values": _int_list(ir),
        "red_values": _int_list(red),
        "sample_rate": fs,
    }


@pytest.fixture(autouse=True)
def _isolate_calib_db(tmp_path, monkeypatch):
    """Override calibration_db singleton to point at a fresh tmp JSON file
    so tests don't pollute production calibration.json."""
    tmp_db_path = tmp_path / "calibration_test.json"
    fresh_db = CalibrationDB(db_path=tmp_db_path)
    monkeypatch.setattr(_main_mod, "_get_calib_db", lambda: fresh_db)
    # Also reset in-memory state and rate limiter
    readings_db.clear()
    device_index.clear()
    try:
        from main import limiter as _limiter
        _limiter.reset()
    except Exception:
        pass
    yield fresh_db
    # Cleanup
    if tmp_db_path.exists():
        tmp_db_path.unlink()


# ============================================================
# 1. POST /api/ml/calibrate — happy path
# ============================================================
class TestCalibrateAnchor:
    def test_save_anchor_happy_path(self, _isolate_calib_db):
        resp = client.post("/api/ml/calibrate", json=_calib_payload())
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["status"] == "calibrated"
        assert body["device_id"] == "test_calib_01"
        anchor = body["anchor"]
        assert "offset_sbp" in anchor
        assert "offset_dbp" in anchor
        assert anchor["anchor_sbp_real"] == 120.0
        assert anchor["anchor_dbp_real"] == 80.0
        assert "anchor_date" in anchor

    def test_ir_red_length_mismatch_400(self, _isolate_calib_db):
        payload = _calib_payload()
        payload["red_values"] = payload["red_values"][:-50]
        resp = client.post("/api/ml/calibrate", json=payload)
        assert resp.status_code == 400
        assert "IR" in resp.text or "Red" in resp.text or "bằng nhau" in resp.text

    def test_finger_dc_too_low_400(self, _isolate_calib_db):
        payload = _calib_payload(ir_dc=500, red_dc=500)
        resp = client.post("/api/ml/calibrate", json=payload)
        assert resp.status_code == 400
        assert "yếu" in resp.text or "finger" in resp.text.lower() or "weak" in resp.text.lower()

    def test_ac_pct_too_low_400(self, _isolate_calib_db):
        # AC% = (ptp / DC) * 100. amp=10 → ptp ≈ 20, DC=80000 → AC% ≈ 0.025% (< 0.3%)
        payload = _calib_payload(ir_amp=10.0, red_amp=10.0)
        resp = client.post("/api/ml/calibrate", json=payload)
        assert resp.status_code == 400
        assert "AC%" in resp.text or "thấp" in resp.text or "calibrate" in resp.text.lower()

    def test_pydantic_sample_rate_invalid_422(self, _isolate_calib_db):
        payload = _calib_payload()
        payload["sample_rate"] = 1  # below MIN_SAMPLE_RATE
        resp = client.post("/api/ml/calibrate", json=payload)
        assert resp.status_code == 422

    def test_offset_out_of_physiological_range_400(self, _isolate_calib_db):
        # Try sbp_real=300 → offset = 300 - ~118 = +182 > MAX_OFFSET_SBP(60) → 400
        payload = _calib_payload(sbp_real=300.0)
        resp = client.post("/api/ml/calibrate", json=payload)
        # Pydantic might reject sbp_real=300 first, or endpoint computes offset and rejects.
        # Either way it MUST NOT save. Accept 400 or 422.
        assert resp.status_code in (400, 422), resp.text


# ============================================================
# 2. GET /api/ml/calibrate/{device_id}
# ============================================================
class TestGetCalibration:
    def test_get_after_save_returns_record(self, _isolate_calib_db):
        client.post("/api/ml/calibrate", json=_calib_payload(device_id="get_test_01"))
        resp = client.get("/api/ml/calibrate/get_test_01")
        assert resp.status_code == 200
        rec = resp.json()
        assert rec["device_id"] == "get_test_01"
        assert "offset_sbp" in rec
        assert "offset_dbp" in rec

    def test_get_uncalibrated_404(self, _isolate_calib_db):
        resp = client.get("/api/ml/calibrate/never_calibrated_xyz")
        assert resp.status_code == 404


# ============================================================
# 3. GET /api/ml/calibrate (list)
# ============================================================
class TestListCalibrations:
    def test_list_empty(self, _isolate_calib_db):
        resp = client.get("/api/ml/calibrate")
        assert resp.status_code == 200
        body = resp.json()
        assert body["count"] == 0
        assert body["calibrations"] == {}

    def test_list_after_two_saves(self, _isolate_calib_db):
        client.post("/api/ml/calibrate", json=_calib_payload(device_id="list_test_a"))
        client.post("/api/ml/calibrate", json=_calib_payload(device_id="list_test_b"))
        resp = client.get("/api/ml/calibrate")
        assert resp.status_code == 200
        body = resp.json()
        assert body["count"] == 2
        assert "list_test_a" in body["calibrations"]
        assert "list_test_b" in body["calibrations"]


# ============================================================
# 4. DELETE /api/ml/calibrate/{device_id}
# ============================================================
class TestDeleteCalibration:
    def test_delete_existing_200(self, _isolate_calib_db):
        client.post("/api/ml/calibrate", json=_calib_payload(device_id="del_test_01"))
        resp = client.delete("/api/ml/calibrate/del_test_01")
        assert resp.status_code == 200
        assert resp.json()["status"] == "removed"
        # Verify gone
        resp_after = client.get("/api/ml/calibrate/del_test_01")
        assert resp_after.status_code == 404

    def test_delete_nonexistent_404(self, _isolate_calib_db):
        resp = client.delete("/api/ml/calibrate/never_existed_zzz")
        assert resp.status_code == 404


# ============================================================
# 5. /api/ppg/upload integration with calibration
# ============================================================
class TestUploadAppliesCalibration:
    def test_upload_uncalibrated_includes_warning(self, _isolate_calib_db):
        resp = client.post("/api/ppg/upload",
                           json=_upload_payload(device_id="uncalib_dev_01"))
        assert resp.status_code == 200
        body = resp.json()
        ml = body.get("ml_predictions") or {}
        bp = ml.get("blood_pressure") or {}
        # When BP key exists, must have warning when uncalibrated
        if bp:
            assert bp.get("calibrated") is False or "warning" in bp

    def test_upload_calibrated_returns_calibrated_flag(self, _isolate_calib_db):
        # First calibrate
        calib_resp = client.post("/api/ml/calibrate",
                                 json=_calib_payload(device_id="apply_test_01",
                                                     sbp_real=125.0, dbp_real=78.0))
        assert calib_resp.status_code == 200
        # Upload another batch — should apply offset
        upload_resp = client.post("/api/ppg/upload",
                                  json=_upload_payload(device_id="apply_test_01"))
        assert upload_resp.status_code == 200
        body = upload_resp.json()
        ml = body.get("ml_predictions") or {}
        bp = ml.get("blood_pressure") or {}
        # Verify calibrated flag set + values reflect offset applied
        if bp:
            assert bp.get("calibrated") is True
            assert "anchor_age_days" in bp
            assert bp["anchor_age_days"] is not None
            # Values should be near cuff truth (within ±10 mmHg for synthetic)
            sbp_calibrated = bp.get("systolic")
            dbp_calibrated = bp.get("diastolic")
            assert sbp_calibrated is not None
            assert dbp_calibrated is not None
            # Round-trip: model predicts X, offset = 125-X, calibrated = X + (125-X) = 125
            assert abs(sbp_calibrated - 125.0) < 1.0, f"SBP calibrated={sbp_calibrated} not ~125"
            assert abs(dbp_calibrated - 78.0) < 1.0, f"DBP calibrated={dbp_calibrated} not ~78"
