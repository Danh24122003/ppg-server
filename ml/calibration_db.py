"""
Per-subject BP calibration storage.

Approach: PracticalBP IMWUT 2025 (doi:10.1145/3749486) — 1-reading anchor.
  offset = BP_cuff_real - BP_model_predicted
  predicted_calibrated = model_predicted + offset[device_id]

Lý do tồn tại: model RF/SVR HIỆN TẠI (train trên PPG-BP transmission, deploy
reflectance MAX30102) bị regression-to-mean degeneracy — predict const ~118.3
cho mọi subject. Per-subject offset bù trừ subject-to-subject variance, giảm
MAE từ ~14.7 xuống ~7-9 mmHg expected.

Storage: JSON file đơn giản (thread-safe via threading.Lock), phù hợp thesis demo.
Future work: migrate Firestore khi production.
"""
from __future__ import annotations
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

from loguru import logger

DEFAULT_DB_PATH = Path(__file__).parent / "data" / "calibration.json"

# Sanity bounds — offset không được vượt physiological range
MAX_OFFSET_SBP = 60.0  # mmHg (vd model 118 vs real 60-180 = ±60)
MAX_OFFSET_DBP = 40.0


class CalibrationDB:
    """JSON-backed per-device calibration offsets, thread-safe."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self._lock = threading.Lock()
        self._data: Dict[str, Dict] = {}
        self._load()

    def _load(self) -> None:
        if not self.db_path.exists():
            logger.info("Calibration DB chưa tồn tại — sẽ tạo {} khi save anchor đầu tiên",
                        self.db_path)
            self._data = {}
            return
        try:
            with self.db_path.open("r", encoding="utf-8") as f:
                self._data = json.load(f)
            logger.info("Loaded calibration DB: {} subjects calibrated",
                        len(self._data))
        except (json.JSONDecodeError, OSError) as exc:
            logger.error("Calibration DB corrupt {}: {} — start fresh", self.db_path, exc)
            self._data = {}

    def _persist(self) -> None:
        # Caller phải hold _lock
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.db_path.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.db_path)  # atomic swap

    def save_anchor(
        self,
        device_id: str,
        sbp_real: float,
        dbp_real: float,
        sbp_pred: float,
        dbp_pred: float,
        notes: str = "",
    ) -> Dict:
        """
        Save calibration anchor. Returns the offset record stored.

        offset = real - pred. Future predictions: predicted + offset = real (initially).
        """
        offset_sbp = float(sbp_real) - float(sbp_pred)
        offset_dbp = float(dbp_real) - float(dbp_pred)

        # Sanity check — offset quá lớn = signal/cuff sai, refuse
        if abs(offset_sbp) > MAX_OFFSET_SBP or abs(offset_dbp) > MAX_OFFSET_DBP:
            raise ValueError(
                f"Offset vượt physiological range: sbp_offset={offset_sbp:.1f} "
                f"(max ±{MAX_OFFSET_SBP}), dbp_offset={offset_dbp:.1f} "
                f"(max ±{MAX_OFFSET_DBP}). Kiểm tra cuff reading hoặc PPG signal."
            )

        record = {
            "device_id": device_id,
            "offset_sbp": round(offset_sbp, 2),
            "offset_dbp": round(offset_dbp, 2),
            "anchor_sbp_real": round(float(sbp_real), 1),
            "anchor_dbp_real": round(float(dbp_real), 1),
            "anchor_sbp_pred": round(float(sbp_pred), 1),
            "anchor_dbp_pred": round(float(dbp_pred), 1),
            "anchor_date": datetime.now(timezone.utc).isoformat(),
            "notes": notes,
        }

        with self._lock:
            old = self._data.get(device_id)
            if old:
                logger.info(
                    "Calib OVERWRITE device={} old_offset_sbp={:+.1f} new={:+.1f}",
                    device_id, old["offset_sbp"], record["offset_sbp"]
                )
            else:
                logger.info(
                    "Calib NEW device={} offset_sbp={:+.1f} offset_dbp={:+.1f}",
                    device_id, record["offset_sbp"], record["offset_dbp"]
                )
            self._data[device_id] = record
            self._persist()

        return record

    def get_offset(self, device_id: str) -> Optional[Dict]:
        """Return calibration record dict or None if not calibrated."""
        with self._lock:
            return self._data.get(device_id, None)

    def apply_offset(
        self,
        device_id: str,
        sbp_pred: float,
        dbp_pred: float,
    ) -> Dict[str, float]:
        """
        Apply calibration offset to raw model predictions.

        Returns dict with:
          - sbp_calibrated, dbp_calibrated: corrected values (or raw if no calib)
          - calibrated: True/False
          - anchor_age_days: days since anchor (None if not calibrated)
        """
        rec = self.get_offset(device_id)
        if rec is None:
            return {
                "sbp_calibrated": float(sbp_pred),
                "dbp_calibrated": float(dbp_pred),
                "calibrated": False,
                "anchor_age_days": None,
            }

        try:
            anchor_dt = datetime.fromisoformat(rec["anchor_date"])
            now = datetime.now(timezone.utc)
            age_days = (now - anchor_dt).total_seconds() / 86400
        except (ValueError, KeyError):
            age_days = None

        return {
            "sbp_calibrated": round(float(sbp_pred) + rec["offset_sbp"], 1),
            "dbp_calibrated": round(float(dbp_pred) + rec["offset_dbp"], 1),
            "calibrated": True,
            "anchor_age_days": round(age_days, 1) if age_days is not None else None,
            "offset_sbp": rec["offset_sbp"],
            "offset_dbp": rec["offset_dbp"],
        }

    def list_calibrated(self) -> Dict[str, Dict]:
        """Return shallow copy of all calibration records."""
        with self._lock:
            return dict(self._data)

    def remove(self, device_id: str) -> bool:
        """Remove calibration for a device. Returns True if was present."""
        with self._lock:
            existed = device_id in self._data
            if existed:
                del self._data[device_id]
                self._persist()
                logger.info("Calib REMOVED device={}", device_id)
            return existed


# Module-level singleton for convenience (FastAPI integration)
_default_db: Optional[CalibrationDB] = None


def get_default_db() -> CalibrationDB:
    global _default_db
    if _default_db is None:
        _default_db = CalibrationDB()
    return _default_db
