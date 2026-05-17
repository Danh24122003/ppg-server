# ML Service — Full Path

FastAPI cloud back-end with the full PPG Monitor pipeline: HR / HRV / SpO₂ (shared with the [`Backend`](../../backend/README.md) lite path) **plus blood-pressure estimation and per-subject calibration**.

Production deployment: `https://ppg-ml.onrender.com`

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/` | Health check + model list |
| `POST` | `/api/ppg/upload` | Full pipeline incl. BP prediction; auto-applies per-subject calibration offset |
| `POST` | `/api/ml/predict` | ML-only prediction (no SQA / no history side-effects) |
| `POST` | `/api/ml/calibrate` | Capture per-subject BP anchor (AC% guard ≥ 0.3%) |
| `GET` | `/api/ml/calibrate/{device_id}` | Read calibration record for one device |
| `GET` | `/api/ml/calibrate` | List all calibrations |
| `DELETE` | `/api/ml/calibrate/{device_id}` | Remove calibration record |
| `POST` | `/api/ml/train` | Re-train BP model on uploaded calibration data |
| `GET` | `/api/ml/models` | Model status |
| (plus the same five SQA/history endpoints as the lite back-end) | | |

## Files

| File | Purpose |
|------|---------|
| `main.py` | FastAPI app, full endpoint set, SQA pipeline (parity with Backend), ML BP prediction wrapper |
| `ml_models.py` | 38-feature extractor (time-domain, frequency-domain, morphological) + RF/SVR ensemble loader |
| `train_ppg_bp.py` | Training script — PPG-BP (Liang 2018) at 1000 Hz polyphase-resampled to 100 Hz + self-collected aggregated cohort. AutoGluon Tabular portfolio with SVR (RBF, ε = 1.5) selected. |
| `calibration_db.py` | Thread-safe JSON store for per-subject BP anchor offsets |
| `test_calibration_endpoints.py` | Integration tests for the four calibration endpoints |
| `models/random_forest_models.pkl` | Production BP regression bundle (N = 234 training rows) |
| `models/random_forest_models.pkl.sha256` | Integrity check for the bundle |
| `experiments/bandpass_widebp/` | Bandpass falsification experiment (Chapter 6 §6.5.1) |

## Local Development

```bash
cd "ml/ml"
python -m venv .venv
.venv\Scripts\activate # Windows
pip install -r requirements.txt
uvicorn main:app --reload --port 8001
```

## Tests

```bash
cd "ml/ml"
pytest -v
```

## Cross-Server Parity

The SQA functions in this `main.py` are byte-identical to the corresponding functions in `backend/main.py`. Twenty of twenty-four common functions match exactly; the remaining four (`_check_device_id`, `_require_token`, `root`, `upload_ppg_data`) differ deliberately to accommodate the additional ML-only endpoints. The parity invariant is asserted by [`tests/test_cross_server_parity.py`](../../tests/test_cross_server_parity.py).

## Model Provenance

The production bundle `random_forest_models.pkl` was retrained on 2026-05-06 on N = 234 training rows: 219 PPG-BP subjects (657 segments at 1000 Hz, polyphase-resampled to 100 Hz to match MAX30102 acquisition) plus 15 self-collected unique subjects aggregated across 21 paired sessions. SHA-256 `63d70043771a05bf857497d10852e4e45c081d48a3d120f4b7598d908e3dd52c`.

## References

- Chapter 5 §5.4.1 — feature engineering (`ml_models.py`)
- Chapter 5 §5.4.2 — transfer learning (`train_ppg_bp.py`)
- Chapter 5 §5.4.3 — leave-one-subject-out evaluation (see also `backend/self_collect/eval_true_loso.py`)
- Chapter 5 §5.4.4 — per-subject calibration drift (`calibration_db.py`)
- Chapter 5 §5.4.5 — industry parallel (Aktiia, Samsung, Biobeat)
- Chapter 6 §6.5.1 — bandpass falsification experiment (`experiments/bandpass_widebp/`)
