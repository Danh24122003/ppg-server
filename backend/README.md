# Backend Service — Lite Path

FastAPI cloud back-end for the PPG Monitor wearable system. This is the **lite path**: heart rate, heart-rate variability, and SpO₂. Blood-pressure estimation is handled by the separate [`ML/`](../ml/README.md) service.

Production deployment: `https://ppg-backend-udze.onrender.com`

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/` | Health check |
| `POST` | `/api/ppg/upload` | Accept 5-second PPG batch from ESP32; return HR / HRV / SpO₂ / signal-quality |
| `GET` | `/api/ppg/latest/{device_id}` | Most-recent reading per device |
| `GET` | `/api/ppg/history/{device_id}?limit=20` | Reading history |
| `GET` | `/api/ppg/stats/{device_id}` | 24-hour aggregate statistics |
| `GET` | `/api/stats` | Cross-device statistics |
| `DELETE` | `/api/ppg/history/{device_id}` | Clear device history |

## Files

| File | Purpose |
|------|---------|
| `main.py` | FastAPI app, all endpoints, signal-processing pipeline |
| `sqa_per_beat.py` | Per-beat tSQI module (Li & Clifford 2012 + Orphanidou 2015) |
| `generate_pseudo_labels.py` | Orphanidou 2015 four-rule pseudo-label generator |
| `eval_sqa_classifier.py` | Multinomial LR classifier training + cross-validation |
| `eval_hr_ablation.py` | HR MAE per SQA tier (Chapter 5 §5.1) |
| `eval_hrv_ablation.py` | HRV SDNN/RMSSD per SQA tier (Chapter 5 §5.2) |
| `eval_spo2_ablation.py` | SpO₂ validity per SQA tier (Chapter 5 §5.3) |
| `eval_bp_ablation.py` | BP confounder analysis (Chapter 5 §5.4) |
| `test_*.py` | Unit + integration tests (≥ 200 tests pass) |
| `self_collect/` | BP-protocol acquisition + LOSO + path-B calibration tooling |
| `sqa_lr_model.pkl` | Production-grade SQA classifier (12,603 windows / 54 subjects) |
| `pseudo_labels.csv` | Pseudo-label corpus used by `eval_sqa_classifier.py` |

## Local Development

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate # Windows
# source .venv/bin/activate # Linux/macOS
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

## Tests

```bash
cd backend
pytest -v
```

535 backend tests pass + 2 skipped (cross-server byte-identical SQA functions verified against the ML service).

## Cross-Server Parity

The signal-quality-assessment functions (`compute_sqi_features`, `assess_signal_quality`, peak detection, RR rejection, SDNN/RMSSD computation, SpO₂ ratio-of-ratios) are byte-identical between this back-end and the ML service. The shared invariant is asserted by [`tests/test_cross_server_parity.py`](../tests/test_cross_server_parity.py) at the repository root.

The endpoints `_check_device_id`, `_require_token`, `root`, and `upload_ppg_data` differ deliberately between the two services because the ML service adds BP prediction and calibration endpoints on top of the shared SQA pipeline.

## Docker

```bash
cd backend
docker build -t ppg-backend .
docker run -p 8000:8000 ppg-backend
```

## References

Detailed methodology is documented in the thesis chapters; the chapter-to-code mapping is:

- Chapter 3 §3.5 — base DSP pipeline (`main.py` bandpass + HeartPy + RR rejection)
- Chapter 4 §4.1 — six raw SQI features (`compute_sqi_features` in `main.py`)
- Chapter 4 §4.2 — per-beat tSQI (`sqa_per_beat.py`)
- Chapter 4 §4.3 — pseudo-labelling (`generate_pseudo_labels.py`)
- Chapter 4 §4.4 — LR classifier (`eval_sqa_classifier.py`, `sqa_lr_model.pkl`)
- Chapter 5 §5.1 – §5.4 — downstream module ablations (`eval_*_ablation.py`)
