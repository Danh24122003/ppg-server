# PPG Monitor — Wearable IoT System with Realtime Signal-Quality Assessment

This repository accompanies the BME capstone thesis *"Design and Realtime Quality Assessment of an IoT-based PPG Wearable System"* (HCMUT, 2026). It contains the complete software stack — embedded firmware, two FastAPI cloud back-ends, and the supporting evaluation scripts that produced the numerical findings reported in the thesis.

## Repository Layout

```
PPG monitor/
├── arduino/ Embedded firmware for NodeMCU-32S (testing) and XIAO ESP32-S3 (target)
├── backend/ Lite FastAPI service — HR, HRV, SpO2, signal-quality
├── ml/ Full FastAPI service — adds BP estimation and per-subject calibration
├── tests/ Cross-cutting invariants (cross-server parity assertion)
├── LICENSE MIT licence
└── README.md This file
```

Self-collected cohort recordings, the PPG-BP and BIDMC and PPG-DaLiA public datasets, the bound thesis-draft PDFs, and the Pulse-PPG transfer-learning experimental folder are excluded from version control. See `.gitignore` for the complete exclusion list.

## Two-Server Architecture

The cloud-side pipeline runs as two parallel FastAPI services rather than a single monolithic service. The lite service ([`backend/`](backend/README.md)) handles heart rate, heart-rate variability, peripheral oxygen saturation, and signal-quality assessment; the full service ([`ml/`](ml/README.md)) adds blood-pressure estimation and per-subject calibration on top of the shared lite-service pipeline.

The signal-quality-assessment functions — the six raw SQI features, the per-beat template-correlation SQI, the peak detection, the two-stage RR rejection, the HRV time-domain and frequency-domain metrics, and the SpO₂ ratio-of-ratios — are byte-identical between the two services. The parity invariant is asserted by [`tests/test_cross_server_parity.py`](tests/test_cross_server_parity.py); twenty of twenty-four common functions match exactly, and the remaining four (`_check_device_id`, `_require_token`, `root`, `upload_ppg_data`) differ deliberately to accommodate the additional ML-only endpoints.

The motivation for the two-service split is that it matches the production deployment topology on Render and that the parity invariant is enforced rather than assumed: any drift in the SQA pipeline between the two services would fail the cross-server parity test.

| Service | Production URL | Endpoints |
|---------|----------------|----------:|
| Backend (lite) | `https://ppg-backend-udze.onrender.com` | 7 |
| ML (full) | `https://ppg-ml.onrender.com` | 12 |

## Data Flow

```
MAX30102 optical front-end
 IR (880 nm) + Red (660 nm) @ 100 Hz
 │
 │ HTTP POST every 5 s (500-sample batch)
 ▼
 FastAPI cloud back-end
 ├── Pydantic validation
 ├── Butterworth bandpass [0.5, 4.0] Hz
 ├── Six raw SQI features + per-beat tSQI
 ├── Multinomial LR classifier → {excellent, acceptable, unfit}
 ├── Heart rate (HeartPy + two-stage RR rejection)
 ├── HRV (SDNN, RMSSD, pNN50, LF / HF on RR tachogram)
 ├── SpO₂ (ratio-of-ratios with quadratic calibration)
 └── (ML service only) BP (Random Forest / SVR regression + per-subject calibration)
 │
 │ JSON response
 ▼
 Android polling client (HTTP, 5 s)
```

## Quick Start

Two terminals, one per service:

```bash
# Terminal 1 — Backend lite service
cd backend
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --port 8000
```

```bash
# Terminal 2 — ML full service
cd "ml/ml"
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --port 8001
```

The two services are independent and can be started in either order. To exercise the cross-server parity invariant:

```bash
pytest tests/test_cross_server_parity.py -v
```

To exercise the per-service regression suites:

```bash
pytest "Backend code" # 535 tests pass + 2 skipped
pytest "ml/ml" # calibration endpoint integration
```

## Chapter-to-Code Mapping

| Thesis section | Code |
|----------------|------|
| Chapter 3 §3.1–§3.2 — hardware platform and firmware | [`arduino/`](arduino/README.md) |
| Chapter 3 §3.3 — cloud back-end | `backend/main.py`, `ml/main.py` |
| Chapter 3 §3.5 — base DSP pipeline | bandpass + HeartPy + RR rejection in both `main.py` files |
| Chapter 4 §4.1 — six raw batch-level SQI features | `compute_sqi_features` in both `main.py` files |
| Chapter 4 §4.2 — per-beat tSQI | `backend/sqa_per_beat.py` |
| Chapter 4 §4.3 — pseudo-labelling | `backend/generate_pseudo_labels.py` |
| Chapter 4 §4.4 — multinomial LR classifier | `backend/eval_sqa_classifier.py`, `backend/sqa_lr_model.pkl` |
| Chapter 4 §4.5 — cross-dataset evaluation | `backend/eval_sqa_classifier.py` (LODO mode) |
| Chapter 5 §5.1 — HR ablation | `backend/eval_hr_ablation.py` |
| Chapter 5 §5.2 — HRV ablation | `backend/eval_hrv_ablation.py` |
| Chapter 5 §5.3 — SpO₂ ablation | `backend/eval_spo2_ablation.py` |
| Chapter 5 §5.4 — BP module | `ml/ml_models.py`, `ml/train_ppg_bp.py`, `ml/calibration_db.py`, `backend/self_collect/eval_*.py` |
| Chapter 6 §6.5.1 — bandpass falsification | `ml/experiments/bandpass_widebp/` |

## Status

The thesis is at the pilot-study stage (N = 15 unique persons, 24 paired sessions). The repository is **not** AAMI/ESH/ISO 81060-2 validated and the software is **not** FDA, CE, or equivalent regulatory-cleared. The system is positioned on the same developmental trajectory as the FDA-cleared cuff-less BP wearables (Aktiia, Samsung, Biobeat) but at stage-1 cohort scale.

See the LICENCE file for the full statement of warranty and the list of third-party datasets and reference designs that this software interfaces with but does not redistribute.
