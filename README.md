# PPG Monitor — Wearable IoT System with Realtime Signal-Quality Assessment

Companion code and thesis for the BME capstone project *"Design and Realtime Quality
Assessment of an IoT-based PPG Wearable System"* (HCMUT, 2026). It contains the embedded
firmware, the two FastAPI cloud back-ends deployed on Render, the final thesis document,
and pointers to the external datasets used.

> **Disclaimer.** Pilot study, **not** a certified medical device (FDA / CE / ISO 81060-2).
> Blood-pressure output is calibration-dependent and for research/education only.

## Repository Layout

```
ppg-server/
├── arduino/        Embedded firmware — NodeMCU-32S (testing) + XIAO ESP32-S3 (target)
├── backend/        Lite FastAPI service — HR, HRV, SpO2, signal-quality
├── ml/             Full FastAPI service — adds BP estimation + per-subject calibration
├── thesis/         Final capstone thesis (.docx)
├── DATABASE.md     Links to the external public datasets (PPG-BP, BIDMC, PPG-DaLiA)
├── render.yaml     Render deployment (two services)
├── LICENSE         MIT
└── README.md       This file
```

The full evaluation / reproduction scripts, regression tests, and generated figures live
on the [`committee-defense`](https://github.com/Danh24122003/ppg-server/tree/committee-defense)
branch. Self-collected cohort recordings and the raw external datasets are **not** in
version control (PII / licence); see [DATABASE.md](DATABASE.md).

## Two-Server Architecture

The cloud pipeline runs as two parallel FastAPI services. The lite service
([`backend/`](backend/)) handles heart rate, heart-rate variability, SpO₂, and
signal-quality assessment; the full service ([`ml/`](ml/)) adds blood-pressure estimation
and per-subject calibration on top of the same lite pipeline. The signal-quality functions
(six raw SQI features, per-beat tSQI, peak detection, two-stage RR rejection, HRV
time/frequency metrics, SpO₂ ratio-of-ratios) are byte-identical between the two services.

| Service | Production URL | Endpoints |
|---------|----------------|----------:|
| Backend (lite) | `https://ppg-backend-udze.onrender.com` | 7 |
| ML (full) | `https://ppg-ml.onrender.com` | 12 |

## Data Flow

```
MAX30102 optical front-end (IR 880 nm + Red 660 nm) @ 100 Hz
        │  HTTP POST every 5 s (500-sample batch)
        ▼
FastAPI cloud back-end
  ├── Pydantic validation
  ├── Butterworth bandpass [0.5, 4.0] Hz
  ├── Six raw SQI features + per-beat tSQI
  ├── Heart rate (HeartPy + two-stage RR rejection)
  ├── HRV (SDNN, RMSSD, pNN50, LF/HF on RR tachogram)
  ├── SpO₂ (ratio-of-ratios with quadratic calibration)
  └── (ML service only) BP (Random Forest / SVR + per-subject calibration)
        │  JSON response
        ▼
Android polling client (HTTP, 5 s)
```

## Quick Start

Two terminals, one per service:

```bash
# Terminal 1 — Backend lite service
cd backend
pip install -r requirements.txt
uvicorn main:app --port 8000
```

```bash
# Terminal 2 — ML full service
cd ml
pip install -r requirements.txt
uvicorn main:app --port 8001
```

Both services are deployed automatically to Render from this branch via `render.yaml`.

## Firmware

See [`arduino/`](arduino/README.md). The empirical results in the thesis were collected on
the NodeMCU-32S; the XIAO ESP32-S3 port (multi-Wi-Fi list, cloud↔local server fallback) is
prepared for the post-defense engineering phase.

## Thesis

The final document is in [`thesis/`](thesis/). It is at the pilot-study stage
(N = 15 unique persons, 24 paired sessions) and is **not** AAMI/ESH/ISO 81060-2 validated;
it sits on the same developmental trajectory as the FDA-cleared cuff-less BP wearables
(Aktiia, Samsung, Biobeat) but at stage-1 cohort scale.

## Datasets

All three training/validation datasets are public — see [DATABASE.md](DATABASE.md) for
official download links and the helper scripts. Raw data is not redistributed here.
