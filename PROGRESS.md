# PPG Monitor — ML Code: Bao Cao Tien Trinh

> Cap nhat lan cuoi: 2026-04-22  
> Trang thai: IDLE (file cuoi sua doi cach day 53.1 gio)

---

## Tong quan nhanh

| Hang muc | Trang thai | Chi tiet |
|----------|-----------|---------|
| Feature Extraction (38 features) | OK | Day du 18 time + 8 freq + 12 morph |
| Classical ML (RF/GB/SVR) | PHAN | Code co, model chua train voi data thuc |
| Deep Learning (CNN-LSTM) | PHAN | NumPy simulation co, weights chua train |
| Transformer (Self-Attention) | PHAN | NumPy simulation co, weights chua train |
| Ensemble Predictor | OK | Co logic weighted average, fallback rule-based |
| Train script (PPG-BP dataset) | THIEU | `train_ppg_bp.py` khong ton tai |
| Model files (.pkl / .npz) | THIEU | Thu muc `models/` khong ton tai |
| Tests (pytest) | THIEU | Khong co file test nao |
| Deploy config | OK | Dockerfile + render.yaml co san |

**Tong tien do: ~55% (6/11 hang muc)**

---

## Cau truc thu muc hien tai

```
ML code/
├── main.py              # FastAPI server v3.0 — co san
├── ml_models.py         # Toan bo ML logic — co san
├── requirements.txt     # 6 dependencies — co san
├── Dockerfile           # python:3.11-slim — co san
├── render.yaml          # Render free tier deploy — co san
├── BP_PROJECT_GUIDE.md  # Huong dan train voi PPG-BP dataset — co san
├── CLAUDE.md            # Project instructions — co san
│
├── [THIEU] train_ppg_bp.py        # Training script voi real dataset
├── [THIEU] models/                # Thu muc chua .pkl va .npz
└── [THIEU] data/ppg-bp/           # PPG-BP Dataset (Liang et al. 2018)
```

---

## Phan tich chi tiet tung module

### ml_models.py (1047 dong) — OK

**Feature Extraction — day du**

- `extract_time_domain_features()`: 18 features gom mean, std, skewness, kurtosis, min, max, range,
  rms, avg_ppi, std_ppi, rmssd, pnn50, num_peaks, peak_rate, zero_crossing_rate, slope_mean,
  slope_std, energy
- `extract_frequency_domain_features()`: 8 features — VLF/LF/HF band powers, LF/HF ratio,
  total power, dominant frequency, spectral entropy, bandwidth — dung Welch PSD
- `extract_morphological_features()`: 12 features — systolic amp, rise/fall time, pulse width,
  area ratio, 1st/2nd derivative, waveform symmetry
- `extract_features(ir, red, fs)`: entry point tich hop, tra ve PPGFeatures dataclass

**Model Classes — co khung, chua co weights**

- `ClassicalMLModel`: ho tro RF / GradientBoosting / SVR; co fallback rule-based khi chua train;
  load tu `models/random_forest_models.pkl` (file nay CHUA TON TAI)
- `DeepLearningModel`: numpy-only CNN-LSTM; load tu `models/cnn_lstm_weights.npz` (CHUA TON TAI);
  voi random weights confidence chi dat 0.3
- `TransformerModel`: numpy-only Multi-Head Self-Attention; load tu `models/transformer_weights.npz`
  (CHUA TON TAI); voi random weights confidence 0.25
- `EnsemblePredictor`: chi them DL va Transformer vao ensemble KHI `is_trained == True` — thiet ke
  dung, tranh lam lech ket qua khi weights random

**Training Utilities — chi co synthetic**

- `generate_synthetic_training_data()`: tao PPG gia lap voi harmonic sine, co DC offset, co nhieu
- `train_classical_models()`: train RF/GB/SVR voi sklearn, chay cross-validation 5-fold, luu .pkl
- `train_ppg_bp.py`: KHONG TON TAI — day la missing piece quan trong nhat

**Van de khi chua train:**

Khi khoi dong server, `EnsemblePredictor` chi gom 1 model (`ClassicalMLModel` voi rule-based
fallback). Tat ca du doan BP, stress, AFib dua tren cac cong thuc rule-of-thumb khong co scientific
validation cu the.

### main.py (509 dong) — PHAN

**Endpoints da implement:**

| Endpoint | Trang thai |
|----------|-----------|
| `GET /` | OK |
| `POST /api/ppg/upload` | OK — co ML prediction tich hop |
| `POST /api/ml/predict` | OK — ML-only mode |
| `POST /api/ml/train` | OK — train tren synthetic data |
| `GET /api/ml/models` | OK — liet ke model status |
| `GET /api/ppg/history/{device_id}` | OK |
| `GET /api/ppg/latest/{device_id}` | OK |
| `DELETE /api/ppg/history/{device_id}` | OK |
| `GET /api/ppg/stats/{device_id}` | THIEU — khong co endpoint nay |

**Xu ly tin hieu co san:**

- `bandpass_filter()`: Butterworth [0.5–5.0 Hz], order 4
- `lowpass_filter()`: Butterworth cutoff 0.5 Hz, order 4
- `calculate_heart_rate()`: find_peaks + valid interval filter; tra ve HR + peaks + HRV SDNN
- `calculate_spo2()`: dung cong thuc `110 − 25×R` — KHAC VOI ROADMAP (`104 − 17×R`)
- `assess_signal_quality()`: tinh SNR + BPM range + amplitude — don gian, khong dung SQA pipeline
  tu `ppg_system/pipeline/ppg_sqa.py`

**HRV tren main.py:** Chi tinh duoc SDNN. RMSSD, pNN50, LF/HF KHONG CO trong response
`PPGResult`. Cac gia tri nay co trong `ml_models.py` nhung khong duoc expose qua model response chinh.

---

## Doi chieu voi Roadmap (CLAUDE.md — Giai doan 1 & 2)

### Giai doan 1 — Core

| Task | Trang thai | Ghi chu |
|------|-----------|---------|
| HRV day du: RMSSD + pNN50 + LF/HF | PHAN | ml_models.py co RMSSD/pNN50 trong time features; main.py chi tra SDNN |
| HeartPy adaptive threshold peak detection | CHUA | Dang dung scipy.find_peaks don gian; khong co spline interpolation |
| SpO2 formula: `104 − 17×R` | CHUA | main.py dang dung `110 − 25×R`; ml_models.py dung rule-based estimate |
| SQA pipeline tich hop | CHUA | `assess_signal_quality()` la SNR don gian; khong dung `ppg_sqa.py` |
| Endpoint `/api/ppg/stats/{device_id}` | CHUA | Endpoint nay khong ton tai |

**Hoan thanh Giai doan 1: 0/5**

### Giai doan 2 — Chat luong

| Task | Trang thai | Ghi chu |
|------|-----------|---------|
| Tach module (models, signal_processing, storage) | PHAN | ml_models.py tach rieng tot; main.py van gom signal processing + API + storage |
| Pytest coverage | CHUA | Khong co file test nao |
| Logging (loguru) | CHUA | Dang dung stdlib `logging`; khong co loguru |
| CORS siet lai | CHUA | `allow_origins=["*"]` — mo hoan toan |

**Hoan thanh Giai doan 2: 0/4**

### Giai doan 3 — Nang cao

| Task | Trang thai | Ghi chu |
|------|-----------|---------|
| BP model training + tich hop | PHAN | Khung co, train_ppg_bp.py va data thieu |
| Firebase Firestore | CHUA | Van dung in-memory dict |
| WebSocket streaming | CHUA | Khong co |
| Dashboard web | CHUA | Khong co |

**Hoan thanh Giai doan 3: 0/4**

---

## Files da thay doi gan day

| File | Sua lan cuoi | Noi dung |
|------|-------------|---------|
| CLAUDE.md | 2026-04-20 05:49 | Project instructions va kien truc |
| main.py | 2026-04-19 14:33 | FastAPI server v3.0 voi ML integration |
| ml_models.py | 2026-04-19 14:33 | Toan bo ML pipeline va feature extraction |
| requirements.txt | 2026-04-19 07:25 | 6 dependencies (fastapi, uvicorn, numpy, scipy, pydantic, scikit-learn) |
| render.yaml | 2026-04-19 07:25 | Deploy config |
| Dockerfile | 2026-04-19 07:25 | python:3.11-slim container |
| BP_PROJECT_GUIDE.md | 2026-04-19 07:19 | Huong dan tich hop PPG-BP dataset |

---

## Van de ton dong

### Loi logic / sai lech so voi spec

1. **SpO2 sai cong thuc**: `main.py:222` dang dung `spo2 = 110.0 - 25.0 * R`. Roadmap va paper
   `020024_1_online.pdf` yeu cau `104 − 17×R`. Can sua.

2. **HRV khong day du trong response chinh**: `PPGResult` chi co `hrv_sdnn`; RMSSD, pNN50, LF/HF
   chi co trong output cua `ml_models.py` (duong ML). Nguoi dung goi `/api/ppg/upload` se khong
   nhan duoc HRV day du tru khi doc `ml_predictions`.

3. **SpO2 trong ML rule-based khong chinh xac**: `ml_models.py:447-451` tinh SpO2 tu systolic
   amplitude, khong phai tu ty le IR/Red. Day la approximation rat thu va se sai lech lon.

### File con thieu

4. **`train_ppg_bp.py` khong ton tai**: CLAUDE.md mo ta day du file nay nhung no chua duoc tao.
   Day la buoc bat buoc truoc khi co bat ky ket qua BP co nghia.

5. **Thu muc `models/` khong ton tai**: Khong co model file nao (.pkl, .npz, .pt). Tat ca 3 model
   class deu chay o che do fallback/demo.

6. **PPG-BP Dataset chua tai**: `data/ppg-bp/` khong ton tai. Can tai thu cong tu Figshare.

### Chat luong code

7. **`import json` khong dung** trong `ml_models.py:12` — harmless nhung can don dep.

8. **`fall_times` duoc khai bao va su dung** trong `extract_morphological_features()` nhung
   CLAUDE.md ghi la "unused" — kiem tra lai, co ve CLAUDE.md ghi sai. `fall_times` duoc ghi vao
   `features[3]` va dung tinh waveform symmetry, vay la DUNG DUOC.

9. **Khong co authentication**: endpoint `/api/ml/train` co the bi goi tu bat ky ai co URL.

10. **CORS mo hoan toan**: `allow_origins=["*"]` — can siet khi production.

---

## Buoc tiep theo duoc de xuat

1. **[CAO] Sua SpO2 formula trong `main.py`**: Doi `110.0 - 25.0 * R` thanh `104.0 - 17.0 * R`
   va clip ve `[85.0, 100.0]`. Mot dong sua, anh huong lon den do chinh xac.

2. **[CAO] Tao `train_ppg_bp.py`**: Theo mo ta trong `BP_PROJECT_GUIDE.md` va `CLAUDE.md`. Tai
   PPG-BP Dataset tu Figshare, resample 1000 Hz -> 100 Hz, train RF/GB/SVR, luu vao
   `models/random_forest_models.pkl`.

3. **[CAO] Bo sung HRV day du vao `PPGResult`**: Them `rmssd`, `pnn50`, `lf_power`, `hf_power`,
   `lf_hf_ratio` vao response cua `/api/ppg/upload` — hien tai chi co `hrv_sdnn`.

4. **[TRUNG BINH] Tich hop SQA pipeline**: Import va goi `ppg_sqa.py` tu `ppg_system/pipeline/`
   thay cho `assess_signal_quality()` don gian hien tai.

5. **[TRUNG BINH] Them endpoint `/api/ppg/stats/{device_id}`**: Tinh HR/SpO2 mean/min/max 24h
   tu `readings_db`.

6. **[TRUNG BINH] Viet pytest**: Toi thieu test `extract_features()`, `calculate_heart_rate()`,
   `calculate_spo2()`, `train_classical_models()` voi synthetic data.

7. **[THAP] Don dep `import json` khong dung** trong `ml_models.py`.

8. **[THAP] Them loguru**: Thay `import logging` bang `from loguru import logger`.
