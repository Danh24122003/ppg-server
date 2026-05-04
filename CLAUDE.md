# PPG Monitor — Pipeline Hoàn Chỉnh

> Cập nhật: 2026-05-01 (sáng) | Version: 4.3 backend / **4.1.0 firmware (fs=100Hz, HTTP+HTTPS auto, NTP bypass)**  
> **Hardware target (final): Seeed Studio XIAO ESP32-S3 + MAX30102** | Backend: FastAPI v4.3 (Python 3.11) | App: Android  
> Hardware testing (hiện tại): ESP32 NodeMCU-32S Ai-Thinker — chỉ dùng tạm để test, sẽ migrate sang XIAO ESP32-S3 cho thesis demo cuối cùng  
> **⚠️ BP estimation: bundle Cách 3 ✅ FIXED 27/4. Eval thực N=6 (1/5): SBP MAE 14.60, predict const 118.3 ⇒ degenerate (cross-domain transmission→reflectance gap, đúng Moulaeifard 2025).**  
> **🟡 Self-collect: 6/30 paired recordings (28-29/4). 3 data-quality issues phát hiện 1/5 cần fix trước train.**

---

## Trạng thái triển khai hiện tại

| Service | URL | Trạng thái |
|---------|-----|-----------|
| ML Server | `https://ppg-ml.onrender.com` | **LIVE v4.3** — merge HRV improvements + Blaney thresholds + ML BP (commit `e3f7682`) |
| Backend Server | `https://ppg-backend-udze.onrender.com` | **LIVE v4.3** — HR/HRV/SpO2 không BP (181/181 tests pass) |
| GitHub Repo | `Danh24122003/ppg-server` | Monorepo: `backend/` + `ml/` đều synced v4.3 |
| Firmware (testing) | NodeMCU-32S Ai-Thinker | ✅ **v4.1.0 + 2 fixes** — fs=100Hz validated 27/4 5:40am + HTTP/HTTPS auto-detect (line 780-785) + NTP bypass (line 743). Self-test 28/4 3:02am: 100% batches AC≥0.5%, 5min FULL, 30,839 samples. |
| **Firmware (final target)** | **XIAO ESP32-S3 (Seeed)** | ⏳ **CHƯA migrate** — pin map khác (SDA=5, SCL=6, LED=21), cần port v4.1.0 trước thesis demo |
| Android App | 🟡 v2.0 sync v4.1.0 (chưa flash test) | OkHttp polling `/api/ppg/latest`, full HRV card + Signal Quality badge + confidences. Bỏ Firebase + waveform + BP card. |
| **BP Model bundle** | `ml/models/random_forest_models.pkl` | ✅ **FIXED 27/4** (Cách 3) — bundle cả SBP (SVR) + DBP (RF) trong 1 file. Train MAE: SBP 15.41, DBP 9.15. **Eval thực N=6 (1/5): MAE SBP 14.60, ME +10.80; MAE DBP 5.03; SBP_pred = 118.3 const cho mọi subject ⇒ regression-to-mean.** |
| **Self-collect pipeline** | `Backend code/self_collect/` | ✅ **READY 28/4** — log_ppg_local.py + eval_bp_metrics.py + eval_collected_data.py (1/5) + train_ppg_bp.py + protocol docs. QA 44/44 + 4 P0 bugs fixed. |
| **Self-collect data** | `collected_data/` | 🟡 **6/30 (20% plan)** — 28/4 Self_test 01 + 29/4 5 subject thực (S001 Danh01, S002 Quoc_01, S005 Minh_01, S006 Sub_3, S007 CH_1). Tất cả Normal BP. |
| **Self-collect data quality** | — | ⚠️ **3 ISSUES PHÁT HIỆN 1/5** — fs khai vs thực drift (Sub_3: 80.7 thay vì 104); duration_s meta vs (ts_last-ts_first)/1000 lệch 19-54%; CSV `sbp_baseline` chỉ là lần 1 thay vì mean(2,3) AHA. Phải fix trước train. |
| **Thesis Methodology** | `Thesis/Chapter4_Methodology.md` | ✅ **DONE 28/4** — Skeleton 9 sections + 15 references AHA/ISO/BHS/Moulaeifard et al. |

---

## Tổng quan hệ thống

```
Seeed Studio XIAO ESP32S3 + MAX30102
  IR (880 nm) + Red (660 nm)
        │
        │ HTTP POST (5s batch, 500 mẫu)
        ▼
  ppg-ml.onrender.com  (ML Server — LIVE)
  ├── Validation
  ├── Signal Quality Assessment
  ├── Filtering (Butterworth bandpass)
  ├── HR + HRV (SDNN)
  ├── SpO2
  └── BP (ML — RF model, train PPG-BP dataset thực)
        │
        │ JSON response
        ▼
  Android App (hiển thị + lịch sử)
```

---

## Cấu trúc repo thực tế (GitHub: Danh24122003/ppg-server)

```
ppg-server/
├── backend/
│   ├── main.py          # FastAPI v3.0 — HR/HRV/SpO2 (chưa deploy)
│   ├── requirements.txt
│   └── Dockerfile
├── ml/
│   ├── main.py          # FastAPI v3.0 — signal processing + ML (LIVE)
│   ├── ml_models.py     # 38-feature extraction + RF/SVR ensemble
│   ├── train_ppg_bp.py  # Training script (PPG-BP dataset)
│   ├── models/
│   │   ├── random_forest_models.pkl       # 2.55 MB — train 24/04/2026
│   │   ├── random_forest_models.pkl.sha256
│   │   ├── svr_models.pkl                 # 67 KB
│   │   └── svr_models.pkl.sha256
│   ├── requirements.txt
│   └── Dockerfile
├── render.yaml          # 2 services: ppg-backend + ppg-ml
└── .gitignore

DATA/ (local only — không commit)
└── PPG_BP database/
    ├── Data File/
    │   ├── 0_subject/   ← 657 files .txt (1000Hz PPG)
    │   └── PPG-BP dataset.xlsx
    └── Table 1.xlsx
```

---

## Phân loại tài liệu tham chiếu

### Cốt lõi — `Paper/1_Cot_loi/`

| File | Nội dung | Dùng cho |
|------|----------|----------|
| `kamal1989.pdf` | Nền tảng PPG: dải tần 0.01–15 Hz, AC/DC | Thiết kế bandpass filter |
| `allen2007.pdf` | Review PPG lịch sử, optical principles | Background + thesis |
| `elgendi2012.pdf` | Phân tích waveform PPG: first/second derivative | Signal interpretation |
| `13534_2019_Article_97.pdf` | Review SpO2 + PPG hiện đại, reflection mode | Architecture decision |
| `shaffer2017.pdf` | HRV metrics đầy đủ: SDNN, RMSSD, pNN50, LF/HF | HRV module |
| `vangent2019.pdf` | HeartPy: adaptive threshold, spline, outlier rejection | Peak detection |
| `sensors-22-06054-v3.pdf` | Dilated CNN peak detection khi SNR thấp | Peak detection nâng cao |
| `sensors-22-01389.pdf` | SpO2 bằng spectrophotometry, AC/DC ratio | SpO2 algorithm |
| `020024_1_online.pdf` | Monte Carlo sim → calibration curve 660/890 nm | SpO2 formula calibration |
| `REN_OB1203-Pulse-Ox-Alg_APN_20220425.pdf` | Thuật toán production: Savitzky-Golay, Kalman, SpO2 | SpO2 Engine + HR |
| `Pulse_Oximeter_Manufacturing_Wireless_Telemetry_fo.pdf` | Analog conditioning, FIR, R-ratio hardware | ESP32 hardware design |
| `Development-of-Blood-Oxygen...ESP32.pdf` | Reference implementation ESP32 + MAX30102 đầy đủ | Arduino + Backend |
| `MAX30102_main_reference.pdf` | Sensor-specific algorithms MAX30102/30105 | Sensor configuration |

### Bổ trợ — `Paper/2_Bo_tro/`

| File | Dùng khi nào |
|------|-------------|
| `clifford2012.pdf` | Xây dựng Signal Quality Index (SQI) nâng cao |
| `li2012.pdf` | Multi-metric signal quality + data fusion |
| `millasseau2002.pdf` | Blood pressure từ pulse contour analysis |
| `maeda2008.pdf` | So sánh wavelength IR vs Green |
| `art3A10.10072Fs10916-010-9506-z.pdf` | Thermal stability wavelength |
| `boukhechba2019.pdf` | Motion artifact handling, activity recognition |
| `nihms763449.pdf` | PPG technology evolution context |
| `mhealth-2024-1-e57158.pdf` | PPG wearable + inertial sensor fusion |
| `computers-13-00125.pdf` | ML cho biomedical signals |

### Không liên quan — `Paper/3_Khong_lien_quan/`

| File | Lý do loại |
|------|-----------|
| `677605.pdf` + `oe-16-26-21434.pdf` | Remote/camera PPG — không dùng contact sensor |
| `BM.DH_.2128_OISP-Bachelor-thesis-template.pdf` | Template luận văn trống, không có nội dung kỹ thuật |
| `IoTPhysiotherapy/*.docx` | Project VLTL / Human Pose Estimation — hoàn toàn khác |

---

## Pipeline chi tiết

### Tầng 1 — Thu thập tín hiệu (ESP32 + MAX30102)

```
MAX30102
├── IR LED (880 nm)  ──→ ADC 18-bit ──→ I2C 400kHz
└── Red LED (660 nm) ──→ ADC 18-bit ──→ I2C 400kHz
```

**Config khuyến nghị:**
- Sample rate: 100 Hz
- Pulse width: 411 µs (độ nhạy cao nhất)
- LED current: 6.4–16.0 mA (điều chỉnh theo môi trường)
- ADC range: 16384 (18-bit)

**Logic ESP32:**
1. Buffer 5 giây = 500 mẫu IR + 500 mẫu Red
2. Finger detection: `mean(IR) > 50,000` ADC units (nếu dùng 18-bit full scale)  
   hoặc `mean(IR) > 1,000` (nếu đã scale)
3. Moving average window=5 để loại nhiễu hardware
4. HTTP POST mỗi 5 giây → `/api/ppg/upload`

**Reference:** `Development-of-Blood-Oxygen...ESP32.pdf`, `MAX30102_main_reference.pdf`

---

### Tầng 2 — Validation (Pydantic)

```python
class PPGReading(BaseModel):
    device_id: str          # 1–64 ký tự
    ir_values: List[int]    # 50–10,000 mẫu
    red_values: List[int]   # phải == len(ir_values)
    sample_rate: int        # 25–400 Hz
```

**Kiểm tra thêm trong endpoint:**
- `len(ir) == len(red)` → HTTP 400
- `mean(IR) > 1000` (ngón tay đặt đúng) → HTTP 400
- `mean(RED) > 1000` → HTTP 400

---

### Tầng 3 — Đánh giá chất lượng tín hiệu (SQA)

> **Ground truth: `ml/main.py:656-689` `assess_signal_quality()`** (also identical in `Backend code/main.py:589-623`). Audit 28/4: doc trước đây drift so với code — table dưới giờ đã match code đúng.

Ba tiêu chí kết hợp theo philosophy multi-index của Clifford 2012 (ECG) và Li & Clifford 2012 (PPG); thresholds là empirical project-specific (Elgendi 2016 benchmark là conceptual basis):

| Tiêu chí | Công thức | Điểm |
|----------|-----------|------|
| **Spectral purity** | `P_at_HR_band(0.5–4 Hz) / P_total` từ Welch PSD | ≥ 0.70 → +2 pts; ≥ 0.40 → +1 pt |
| **HR plausibility** | `bpm = peak_count / duration * 60` | 40–180 BPM → +2 pts; 30–200 BPM → +1 pt |
| **Amplitude check** | `ptp(filtered) > 0.01 × mean(\|raw\|)` | True → +1 pt |

**Score aggregation:**
- score ≥ 4 → `"good"` → tiếp tục xử lý
- score ≥ 2 → `"fair"` → tiếp tục xử lý (warn)
- score < 2 → `"poor"` → reject batch

> **Lưu ý:** Code KHÔNG compute SNR(dB) explicitly và KHÔNG dùng `std/mean` cho amplitude. HeartPy `hr_mad` có sẵn nhưng KHÔNG dùng làm gating metric — chỉ là HRV stability indicator.

---

### Tầng 4 — Lọc tín hiệu (Filtering)

**IR channel** (dùng cho HR + HRV):

```python
# Bước 1: Butterworth bandpass [0.5 – 4.0 Hz], order=4
b, a = butter(4, [0.5, 4.0], btype='band', fs=sample_rate)
ir_filtered = filtfilt(b, a, ir_values)

# Bước 2: Savitzky-Golay làm mượt (window=11, poly=3)
from scipy.signal import savgol_filter
ir_smooth = savgol_filter(ir_filtered, window_length=11, polyorder=3)
```

**Red + IR** (dùng cho SpO2):

```python
# AC component (pulse)
b_bp, a_bp = butter(4, [0.5, 4.0], btype='band', fs=sample_rate)
ir_ac  = filtfilt(b_bp, a_bp, ir_values)
red_ac = filtfilt(b_bp, a_bp, red_values)

# DC component (baseline)
b_lp, a_lp = butter(4, 0.5, btype='low', fs=sample_rate)
ir_dc  = filtfilt(b_lp, a_lp, ir_values)
red_dc = filtfilt(b_lp, a_lp, red_values)
```

**Lý do dải tần [0.5–4.0 Hz]:** tương đương 30–240 BPM, loại nhiễu DC và cao tần  
**Reference:** `kamal1989.pdf`, `REN_OB1203.pdf`, `vangent2019.pdf`

---

### Tầng 5A — Heart Rate + HRV

#### Peak Detection (HeartPy algorithm — chưa implement đầy đủ)

```
1. Moving average MA(window = 0.75s)
2. Adaptive threshold = mean(MA) × 1.2
3. scipy.find_peaks(ir_smooth,
       distance  = sample_rate × 0.3,   # max 200 BPM
       prominence = threshold × 0.5)
4. Spline interpolation tại mỗi peak → sub-sample accuracy
5. Outlier rejection: loại RR ngoài ±20% median(RR)
```

> **Hiện tại:** `backend/main.py` dùng `scipy.find_peaks` đơn giản (bước 3 only).  
> Steps 1, 2, 4, 5 chưa implement.

#### Heart Rate

```python
rr_intervals_ms = np.diff(peak_indices) / sample_rate * 1000  # ms
hr_bpm = 60_000 / np.mean(rr_intervals_ms)
# Valid range: 40 – 200 BPM
```

#### HRV Metrics (ref: `shaffer2017.pdf`)

**Time domain** — đã implement trong `backend/main.py`:

```python
sdnn  = np.std(rr_intervals_ms)
rmssd = np.sqrt(np.mean(np.diff(rr_intervals_ms)**2))
diff_nn = np.diff(rr_intervals_ms)
# Task Force 1996 / Ewing 1984: chia (N-1) = len(diff_nn)
pnn50 = np.sum(np.abs(diff_nn) > 50) / len(diff_nn) * 100
pnn20 = np.sum(np.abs(diff_nn) > 20) / len(diff_nn) * 100
mean_nn = np.mean(rr_intervals_ms)
```

**Frequency domain** — đã implement v4.1 (hàm `_compute_lf_hf`):

```python
from scipy.interpolate import interp1d
from scipy.signal import welch
# Cubic spline interpolate RR về lưới đều 4 Hz, sau đó Welch PSD
t_uniform = np.arange(t[0], t[-1], 1.0 / HRV_RESAMPLE_FS)
rr_uniform = interp1d(t, rr_ms, kind="cubic")(t_uniform)
freqs, psd = welch(rr_uniform, fs=4.0, nperseg=256, window="hann")
lf = float(np.trapezoid(psd[(freqs >= 0.04) & (freqs < 0.15)], ...))
hf = float(np.trapezoid(psd[(freqs >= 0.15) & (freqs < 0.40)], ...))
lf_hf = lf / hf  # guard: chỉ tính khi ≥60 RR intervals
```

| Metric | Bình thường | Trạng thái |
|--------|------------|-----------|
| SDNN | 50–100 ms | ✅ DONE v4.0 |
| RMSSD | 20–50 ms | ✅ DONE v4.0 |
| pNN50 | 5–30 % | ✅ DONE v4.3 (formula chuẩn Task Force 1996: chia `len(diff_nn)`) |
| **pNN20** | 15–60 % | ✅ **DONE v4.3** (phân tách nhóm tốt hơn pNN50 [Mietus 2002]) |
| MeanNN | — | ✅ DONE v4.0 |
| **LF/HF** | 1–2 | ✅ **DONE v4.1** (guard ≥60 RR intervals) |
| **reliability** | low/medium/high | ✅ **DONE v4.3** (dựa rr_count) |

---

### Tầng 5B — SpO2

#### Hiện tại trong `backend/main.py` (v2.3 Classical — piecewise)

```python
# AC = RMS sau bandpass
ac_ir  = sqrt(mean(ir_ac**2))
ac_red = sqrt(mean(red_ac**2))

# DC = Hybrid (median raw + mean lowpass) / 2
dc_ir  = (median(ir) + mean(ir_lp)) / 2
dc_red = (median(red) + mean(red_lp)) / 2

R = (ac_red / dc_red) / (ac_ir / dc_ir)

# Validation guards (evidence-backed, applied 2026-04-26):
# R range [0.4, 1.4] — Blaney et al. 2024 (J Biomed Opt 29(S3):S33313, PMC12238718)
#   covers physiological SpO2 70-100% range
# PI ≥ 0.2% — Schneider 2024 (JECCM) + JAMA 2024 (PubMed 38109495)
#   conservative vs clinical concern threshold 0.5-0.6%
if R < 0.4 or R > 1.4: reject "ratio_r_out_of_range"
if PI < 0.002:         reject "low_perfusion_index"

# Piecewise calibration (3 đoạn, tối ưu MAX30102)
if R < 0.7:   spo2 = 105.5 - 17.0 * R
elif R < 1.1: spo2 = 102.0 - 19.5 * R
else:         spo2 = 108.0 - 23.0 * R

spo2 = clip(spo2, 85.0, 100.0)
```

> **Lưu ý:** Công thức piecewise 3 đoạn hiện tại tốt hơn công thức đơn tuyến `104-17×R` trong spec gốc. Giữ nguyên.
> **Update 2026-04-26:** Thresholds tightened theo Blaney 2024 — peer-reviewed measurement xác nhận R ∈ [0.4, 1.4] cho SpO2 70-100%. Test thực tế với firmware v4.0.9: SpO2 valid 68% batches, median 97.5% (healthy young user).

**Perfusion Index (PI):**
```python
pi = (ac_ir / dc_ir) * 100  # % (bình thường: 0.2–20%)
```

---

### Tầng 5C — Blood Pressure (ML — LIVE)

**Dataset:** PPG-BP (Liang et al. 2018) — 219 subjects, 657 recordings  
**Location:** `DATA/PPG_BP database/` (local only)  
**Models:** RF + SVR train 24/04/2026, deploy tại `ppg-ml.onrender.com`

#### Feature Extraction (38 features — `ml/ml_models.py`)

```
Time domain (18):   mean_RR, SDNN, RMSSD, pNN50, mean_amp, std_amp,
                    skewness, kurtosis, PI, peak_count, ...

Frequency (8):      LF, HF, LF/HF, total_power, spectral_entropy,
                    peak_frequency, bandwidth, spectral_centroid

Morphological (12): peak_width, rise_time, fall_time, augmentation_index,
                    dicrotic_notch_ratio, pulse_area, systolic_area,
                    diastolic_area, large_artery_stiffness, SIDVP, ...
```

**Accuracy thực tế:** ±8–15 mmHg (IEEE standard cuff-less: ±5 mmHg)  
**Bắt buộc:** đính kèm disclaimer trong mọi response BP

---

### Tầng 6 — Response JSON

**Backend server** (`backend/main.py` v4.3):
```json
{
  "reading_id": "a1b2c3d4e5f6789012345678901234ab",
  "device_id": "esp32_01",
  "heart_rate": 72.5,
  "hr_confidence": 0.85,
  "spo2": 97.3,
  "spo2_raw": 97.1,
  "spo2_confidence": 0.80,
  "ratio_r": 0.62,
  "perfusion_index": 2.4,
  "hrv": {
    "sdnn_ms": 45.2,
    "rmssd_ms": 38.1,
    "pnn50_pct": 12.4,
    "pnn20_pct": 28.7,
    "mean_nn_ms": 832.0,
    "rr_count": 6,
    "reliability": "low",
    "lf_ms2": null,
    "hf_ms2": null,
    "lf_hf": null
  },
  "signal_quality": "good",
  "timestamp": "2026-04-25T10:30:00+00:00"
}
```
*Note:* `lf_ms2`, `hf_ms2`, `lf_hf` chỉ có giá trị khi rr_count ≥ 60. `reliability`: "low" (<10 RR), "medium" (10-59), "high" (≥60).

**ML server** (`ml/main.py`) — thêm `ml_predictions`:
```json
{
  "heart_rate": 72.5,
  "spo2": 97.3,
  "signal_quality": "good",
  "ml_predictions": {
    "blood_pressure": {"systolic": 120, "diastolic": 80, "unit": "mmHg"},
    "model_used": "classical_ml",
    "confidence": 0.75
  }
}
```

---

### Tầng 7 — API Endpoints

**Backend** (`backend/main.py` — chưa deploy):

| Method | Endpoint | Chức năng |
|--------|----------|-----------|
| `GET` | `/` | Health check |
| `POST` | `/api/ppg/upload` | ESP32 gửi dữ liệu → xử lý → trả kết quả |
| `GET` | `/api/ppg/latest/{device_id}` | Kết quả đo mới nhất |
| `GET` | `/api/ppg/history/{device_id}?limit=20` | Lịch sử đo |
| `GET` | `/api/ppg/stats/{device_id}` | HR/SpO2 mean/min/max 24h — **CHUA CO** |
| `GET` | `/api/stats` | Thống kê toàn hệ thống |
| `DELETE` | `/api/ppg/history/{device_id}` | Xóa lịch sử thiết bị |

**ML** (`ml/main.py` — LIVE tại `ppg-ml.onrender.com`):

| Method | Endpoint | Chức năng |
|--------|----------|-----------|
| `GET` | `/` | Health check + model list |
| `POST` | `/api/ppg/upload` | Full pipeline + ML prediction |
| `POST` | `/api/ml/predict` | ML-only prediction |
| `POST` | `/api/ml/train` | Train trên synthetic data |
| `GET` | `/api/ml/models` | Trạng thái models |
| `GET` | `/api/ppg/history/{device_id}` | Lịch sử đo |
| `GET` | `/api/ppg/latest/{device_id}` | Kết quả mới nhất |
| `DELETE` | `/api/ppg/history/{device_id}` | Xóa lịch sử |

---

## Roadmap triển khai

### Giai đoạn 1 — Ổn định core ✅ HOÀN THÀNH

- [x] HRV time-domain: SDNN + RMSSD + pNN50 + MeanNN
- [x] SpO2 piecewise calibration (3 đoạn)
- [x] Signal Quality Assessment (Welch PSD spectral purity)
- [x] **HeartPy adaptive threshold peak detection** (v4.0, overlap buffer + RR accumulator + deduplication)
- [x] **LF/HF ratio** (v4.1, cubic spline + Welch PSD, guard ≥60 RR)
- [x] **Endpoint `/api/ppg/stats/{device_id}`** (HR/SpO2/HRV 24h)
- [x] **pNN20** (v4.3, phân tách nhóm tốt hơn pNN50)
- [x] **reliability indicator** (v4.3, low/medium/high)

### Giai đoạn 2 — Chất lượng ✅ HOÀN THÀNH (6/7)

- [ ] Tách `main.py` thành modules (defer — 900 dòng vẫn đọc được)
- [x] **Pytest backend** (182 tests, 28 classes, synthetic + concurrent + security)
- [x] **Logging (loguru)** (v4.1, 5 log points)
- [x] **CORS env-based** (v4.1, `ALLOWED_ORIGINS` env var)
- [x] **Authentication** (v4.1+v4.2, `X-Device-Token` + `hmac.compare_digest` constant-time)
- [x] **Rate limiting** (v4.1, slowapi: 20/60/10 per minute)
- [x] **Thread safety** (v4.1+v4.2, 3-phase lock + per-device lock + race-free refactor)

### Giai đoạn 3 — Tính năng nâng cao

- [x] Train BP model với PPG-BP Dataset → deploy ML server
- [x] **Real-data validation** (v4.3, Firebase replay 86s data, HR error 0.3 BPM vs FFT)
- [x] **BP bundle Cách 3 fix** (27/4) — `random_forest_models.pkl` chứa cả SBP+DBP
- [x] **Self-collect pipeline + protocol docs** (28/4) — log_ppg_local.py, eval_bp_metrics.py, QUY_TRINH_THU_MAU
- [x] **Recruit batch 1** (29/4) — 5 subject thực thu trong 1 ngày + Self_test = 6/30 (20% plan)
- [x] **Sanity-check ML eval N=6** (1/5) — `Backend code/self_collect/eval_collected_data.py`. SBP_pred const 118.3 ⇒ degenerate; MAE 14.60.
- [ ] **Audit `log_ppg_local.py`** — duration_s meta vs timestamp range lệch 19-54%
- [ ] **Sửa CSV header** sinh `sbp_baseline_mean = mean(reading_2, 3)` thay vì lần 1
- [ ] **Sửa `train_ppg_bp.load_self_collected_features()`** dùng fs từ timestamp
- [ ] **Tuyển 9 subject còn thiếu** (ưu tiên Elevated/Stage 1/Stage 2 HTN)
- [ ] **Đo session 2** cho 6 subject hiện có
- [ ] **Train AutoGluon + per-subject calibration** sau khi đủ ≥20 paired
- [ ] **Deploy bundle pkl mới** lên Render
- [ ] Firebase Firestore thay in-memory dict
- [ ] WebSocket streaming thay batch POST
- [ ] Dashboard web xem lịch sử
- [ ] Deep Learning BP (CNN-LSTM, cần GPU + PulseDB)
- [ ] **Fix P0:** `clear_history` acquire `device_lock` (1 bug còn sót từ review round 3)
- [x] ~~**Fix firmware ESP32:** sample_rate claim sai (100Hz vs thực tế 28.5Hz)~~ ✅ DONE — v4.0.9 stable fs=50Hz, sau đó v4.1.0 đẩy lên fs=100Hz validated 27/4 (drops=0 trong 9 phút streaming)

---

## Hạn chế đã biết

| Hạn chế | Mức độ | Trạng thái |
|---------|--------|-----------|
| DB in-memory (mất khi Render restart) | Cao | ⏳ Cần migrate Firebase/PostgreSQL |
| BP accuracy ±8–15 mmHg | Cao (inherent) | ✅ Disclaimer bắt buộc |
| ~~Không có authentication~~ | ~~Trung bình~~ | ✅ FIXED v4.1 (X-Device-Token + hmac) |
| ~~CORS mở hoàn toàn~~ | ~~Thấp~~ | ✅ FIXED v4.1 (env-based ALLOWED_ORIGINS) |
| ~~Không có rate-limit~~ | ~~Trung bình~~ | ✅ FIXED v4.1 (slowapi 20/60/10 per minute) |
| ~~Không có tests backend~~ | ~~Cao~~ | ✅ FIXED (182 tests pass) |
| BP không thực tế với synthetic data | Expected | ✅ Ready khi có ESP32 data |
| ~~**ESP32 firmware sample_rate bug**~~ | ~~Trung bình~~ | ✅ FIXED v4.0.9 (timing bug → fs=50Hz stable) → v4.1.0 (fs=100Hz validated 27/4) |
| **SpO2 reject với chunks ngắn** | Trung bình | ⚠️ Cần điều tra với ESP32 data ổn định |
| **`clear_history` race condition** | P0 | ⏳ 1 fix nhỏ trước deploy |

---

## Real-World Data Validation (2026-04-25)

Đã validate backend v4.3 với data PPG thực tế từ Firebase export (`ppg-data-50e8b-default-rtdb-export.json`):

| Metric | Kết quả |
|--------|---------|
| Data source | 2450 samples × 86s (Firebase Realtime DB) |
| Chunks xử lý OK | 16/16 (100%) |
| HR backend median | 105.0 BPM |
| HR FFT independent | 104.7 BPM |
| **Sai lệch** | **0.3 BPM (0.3%)** ✅ |
| Accumulated HRV | 127 RR intervals, reliability=high |

Script: `Backend code/replay_firebase_data.py`  
Báo cáo: `Backend code/replay_firebase_report.md`

### Live ESP32 Test (2026-04-26 → 2026-05-01)

| Test | Firmware | Result |
|---|---|---|
| 26/4 7am — backend v4.3 switch | v4.0.9 fs=50Hz | stable, HR conf 0.55-1.0, HRV rr_count 194-210, reliability=high |
| 26/4 8:28pm — Blaney thresholds | v4.0.9 fs=50Hz | SpO2 valid 13/19 (68%), median 97.5% |
| 26/4 9:20pm — ngón cái + ML server v4.3 | v4.0.9 fs=50Hz | **20/20 SpO2 valid (100%)**, median 98%, HR conf 0.95-1.00 |
| **27/4 5:40am — fs=100Hz upgrade** ⭐ | **v4.1.0 fs=100Hz** | **108 batches/9 phút, drops=0 fails=0**, SpO₂ valid 99/109 (90.8%), HRV reliability=high 102/109 (93.6%), Quality good 107/109 (98.2%), heap stable 143KB |
| **28/4 3:02am — Self_test 01 SUCCESS** ⭐⭐ | v4.1.0 + 2 fixes | 5 phút FULL, 30,839 samples, **100% batches AC ≥ 0.5%**, BP_baseline 97/64, CSV saved |
| **29/4 — Recruit batch 1** | v4.1.0 | 5 subject (S001/S002/S005/S006/S007) thu trong 1 ngày, BP 96-127/64-83, fs drift Sub_3 còn 80.7Hz, dur gap 19-54% |
| **1/5 — Eval N=6 self-collected** | — | SBP_pred = 118.3 const ⇒ regression-to-mean. MAE SBP 14.60 / DBP 5.03 |

---

## ⚠️ ML BP Issues — UPDATED 2026-05-01 (eval thực N=6)

### Tóm tắt status hiện tại

| Vấn đề | Trạng thái |
|---|---|
| ~~BP=0/0 (clamp bug)~~ | ✅ **FIXED 27/4 Cách 3** — bundle SBP+DBP vào 1 file `random_forest_models.pkl` |
| ~~Sample rate mismatch (50Hz vs 100Hz)~~ | ✅ **FIXED 27/4 v4.1.0** — user stream 100Hz match training |
| **Model degenerate (predict const)** | ⚠️ **CONFIRMED 1/5** — SBP_pred = 118.3 cho cả 6 subject (= mean PPG-BP). MAE SBP 14.60, ME +10.80; MAE DBP 5.03 / ME −1.20 (sát AAMI nhờ phân phối hẹp). HR_pred = 30 do bundle không train HR. |
| Reflectance vs Transmission gap | 🟡 IN PROGRESS — cần ≥20 paired self-collected reflectance + per-subject calibration |
| Dataset 219 subjects nhỏ | 🟡 IN PROGRESS — đang ở 6/30 paired recordings |
| Self-collect data quality | ⚠️ NEW 1/5 — 3 issues (fs khai vs thực, duration_s gap, sbp_baseline lần 1 vs mean(2,3)) phải fix trước train |

### Root cause (từ debug 27/4 + eval 1/5)
1. ✅ ~~`random_forest_models.pkl` chỉ chứa DBP~~ — đã bundle Cách 3.
2. ⚠️ Model rơi về **regression-to-mean** trên reflectance MAX30102: variance feature gần như không di chuyển dự đoán. Đúng signature cross-domain transmission→reflectance gap.
3. **3 nguyên nhân gốc rễ còn lại:**
   - Reflectance vs Transmission domain gap (PPG-BP là transmission, MAX30102 là reflectance) — chính
   - Dataset nhỏ (219 subjects, BP regression cần ≥5000)
   - Subject hiện tại tất cả Normal BP 96-127/64-83 ⇒ 0% coverage Elevated/Stage 1/Stage 2 ⇒ training-time variance còn hẹp hơn nữa

### Approach để cải thiện (đang triển khai)

| Bước | Status | Expected MAE |
|---|---|---|
| Bundle Cách 3 (cho có SBP) | ✅ DONE 27/4 | SBP 15.4 (CV) |
| Self-collect 30 paired reflectance | 🟡 6/30 (20%) | n/a (chưa train) |
| AutoGluon Tabular + per-subject calibration | ⏳ pending data | ~7-10 mmHg expected |
| Pre-train PulseDB + fine-tune reflectance | ⏳ future work | ~5-8 mmHg target |

### Important context — Đây là known limitation toàn ngành

Theo Moulaeifard, Charlton & Strodthoff 2025 (arXiv:2502.19167) — *"Generalizable deep learning for photoplethysmography-based blood pressure estimation — a benchmarking study"*:
> "Five DL architectures (LeNet1D, XResNet1d50, XResNet1d101, Inception1D, S4) train trên PulseDB → test calibration-free trên external datasets (Sensors, UCI, PPGBP) đều có **SBP MAE ~15–25 mmHg**."

(Số chi tiết Table V: Sensors 18.45–21.15, UCI 21.22–25.05, PPGBP 18.69–25.03 mmHg. BCG dataset thấp hơn 10–18 mmHg; MIMIC-trained models cao hơn 33–44 mmHg.)

→ **MAE 15.4 mmHg KHÔNG phải implementation failure** — là kết quả điển hình của cuff-less BP estimation reflectance không có user calibration. Cite paper này làm justification trong thesis.

> **Citation audit 2026-04-28 round 2:** Note — researcher agent verify lại lần 2: tên tác giả "Nabian" sai (đúng là Moulaeifard/Charlton/Strodthoff); architectures Transformer/U-Net/BiLSTM cũng sai (paper benchmark LeNet1D + 2 XResNet1d + Inception1D + S4). Đã sửa.

### Decision cho thesis demo

**Demo metrics:**
- ✅ HR (62-95 BPM, conf 0.95-1.00)
- ✅ HRV (SDNN, RMSSD, pNN50, pNN20, LF/HF, reliability)
- ✅ SpO2 (97-98%, valid 100% với placement đúng)
- ❌ BP (document làm future work với citation Moulaeifard et al. 2025)

---

## Scope chính thức — Thesis MVP

Project này là **luận văn tốt nghiệp**, KHÔNG phải production product.
Các quyết định scope sau đây là **CỐ Ý**, không phải thiếu sót.
Reviewers / agents tương lai KHÔNG cần flag những điểm trong "OUT OF SCOPE" lặp lại.

### IN SCOPE
- Firmware ESP32 ổn định cho demo (không crash, không mất data trong 5-10 phút demo)
- Backend signal processing chính xác (HR ±5%, SpO2 ±3%, HRV time-domain, BP ±10mmHg)
- Authentication ở backend layer (X-Device-Token + hmac.compare_digest)
- Real-data validation (đã có 86s Firebase replay, HR error 0.3 BPM)
- Documentation đầy đủ cho hội đồng bảo vệ

### OUT OF SCOPE — không cần fix khi review
- **TLS cert pinning** — `WiFiClientSecure.setInsecure()` OK cho lab WiFi, Render auto-rotate cert
- **NVS encryption** — API key plaintext NVS OK, không production
- **Offline buffering / persistent retry** — lab WiFi ổn định, drop khi mất mạng OK
- **Adaptive thresholds** — fixed values từ datasheet + Liang 2018 đã cover 95% case
- **Memory pool / static buffers** — malloc/free pattern hiện tại đủ ổn định cho 5s/batch
- **Task notification thay polling** — CPU cost của polling 20Hz là 0.02%, không đáng refactor
- **Multi-device load test** — 1 device cho thesis demo
- **HIPAA / FDA / CE compliance** — disclaimer y tế đã có

### Threat model
- Lab demo trong môi trường WiFi cô lập, không có adversary
- Hardware không bị ăn cắp / dump flash
- Backend deploy Render free tier, accept rủi ro free-tier (cold start, restart mất in-memory data)

---

> **Disclaimer:** Dự án này KHÔNG phải thiết bị y tế được chứng nhận (FDA/CE/BYT).  
> Kết quả chỉ mang tính tham khảo / học tập.  
> BP accuracy ±8–15 mmHg (vượt chuẩn IEEE ±5 mmHg) — phải đính kèm disclaimer trong mọi response.
