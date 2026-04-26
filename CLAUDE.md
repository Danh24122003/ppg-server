# PPG Monitor — Pipeline Hoàn Chỉnh

> Cập nhật: 2026-04-26 (evening) | Version: 4.3  
> **Hardware target (final): Seeed Studio XIAO ESP32-S3 + MAX30102** | Backend: FastAPI v4.3 (Python 3.11) | App: Android  
> Hardware testing (hiện tại): ESP32 NodeMCU-32S Ai-Thinker — chỉ dùng tạm để test, sẽ migrate sang XIAO ESP32-S3 cho thesis demo cuối cùng

---

## Trạng thái triển khai hiện tại

| Service | URL | Trạng thái |
|---------|-----|-----------|
| ML Server | `https://ppg-ml.onrender.com` | **LIVE v4.3** — merge HRV improvements + Blaney thresholds + ML BP (commit `e3f7682`) |
| Backend Server | `https://ppg-backend-udze.onrender.com` | **LIVE v4.3** — HR/HRV/SpO2 không BP (181/181 tests pass) |
| GitHub Repo | `Danh24122003/ppg-server` | Monorepo: `backend/` + `ml/` đều synced v4.3 |
| Firmware (testing) | NodeMCU-32S Ai-Thinker | **v4.0.9** — fs=50Hz, sensor recovery, calibration robust. Đang dùng để test pipeline. |
| **Firmware (final target)** | **XIAO ESP32-S3 (Seeed)** | ⏳ **CHƯA migrate** — pin map khác (SDA=5, SCL=6, LED=21), cần port v4.0.9 trước thesis demo |

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

Ba tiêu chí kết hợp (ref: `clifford2012.pdf`, `li2012.pdf`):

| Tiêu chí | Công thức | Good | Fair | Poor |
|----------|-----------|------|------|------|
| SNR | `20·log10(P_signal / P_noise)` dB | ≥ 10 dB | ≥ 5 dB | < 5 dB |
| Spectral purity | `P_at_fHR / P_total` (trong 0.5–4 Hz) | ≥ 0.70 | ≥ 0.40 | < 0.40 |
| Amplitude ratio | `std(signal) / mean(signal)` | 0.01–0.15 | 0.005–0.20 | ngoài range |

**Kết quả:**
- `"good"` hoặc `"fair"` → tiếp tục xử lý
- `"poor"` → trả về lỗi ngay, yêu cầu đo lại

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
pnn50 = np.sum(np.abs(np.diff(rr_intervals_ms)) > 50) / len(rr_intervals_ms) * 100
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
- [ ] Firebase Firestore thay in-memory dict
- [ ] WebSocket streaming thay batch POST
- [ ] Dashboard web xem lịch sử
- [ ] Deep Learning BP (CNN-LSTM, cần GPU + PulseDB)
- [ ] **Deploy backend lên Render** (code ready, chưa deploy)
- [ ] **Fix P0:** `clear_history` acquire `device_lock` (1 bug còn sót từ review round 3)
- [ ] **Fix firmware ESP32:** sample_rate claim sai (100Hz vs thực tế 28.5Hz)

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
| **ESP32 firmware sample_rate bug** | **Trung bình** | ⚠️ Phát hiện 2026-04-25: claim 100Hz nhưng thực tế 28.5Hz |
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