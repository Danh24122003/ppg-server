# PPG Monitor — Báo Cáo Tiến Trình Tổng Thể

> Cập nhật lần cuối: **2026-05-01 (sáng)**  
> Trạng thái: **ACTIVE** — Backend + ML LIVE v4.3, firmware v4.1.0 fs=100Hz, BP bundle Cách 3 fixed, **6/30 paired recordings thu được, BP eval thực tế xác nhận model degenerate** (SBP_pred const 118.3 cho mọi subject — domain gap reflectance vs PPG-BP transmission).

---

## Tổng quan nhanh

| Hạng mục | Trạng thái | Chi tiết |
|----------|-----------|---------|
| Backend Core (FastAPI) | **✅ DONE v4.3** | `main.py` 900 dòng, 181/181 tests pass, Blaney 2024 thresholds |
| ML Pipeline | ✅ DONE + enhanced | RF + SVR train PPG-BP dataset, **vừa merge backend v4.3 HRV logic vào ml/main.py** |
| GitHub Repo | ✅ DONE | Monorepo `backend/` + `ml/` tại `Danh24122003/ppg-server` |
| Render Deploy (ML) | ⏳ Transition v3.0 → v4.3 | `ppg-ml.onrender.com` — code v4.3 ready, chờ push deploy |
| Render Deploy (Backend) | ✅ **LIVE v4.3** | `ppg-backend-udze.onrender.com` |
| ESP32 Firmware (testing) | ✅ **v4.0.9** | NodeMCU-32S Ai-Thinker — fs=50Hz stable, calibration robust. Dùng tạm để test pipeline. |
| **ESP32 Firmware (final)** | ⏳ **CHƯA migrate** | **XIAO ESP32-S3 (Seeed) là hardware cuối cùng cho thesis demo** — cần port v4.0.9 (đổi pin SDA=5, SCL=6, LED=21) |
| Android App | ❓ UNKNOWN | Chưa kiểm tra trong scope này |

**Hardware target (final cho thesis):** **Seeed Studio XIAO ESP32-S3** (LX7 dual-core, 8MB PSRAM, native USB CDC, form factor mini 2×2cm). NodeMCU-32S hiện tại chỉ là module test pipeline — sẽ migrate sang XIAO trước demo cuối.

**Tổng tiến độ: ~93%** (chỉ còn deploy ML v4.3 + **migrate firmware sang XIAO ESP32-S3** + final thesis demo)

---

## Giai đoạn 1 — Core (BACKEND GẦN HOÀN THIỆN)

| Task | Trạng thái | Ghi chú |
|------|-----------|---------|
| HRV time-domain: SDNN + RMSSD + pNN50 + MeanNN | ✅ DONE | pNN50 theo chuẩn Task Force 1996 (chia `len(diff_nn)`) |
| SpO2 piecewise calibration | ✅ DONE | 3 đoạn theo R — tốt hơn đơn tuyến |
| Signal Quality Assessment | ✅ DONE | Welch PSD spectral purity + peak density + amplitude |
| HeartPy adaptive threshold peak detection | ✅ DONE | Overlap buffer 1.5s + RR accumulator + deduplication |
| **LF/HF ratio (frequency-domain HRV)** | ✅ **DONE v4.1** | Cubic spline + Welch PSD, guard ≥60 RR intervals |
| **Endpoint `/api/ppg/stats/{device_id}`** | ✅ **DONE** | HR/SpO2/HRV mean/min/max 24h |
| **pNN20 (threshold 20ms)** | ✅ **DONE v4.3** | Research-backed: phân tách nhóm bệnh tốt hơn pNN50 [Mietus 2002] |
| **Reliability indicator** | ✅ **DONE v4.3** | "low/medium/high" theo rr_count để client biết mức tin cậy |

**Hoàn thành Giai đoạn 1: 8/8** ✅

---

## Giai đoạn 2 — Chất lượng (HOÀN THÀNH)

| Task | Trạng thái | Ghi chú |
|------|-----------|---------|
| **Pytest coverage backend** | ✅ **DONE** | 182 tests (unit + integration + security + concurrent) |
| **Logging (loguru)** | ✅ **DONE v4.1** | 5 log points có cấu trúc |
| **CORS env-based** | ✅ **DONE v4.1** | `ALLOWED_ORIGINS` env var, default `*` cho dev |
| **Authentication (X-Device-Token)** | ✅ **DONE v4.1+v4.2** | Header auth + `hmac.compare_digest` constant-time |
| **Rate limiting (slowapi)** | ✅ **DONE v4.1** | 20/min upload, 60/min GET, 10/min DELETE |
| **Thread safety (3-phase lock + per-device lock)** | ✅ **DONE v4.1+v4.2** | Signal processing ngoài lock, per-device serialization |
| Module tách biệt (models/signal/storage) | ⏳ DEFER | Chưa cần — 900 dòng vẫn dễ đọc |

**Hoàn thành Giai đoạn 2: 6/7** ✅ (1 defer)

---

## Giai đoạn 3 — Nâng cao

| Task | Trạng thái | Ghi chú |
|------|-----------|---------|
| BP model training + deploy ML server | ✅ DONE | RF + SVR train PPG-BP dataset, LIVE tại `ppg-ml.onrender.com` |
| **BP bundle fix Cách 3** | ✅ **DONE 27/4** | `train_ppg_bp.py:629-666` luôn bundle SBP+DBP vào 1 file `random_forest_models.pkl`, auto-cleanup legacy `svr_models.pkl`. Synthetic SBP=118.2, DBP=69.7. |
| **BP eval trên 6 self-collected** | ⚠️ **DEGENERATE 1/5** | SBP_pred = 118.3 const cho cả 6 subject (= training mean). MAE SBP 14.60, ME +10.80; MAE DBP 5.03 (sát AAMI nhờ phân phối hẹp). Confirm cross-domain gap không bridge được nếu không có self-collected reflectance training data. |
| Firebase Firestore thay in-memory | ⏳ CHƯA | Vẫn dùng Python dict (production cần migrate) |
| WebSocket streaming | ⏳ CHƯA | Batch POST 5s vẫn OK cho MVP |
| Dashboard web | ⏳ CHƯA | Android app làm UI chính |
| Deep Learning BP (CNN-LSTM) | ⏳ CHƯA | Cần GPU + PulseDB dataset |
| **Real data validation (Firebase replay)** | ✅ **DONE 2026-04-25** | Backend validate end-to-end với 86s data thật |

**Hoàn thành Giai đoạn 3: 2/6**

---

## Timeline sự kiện quan trọng

### 2026-04-24 (thứ 4)
| Thời gian | Sự kiện |
|-----------|---------|
| 06:47 | Train thành công RF + SVR trên PPG-BP dataset thực (657 files, 219 subjects) |
| 07:00 | Restructure repo → monorepo `backend/` + `ml/` |
| 07:10 | Push GitHub `Danh24122003/ppg-server` |
| 07:20 | Deploy ML server lên Render → `ppg-ml.onrender.com` |
| 07:30 | Test end-to-end ML: HR=71.6 BPM, `classical_ml is_trained=true` |
| Morning | **Backend v4.0** — Review round 1, fix 16 issues (142 tests) |
| Afternoon | **Backend v4.1** — Review round 2, refactor 3-phase lock + 5 features (LF/HF, auth, rate limit, loguru, CORS). 159 tests |
| Evening | **Backend v4.2** — Review round 3, security hardening (hmac.compare_digest, per-device lock, auth-before-ratelimit). 176 tests |

### 2026-04-25 (thứ 5)
| Thời gian | Sự kiện |
|-----------|---------|
| Morning | Research report về pNN50 formula (`docs/pNN50_research_report.md`) |
| Mid-day | **Backend v4.3** — Apply Task Force 1996 formula + pNN20 + reliability indicator. 182 tests |
| Afternoon | **Firebase Replay Validation** — Script `replay_firebase_data.py` chạy real data → 16/16 chunks OK, HR error 0.3 BPM vs FFT |

### 2026-04-26 (thứ 6)
| Thời gian | Sự kiện |
|-----------|---------|
| Morning (~7am) | **Test 1:** Switch firmware URL → backend v4.3, threshold loosen v1 (R 0.3-2.0, PI 0.003). NodeMCU-32S v4.0.9 + Backend v4.3 → fs=50Hz, AC% 0.06-18.94% varied, HR conf 0.55-1.0, HRV rr_count 194-210 reliability=high, SpO2 đôi lúc valid (R 0.4-1.5) |
| Evening (~8:28pm) | **Test 2 (sau threshold tighten Blaney 2024):** SpO2 valid 13/19 batches (68%), median 97.5%, HR conf 0.40-1.0, HRV rr_count progression 32→89 medium→high → **THESIS-READY** |
| Evening | **Threshold tighten evidence-backed:** SPO2_RATIO_MAX 2.0→1.4 (Blaney 2024), SPO2_PI_MIN 0.003→0.002 (Schneider 2024 / JAMA 2024). 181/181 tests pass |
| Evening | **ML server v4.3 merge:** Vừa merge backend v4.3 logic vào `ML code/ml/main.py` (1190 dòng) — đầy đủ HRV (HeartPy, RR accumulator, pNN20, reliability, LF/HF, Blaney thresholds) + ML BP. 10 endpoints, auth + rate-limit applied |

### 2026-04-27 (thứ 7)
- **BP bundle fix Cách 3** — `train_ppg_bp.py` retrain bundle SBP (SVR) + DBP (RF) vào 1 file `random_forest_models.pkl`. Synthetic SBP=118.2, DBP=69.7. CV metrics: MAE SBP 15.41, DBP 9.15.
- **Firmware v4.1.0** — fs=50→100Hz, validated 9 phút streaming, drops=0.

### 2026-04-28 (chủ nhật)
- **Self-collect pipeline READY** — log_ppg_local.py + eval_bp_metrics.py + protocol docs. QA 44/44 pass + 4 P0 bugs fixed. First success Self_test 01 (5min FULL, 30,839 samples).

### 2026-04-29 (thứ 4) — Recruit batch 1
- 5 subject thực thu trong ngày: S001 Danh01 (5:13am), S002 Quoc_01 (10:42am), S005 Minh_01 (14:51pm), S006 Sub_3 (15:08pm), S007 CH_1 (16:26pm). Tổng cộng `collected_data/` có 6 CSV / 6 subject (incl. Self_test 01).

### 2026-05-01 (hôm nay) — Sanity-check ML pipeline
| Thời gian | Sự kiện |
|---|---|
| Sáng | **Tạo `Backend code/self_collect/eval_collected_data.py`** — load EnsemblePredictor (bundle SBP+DBP OK), extract 38 features từ 6 CSV, predict, so cuff baseline. |
| Sáng | **Phát hiện 3 data-quality issues** (xem section "Self-collect data quality" dưới): fs khai vs thực drift (Sub_3 lệch 22%); duration_s metadata vs (ts_last-ts_first)/1000 lệch 19-54%; CSV header `sbp_baseline` chỉ là lần 1 thay vì mean(2,3) theo AHA. |
| Sáng | **BP eval result:** SBP_pred = **118.3 const cho cả 6 subject** (= mean PPG-BP training set) ⇒ degenerate. SBP MAE 14.60, ME +10.80; DBP MAE 5.03 (sát AAMI nhờ phân phối hẹp). Đúng signature regression-to-mean cross-domain transmission→reflectance theo Moulaeifard 2025. |

---

## Self-collect data quality (2026-05-01)

**Inventory `collected_data/` — 6 paired recording (28-29/4):**

| # | Subject | fs khai/thực | Dur meta/thực (s) | n samples | SBP cuff | SBP_pred | DBP cuff | DBP_pred |
|---|---|---|---|---|---|---|---|---|
| 1 | Self_test 01 | 104 / 104.5 ✓ | 362 / 295 (−19%) | 30,839 | 97 | 118.3 | 64 | 70.0 |
| 2 | S001 Danh01 | 104 / 104.2 ✓ | 374 / 302 (−19%) | 31,425 | 106 | 118.3 | 70 | 70.6 |
| 3 | S002 Quoc_01 | 104 / 99.6 | 579 / 321 (−45%) ⚠ | 31,952 | 98 | 118.3 | 68 | 70.9 |
| 4 | S005 Minh_01 | **91** / 88.0 | 462 / 352 (−24%) | 31,000 | 96 | 118.3 | 71 | 73.0 |
| 5 | S006 Sub_3 | 104 / **80.7** ⚠ | 705 / 403 (−43%) ⚠ | 32,500 | 121 | 118.3 | 79 | 72.6 |
| 6 | S007 CH_1 | 104 / 100.3 ✓ | 692 / 318 (−54%) ⚠ | 31,873 | 127 | 118.3 | 83 | 70.7 |

**Aggregate metrics (N=6, vs cuff baseline lần 1):**
- SBP: ME +10.80 mmHg, MAE **14.60**, SD 13.40, range −8.7…+22.3
- DBP: ME −1.20, MAE **5.03**, SD 6.82, range −12.3…+6.0
- HR_pred = 30 cho mọi subject (bundle không train HR → ensemble lower-bound clamp)

**Diagnosis: model degenerate (regression-to-mean)** — SBP_pred const 118.3 = chính giá trị mean của PPG-BP training set. Variance feature từ MAX30102 reflectance gần như **không di chuyển dự đoán** → khớp benchmark cross-dataset calibration-free Moulaeifard et al. 2025 (PPGBP SBP MAE 18.7-25.0). N=6 ≪ N≥85 yêu cầu AAMI ⇒ chỉ là sanity-check.

**3 data-quality issues phải fix trước khi train với self-collected data:**
1. **fs khai báo (104) vs fs thực từ timestamp** — Sub_3 lệch 22%, Quoc_01 lệch 4%. `train_ppg_bp.load_self_collected_features()` resample theo `fs_meta` ⇒ nếu sai sẽ warp tần số. Fix: dùng `fs = (n-1) / ((ts_last-ts_first)/1000)`.
2. **`duration_s` metadata vs `(ts_last-ts_first)/1000`** lệch 19-54% trên 5/6 file — nghi `log_ppg_local.py` ghi wall-clock thay vì stream-active time. Cần audit trước khi train.
3. **CSV header `sbp_baseline` chỉ là lần 1** thay vì `mean(lần 2,3)` theo AHA. Excel mới có 2/3 lần đo. Ground truth đã sai lệch ~3-5 mmHg.

**Coverage gap:** 6 N + 0 E + 0 Stage 1 + 0 Stage 2 (plan: 5+5+5). Cần 9 subject HTN.

---

## Real-World Validation (Firebase replay 2026-04-25)

**Đã validate backend v4.3 với 86 giây PPG data thật từ Firebase Realtime DB.**

| Metric | Kết quả | Ghi chú |
|--------|---------|---------|
| Chunks xử lý OK | 16/16 (100%) | Không crash bất kỳ chunk nào |
| HR median backend | 105.0 BPM | |
| HR FFT independent | 104.7 BPM | Ground truth độc lập |
| **Sai lệch HR** | **0.3 BPM (0.3%)** | ✅ Excellent accuracy |
| Signal quality | 15 good, 1 fair | |
| HRV reliability transition | low → medium → high | Threshold logic đúng |
| Accumulated RR | 127 intervals (86s) | |
| LF/HF | 3.12 | Computed when ≥60 RR |

### Phát hiện quan trọng (cho firmware team)

⚠️ **ESP32 firmware có bug timing:**
- Claim `sample_rate=100` Hz trong field `r`
- Thực tế chỉ ~28.5 Hz (median dt=57ms, không phải 10ms)
- Nguyên nhân: buffering không đều giữa các samples
- **Ảnh hưởng:** Backend bandpass + HeartPy sẽ tính sai HR nếu trust claim 100Hz → cần ESP32 gửi `sample_rate` thực tế

⚠️ **SpO2 bị reject 100%** (ratio_r ngoài range [0.4, 2.0])
- Có thể do chunk ngắn (5.3s @ 28Hz) bandpass filter thiếu "runway"
- Hoặc Red LED firmware không ổn định

---

## Cấu trúc project hiện tại (local)

```
PPG monitor/
├── Android code/
├── Antigravity DX/
├── Arduino code/              ← Firmware ESP32
├── Backend code/               ← ⭐ Focus chính
│   ├── main.py                 # v4.3, 900 dòng, 182 tests pass
│   ├── test_main.py            # 182 tests
│   ├── requirements.txt        # fastapi, heartpy, slowapi, loguru, ...
│   ├── ISSUES.md               # 16 issues v4.0 (đã fix hết)
│   ├── PROGRESS.md             # Progress chi tiết backend
│   ├── replay_firebase_data.py # Script validate với real data
│   ├── replay_firebase_report.md
│   └── Older code/             # Backups qua các version
├── DATA/
│   └── PPG_BP database/        # 657 files training ML
├── ML code/                    # Train scripts (ml/ trên GitHub)
├── Model(Ref)/
├── Paper/                      # References phân loại 3 tier
├── Thesis/
├── docs/
│   └── pNN50_research_report.md  ← Research báo cáo 448 dòng
├── .claude/
│   ├── CLAUDE.md               # Project instructions
│   ├── agents/                 # Custom agent definitions
│   └── progress/               # Progress history
├── PROGRESS.md                 # File này
└── ppg-data-50e8b-default-rtdb-export.json  # Firebase export (369KB)
```

---

## Cấu trúc repo GitHub (deployed)

```
Danh24122003/ppg-server (main)
├── backend/
│   ├── main.py          # ⚠️ Cần sync với local main.py v4.3 (chưa push)
│   ├── test_main.py
│   ├── requirements.txt
│   └── Dockerfile
├── ml/
│   ├── main.py          # LIVE
│   ├── ml_models.py
│   ├── train_ppg_bp.py
│   ├── models/          # RF 2.55MB + SVR 67KB
│   ├── requirements.txt
│   └── Dockerfile
├── render.yaml
└── .gitignore
```

---

## Deploy Status

| Service | URL | Trạng thái |
|---------|-----|-----------|
| `ppg-ml` | `ppg-ml.onrender.com` | ⏳ Transition v3.0 → **v4.3** (code merged, chờ push) |
| `ppg-backend` | `ppg-backend-udze.onrender.com` | ✅ **LIVE v4.3** |

---

## Vấn đề tồn đọng

### Ưu tiên cao
1. **ML v4.3 chưa deploy Render** — code merged xong, chờ push để LIVE thay v3.0
2. **P0 còn 1 bug:** `clear_history` không acquire `device_lock` (race với upload đang ở Phase B) — documented trong `Backend code/PROGRESS.md`

### Ưu tiên trung bình
3. **DB in-memory** — Render restart mất data, cần migrate Firebase Firestore
4. **Sync GitHub:** sync `ml/main.py` v4.3 + `backend/main.py` v4.3 lên repo

### Ưu tiên thấp
5. Dashboard web
6. WebSocket streaming
7. Deep Learning BP (cần GPU)

### Issues đã đóng (mới — 2026-04-26)
- ✅ **SpO2 ratio_r threshold quá rộng** → tightened 2.0 → 1.4 (Blaney 2024 PMC12238718)
- ✅ **PI threshold quá strict** → loosened 0.3% → 0.2% (Schneider 2024 / JAMA 2024 PubMed 38109495)
- ✅ **ML server v3.0 thiếu HRV improvements** → đã merge v4.3 logic vào `ml/main.py`
- ✅ **Calibration finger detection bug v4.0.4** → fix v4.0.8 (CALIB_THRESHOLD vs IR_MIN)
- ✅ **Backend chưa deploy Render** → LIVE v4.3 tại `ppg-backend-udze.onrender.com`
- ✅ **ESP32 firmware bug timing** → v4.0.9 fs=50Hz stable

---

## Testing Strategy

### Tier 1 — Synthetic unit/integration tests ✅
- 182 tests pass
- Sine wave @ 72 BPM, 100Hz, 500 samples
- Coverage: signal processing, API endpoints, auth, rate limit, concurrency

### Tier 2 — Real data replay ✅ (mới!)
- 86s PPG từ Firebase user
- HR error 0.3 BPM vs FFT ground truth
- Không crash, quality assessment đúng

### Tier 3 — Hardware-in-loop ⏳
- Cần ESP32 + MAX30102 thật
- So sánh với oximeter y tế (Masimo, Nonin)
- Scope của đồ án cuối

---

## Disclaimer y tế

> **QUAN TRỌNG:** Dự án này KHÔNG phải thiết bị y tế được chứng nhận (FDA/CE/BYT).  
> Kết quả HR/SpO2/HRV/BP chỉ mang tính tham khảo / học tập.  
> BP accuracy ±8–15 mmHg (vượt chuẩn IEEE cuff-less ±5 mmHg) — phải đính kèm disclaimer trong mọi response.
