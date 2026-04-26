# Kế Hoạch Triển Khai Hệ Thống ML Dự Đoán Huyết Áp

> Cập nhật: 2026-04-24 — **Tất cả 4 bước đã hoàn thành**

## Mục Tiêu

Xây dựng model ML từ tín hiệu PPG (ESP32 + MAX30102) dự đoán **huyết áp tâm thu (SBP)** và **huyết áp tâm trương (DBP)** không xâm lấn, không cần cuff.

## Dataset

**PPG-BP Database** (Liang et al., 2018 — Scientific Data)
- 219 subjects, tuổi 20-89, có cả bình thường và có bệnh tim mạch
- 3 recordings × 2.1 giây @ 1kHz cho mỗi người = 657 recordings
- Single-channel finger PPG (đúng form factor MAX30102)
- Labels: SBP, DBP, HR, age, sex, BMI, hypertension stage, diabetes, CVD

**Vị trí dataset:** `C:\Users\Acer\Desktop\PPG monitor\DATA\PPG_BP database\`

## Các Bước

### Bước 1: Tải dataset ✅ HOÀN THÀNH

Dataset đã tải tại:
```
DATA/PPG_BP database/
├── Data File/
│   ├── 0_subject/     ← 657 files .txt
│   └── PPG-BP dataset.xlsx
└── Table 1.xlsx
```

### Bước 2: Cài thư viện bổ sung ✅ HOÀN THÀNH

```bash
pip install pandas openpyxl scikit-learn
```

Đã có trong `requirements.txt`.

### Bước 3: Chạy training ✅ HOÀN THÀNH (24/04/2026 06:47)

```bash
cd "ML code"
python ml/train_ppg_bp.py
```

**Kết quả thực tế:**
- Load 219 subjects từ Excel
- Xử lý 657 recordings, resample 1000Hz → 100Hz
- Train RF + SVR (5-fold GroupKFold CV)
- RF được chọn cho cả SBP và DBP
- Lưu: `ml/models/random_forest_models.pkl` (2.55 MB) + `svr_models.pkl` (67 KB)
- Mỗi file có `.sha256` sidecar để verify integrity

### Bước 4: Tích hợp + Deploy ✅ HOÀN THÀNH

Server tự load model khi khởi động:

```
GET https://ppg-ml.onrender.com/api/ml/models
→ {"classical_ml": {"is_trained": true, "base_confidence": 0.75}}
```

Test end-to-end thành công:
```
POST https://ppg-ml.onrender.com/api/ppg/upload
→ HR=71.6 BPM, signal_quality=good, model_used=classical_ml
```

---

## Kỳ Vọng Kết Quả

| Chỉ số | Target | Ghi chú |
|---|---|---|
| SBP MAE | 7-9 mmHg | Script chỉ print stdout, chưa lưu ra file |
| DBP MAE | 5-7 mmHg | Khớp với AAMI standard (<8mmHg) |
| R² | 0.4-0.6 | Dataset nhỏ (219 subjects), không thể cao hơn |

**Lưu ý:** MAE dưới 5 mmHg yêu cầu Deep Learning (CNN-LSTM hoặc Transformer) + dataset lớn hơn. Classical ML trên 38 features đạt ~7-9 mmHg — đây là baseline hợp lý.

---

## Bước Tiếp Theo (Sau Khi Có Baseline)

1. **Đo MAE thực tế:** Thêm bước lưu `cv_results.json` vào `train_ppg_bp.py`
2. **Validate với data ESP32 thực:** Gửi PPG từ ngón tay → so sánh BP với máy đo cuff
3. **Train Gradient Boosting:** Chạy lại script, so sánh RF vs GB vs SVR
4. **Cải thiện accuracy:** Xem xét CNN-LSTM với PulseDB (cần GPU)
5. **Personalized calibration:** Fine-tune model cho từng user (2-3 reading đầu)

---

## Cảnh Báo Quan Trọng

⚠️ Đây là **research prototype**, không phải thiết bị y tế. Không thay thế máy đo huyết áp có cuff. Kết quả chỉ mang tính tham khảo / học tập.
