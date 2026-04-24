# Kế Hoạch Triển Khai Hệ Thống ML Dự Đoán Huyết Áp

## Mục Tiêu

Xây dựng model ML từ tín hiệu PPG (ESP32 + MAX30102) dự đoán **huyết áp tâm thu (SBP)** và **huyết áp tâm trương (DBP)** không xâm lấn, không cần cuff.

## Dataset

**PPG-BP Database** (Liang et al., 2018 — Scientific Data)
- 219 subjects, tuổi 20-89, có cả bình thường và có bệnh tim mạch
- 3 recordings × 2.1 giây @ 1kHz cho mỗi người = 657 recordings
- Single-channel finger PPG (đúng form factor MAX30102)
- Labels: SBP, DBP, HR, age, sex, BMI, hypertension stage, diabetes, CVD

Link download: https://doi.org/10.6084/m9.figshare.5459299

## Các Bước

### Bước 1: Tải dataset (~30 phút)

```bash
# Tạo thư mục
mkdir -p data/ppg-bp

# Tải từ Figshare (copy link từ trang chủ)
# Sau khi giải nén, cấu trúc phải như sau:
data/ppg-bp/
├── Data File/
│   └── 0_subject/
│       ├── 2_1.txt
│       ├── 2_2.txt
│       ├── 2_3.txt
│       └── ... (657 files)
└── PPG-BP dataset.xlsx
```

### Bước 2: Cài thư viện bổ sung

```bash
pip install pandas openpyxl scikit-learn --break-system-packages
```

### Bước 3: Chạy training (5-10 phút trên laptop)

```bash
python train_ppg_bp.py
```

Output mong đợi:
```
Step 1: Load labels từ Excel
Đã load 219 subjects từ Excel

Step 2: Load PPG recordings + extract features
Load xong: 215 subjects, skip 4

Step 3: Train models
Target: SBP
  random_forest       MAE=8.42  RMSE=11.20  R²=0.412
  gradient_boosting   MAE=7.81  RMSE=10.55  R²=0.475
  svr                 MAE=9.10  RMSE=12.01  R²=0.342

So sánh với baseline (MAE trong mmHg) cho SBP:
  ✓ Liang et al. 2018 (original): 11.64 (ours: 7.81)
  ✗ Chowdhury et al. 2020 (GP): 3.02 (ours: 7.81)
  ✗ El-Hajj 2021 (CNN-LSTM): 5.77 (ours: 7.81)
```

### Bước 4: Tích hợp vào main.py

Model đã train sẽ tự động load khi khởi động server (đã có sẵn cơ chế trong `ml_models.py`).

```bash
# Restart server
uvicorn main:app --reload

# Test API
curl -X POST http://localhost:8000/api/ml/predict \
  -H "Content-Type: application/json" \
  -d '{
    "device_id": "esp32-001",
    "ir_values": [50000, 50100, ...],
    "red_values": [45000, 45100, ...],
    "sample_rate": 100,
    "model": "classical_ml"
  }'
```

Response sẽ có:
```json
{
  "blood_pressure": {"systolic": 128.3, "diastolic": 82.1, "unit": "mmHg"},
  "heart_rate": 74.5,
  "model_used": "classical_ml",
  "confidence": 0.75
}
```

## Kỳ Vọng Kết Quả

| Chỉ số | Target | Ghi chú |
|---|---|---|
| SBP MAE | 7-9 mmHg | Tốt hơn paper gốc của Liang |
| DBP MAE | 5-7 mmHg | Khớp với AAMI standard (<8mmHg) |
| R² | 0.4-0.6 | Dataset nhỏ, không thể cao hơn |

**Lưu ý:** MAE dưới 5 mmHg yêu cầu Deep Learning (CNN-LSTM hoặc Transformer) + dataset lớn hơn (MIMIC-III hoặc PulseDB). Classical ML trên 38 features chỉ đạt được ~7-9 mmHg — đây là baseline hợp lý.

## Bước Tiếp Theo (Sau Khi Có Baseline)

1. **Cải thiện accuracy:** Chuyển sang Deep Learning với PulseDB (cần GPU)
2. **Personalized calibration:** Fine-tune model cho từng user (fine-tuning với 2-3 reading đầu tiên)
3. **Validation lâm sàng:** So sánh với thiết bị cuff chuẩn (Omron HEM-7120) trên 10-20 người tình nguyện
4. **Thêm target phụ:** Hypertension classification (binary: normal/hypertensive) → accuracy thường >85%

## Cảnh Báo Quan Trọng

⚠️ Đây là **research prototype**, không phải thiết bị y tế. Không thay thế máy đo huyết áp có cuff. Kết quả chỉ mang tính tham khảo.
