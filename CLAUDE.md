# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Bối cảnh

Sub-project ML của hệ thống PPG Monitor. Nhận tín hiệu IR/Red từ ESP32 + MAX30102, trích xuất 38 features, dự đoán huyết áp (SBP/DBP), SpO2, HR, HRV, AFib thông qua ML pipeline. Deploy song song hoặc thay thế cho `Backend code/main.py`.

---

## Lệnh thường dùng

Tất cả lệnh phải chạy từ thư mục `ML part/` (vì `main.py` import `ml_models` theo relative path):

```bash
cd "ML code/ML part"

# Chạy server local
uvicorn main:app --reload --port 8000

# Cài dependencies (thêm pandas + openpyxl để train với dataset thực)
pip install -r requirements.txt
pip install pandas openpyxl

# Train với PPG-BP dataset (cần tải trước — xem BP_PROJECT_GUIDE.md)
python train_ppg_bp.py
# → sinh ra models/random_forest_models.pkl

# Test nhanh endpoint sau khi server chạy
curl -X GET http://localhost:8000/api/ml/models
curl -X POST http://localhost:8000/api/ml/train \
  -H "Content-Type: application/json" \
  -d '{"model_type": "random_forest", "num_samples": 200}'
```

**Build Docker:**
```bash
cd "ML code/ML part"
docker build -t ppg-ml .
docker run -p 8000:8000 ppg-ml
```

---

## Kiến trúc

### Tổng quan luồng dữ liệu

```
ESP32 raw IR/Red values
        │
        ▼
extract_features(ir, red, fs)          ← ml_models.py
        │
        ├─ extract_time_domain_features()     → 18 features
        ├─ extract_frequency_domain_features() → 8 features  (Welch PSD)
        └─ extract_morphological_features()    → 12 features (per-cycle)
        │
        ▼
PPGFeatures dataclass (38 features + raw_segment)
        │
        ▼
EnsemblePredictor.predict(features)    ← ml_models.py
        │
        ├─ ClassicalMLModel  → loads models/random_forest_models.pkl
        ├─ DeepLearningModel → loads models/cnn_lstm_model.npz  (nếu tồn tại)
        └─ TransformerModel  → loads models/transformer_model.npz (nếu tồn tại)
        │
        ▼
PredictionResult.to_dict()
        │
        ▼
API Response JSON
```

### Các file

**`ml_models.py`** — toàn bộ ML logic:
- `PPGFeatures` dataclass: container cho 38 features + raw signal
- `PredictionResult` dataclass: container cho output (SBP, DBP, SpO2, HR, HRV, AFib, stress)
- `extract_features(ir, red, fs)` → `PPGFeatures`: entry point cho feature extraction
- `BasePPGModel`: abstract base với `predict()`, `get_confidence()`, `is_trained`
- `ClassicalMLModel`: wrapper scikit-learn (RF/GB/SVR). Load từ `.pkl`, fallback rule-based nếu chưa train
- `DeepLearningModel`: numpy-only CNN-LSTM simulation. Load từ `.npz`
- `TransformerModel`: numpy-only Transformer simulation. Load từ `.npz`
- `EnsemblePredictor`: weighted average của các model **đã train** (chỉ Classical có fallback rule-based)
- `generate_synthetic_training_data()` + `train_classical_models()`: dùng cho `/api/ml/train`

**`main.py`** — FastAPI server v3.0:
- Import toàn bộ từ `ml_models.py` (phải cùng thư mục)
- `ensemble_predictor` khởi tạo 1 lần lúc startup
- Hai in-memory dict: `readings_db` (PPG readings) và `ml_results_db` (ML results)

**`train_ppg_bp.py`** — training script cho PPG-BP dataset (Liang et al. 2018):
- Đọc từ `data/ppg-bp/Data File/0_subject/*.txt` + `data/ppg-bp/PPG-BP dataset.xlsx`
- Resample 1000 Hz → 100 Hz để khớp ESP32
- Thử 3 model type (RF/GB/SVR), chọn best theo MAE, save vào `models/random_forest_models.pkl`
- **Format save bắt buộc:** `{"models": {target: model}, "scalers": {target: scaler}, "metadata": {...}}`
  → Đây là format duy nhất `ClassicalMLModel._try_load_models` chấp nhận

---

## API Endpoints

| Method | Path | Ghi chú |
|--------|------|---------|
| `GET` | `/` | Health check + danh sách models |
| `POST` | `/api/ppg/upload` | Full pipeline: classical signal processing + ML |
| `POST` | `/api/ml/predict` | ML-only, bỏ qua signal processing |
| `POST` | `/api/ml/train` | Train trên synthetic data (`num_samples` 50–500) |
| `GET` | `/api/ml/models` | Trạng thái từng model (is_trained, confidence) |
| `GET` | `/api/ppg/history/{device_id}` | Lịch sử đo |
| `GET` | `/api/ppg/latest/{device_id}` | Kết quả mới nhất |
| `DELETE` | `/api/ppg/history/{device_id}` | Xóa lịch sử |

---

## Quy tắc code (kế thừa từ project CLAUDE.md)

- Pydantic v2: dùng `.model_dump()`, không dùng `.dict()`
- Datetime: dùng `datetime.now(timezone.utc)`, không dùng `datetime.utcnow()`
- Mọi lỗi trả về client phải bằng tiếng Việt
- Comment nghiệp vụ tiếng Việt; thuật ngữ kỹ thuật (bandpass, PSD, RMSSD) tiếng Anh
- `MODEL_DIR` qua `os.environ.get("MODEL_DIR", "models")`, không hardcode

---

## Các điểm quan trọng (đã fix, không được revert)

**Ensemble chỉ nhận model đã train** (`ml_models.py:~860`): `DeepLearningModel` và `TransformerModel` chỉ được thêm vào `EnsemblePredictor` khi `is_trained == True`. Nếu không có file `.npz`, các model này dùng random weights và sẽ làm lệch kết quả ensemble.

**`glucose_estimate` luôn trả `null`** trong `PredictionResult.to_dict()`: PPG-based glucose chưa có scientific backing đủ mạnh; trả số sẽ gây hiểu nhầm y tế.

**`/api/ml/train` giới hạn `num_samples ∈ [50, 500]`**: cross-validation 5-fold trên Render free-tier (512 MB RAM) sẽ crash nếu không giới hạn.

---

## Hạn chế đã biết

- `readings_db` và `ml_results_db` in-memory, mất khi server restart
- Không có authentication — bất kỳ ai biết URL đều gọi được `/api/ml/train`
- BP accuracy: ±8–15 mmHg với Classical ML; cần CNN-LSTM + dataset lớn hơn để đạt ±5 mmHg (IEEE cuff-less standard)
- `DeepLearningModel` và `TransformerModel` là numpy-only simulation, không phải PyTorch/TensorFlow thực sự
- `fall_time` và import `json` trong `ml_models.py` hiện unused (harmless)
