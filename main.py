"""
PPG Backend Server v3.0 - ML Pipeline Integrated
Nhận dữ liệu từ ESP32, xử lý tín hiệu PPG + ML prediction, trả kết quả về Mobile App
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from typing import List, Optional, Dict
from datetime import datetime, timezone
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
import uuid
import logging

from ml_models import (
    extract_features,
    ClassicalMLModel,
    DeepLearningModel,
    TransformerModel,
    EnsemblePredictor,
    PredictionResult,
    PPGFeatures,
    generate_synthetic_training_data,
    train_classical_models,
    TARGET_NAMES,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="PPG Server", version="3.0.0")

# Cho phép ESP32 và Mobile App kết nối
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# Database đơn giản (in-memory) - dùng cho học tập
# ============================================================
readings_db: dict = {}
ml_results_db: dict = {}

# ============================================================
# Giới hạn
# ============================================================
MIN_SAMPLES = 50
MAX_SAMPLES = 10_000
MIN_SAMPLE_RATE = 25
MAX_SAMPLE_RATE = 400
MAX_HISTORY = 100

# ============================================================
# ML Models (khởi tạo 1 lần khi server start)
# ============================================================
ensemble_predictor = EnsemblePredictor()
logger.info("ML Ensemble predictor đã khởi tạo với %d models", len(ensemble_predictor.models))


# ============================================================
# Models
# ============================================================
class PPGReading(BaseModel):
    """Dữ liệu thô từ ESP32 MAX30102"""
    device_id: str
    ir_values: List[int]
    red_values: List[int]
    sample_rate: int = 100
    timestamp: Optional[str] = None

    @field_validator("sample_rate")
    @classmethod
    def validate_sample_rate(cls, v: int) -> int:
        if v < MIN_SAMPLE_RATE or v > MAX_SAMPLE_RATE:
            raise ValueError(
                f"sample_rate phải từ {MIN_SAMPLE_RATE} đến {MAX_SAMPLE_RATE} Hz"
            )
        return v

    @field_validator("ir_values", "red_values")
    @classmethod
    def validate_array_size(cls, v: List[int]) -> List[int]:
        if len(v) > MAX_SAMPLES:
            raise ValueError(f"Tối đa {MAX_SAMPLES} mẫu mỗi lần gửi")
        return v

    @field_validator("device_id")
    @classmethod
    def validate_device_id(cls, v: str) -> str:
        v = v.strip()
        if not v or len(v) > 64:
            raise ValueError("device_id phải từ 1-64 ký tự")
        return v


class PPGResult(BaseModel):
    """Kết quả xử lý tín hiệu cơ bản + ML predictions"""
    reading_id: str
    device_id: str
    # Kết quả xử lý tín hiệu truyền thống
    heart_rate: float
    spo2: float
    hrv_sdnn: float
    signal_quality: str
    filtered_signal: List[float]
    peaks: List[int]
    timestamp: str
    # Kết quả ML predictions
    ml_predictions: Optional[dict] = None


class MLPredictionRequest(BaseModel):
    """Request chỉ dùng ML prediction (không cần xử lý tín hiệu truyền thống)"""
    device_id: str
    ir_values: List[int]
    red_values: List[int]
    sample_rate: int = 100
    model: str = "ensemble"  # "classical_ml", "deep_learning", "transformer", "ensemble"

    @field_validator("model")
    @classmethod
    def validate_model(cls, v: str) -> str:
        valid = {"classical_ml", "deep_learning", "transformer", "ensemble"}
        if v not in valid:
            raise ValueError(f"model phải là một trong: {valid}")
        return v


class TrainRequest(BaseModel):
    """Request train model với dữ liệu synthetic"""
    model_type: str = "random_forest"
    num_samples: int = 500

    @field_validator("model_type")
    @classmethod
    def validate_model_type(cls, v: str) -> str:
        valid = {"random_forest", "xgboost", "svm"}
        if v not in valid:
            raise ValueError(f"model_type phải là một trong: {valid}")
        return v

    @field_validator("num_samples")
    @classmethod
    def validate_num_samples(cls, v: int) -> int:
        # Giới hạn chống DoS + quá tải free-tier RAM:
        # synthetic data + cross-validation 5-fold có thể tốn nhiều CPU/RAM
        if v < 50 or v > 500:
            raise ValueError(
                "num_samples phải từ 50 đến 500 "
                "(giới hạn để tránh quá tải Render free-tier)"
            )
        return v


# ============================================================
# Xử lý tín hiệu PPG (giữ nguyên từ v2)
# ============================================================
def bandpass_filter(signal: np.ndarray, fs: int,
                    lowcut: float = 0.5, highcut: float = 5.0,
                    order: int = 4) -> np.ndarray:
    nyquist = fs / 2.0
    low = max(lowcut / nyquist, 0.001)
    high = min(highcut / nyquist, 0.999)
    if low >= high:
        return signal
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)


def lowpass_filter(signal: np.ndarray, fs: int,
                   cutoff: float = 0.5, order: int = 4) -> np.ndarray:
    nyquist = fs / 2.0
    normalized = min(cutoff / nyquist, 0.999)
    b, a = butter(order, normalized, btype='low')
    return filtfilt(b, a, signal)


def calculate_heart_rate(filtered_signal: np.ndarray, fs: int) -> tuple:
    min_distance = max(int(fs * 60 / 180), 1)
    peaks, _ = find_peaks(
        filtered_signal,
        distance=min_distance,
        height=np.mean(filtered_signal),
        prominence=0.1 * np.std(filtered_signal),
    )
    if len(peaks) < 2:
        return 0.0, peaks.tolist(), 0.0

    intervals = np.diff(peaks) / fs
    valid_mask = (intervals > 60.0 / 180) & (intervals < 60.0 / 30)
    valid_intervals = intervals[valid_mask]
    if len(valid_intervals) == 0:
        return 0.0, peaks.tolist(), 0.0

    avg_interval = np.mean(valid_intervals)
    heart_rate = 60.0 / avg_interval if avg_interval > 0 else 0.0
    hrv_sdnn = float(np.std(valid_intervals) * 1000)
    return round(heart_rate, 1), peaks.tolist(), round(hrv_sdnn, 1)


def calculate_spo2(ir_values: np.ndarray, red_values: np.ndarray,
                   fs: int) -> float:
    ir_ac_signal = bandpass_filter(ir_values, fs)
    red_ac_signal = bandpass_filter(red_values, fs)
    ac_ir = (np.max(ir_ac_signal) - np.min(ir_ac_signal)) / 2.0
    ac_red = (np.max(red_ac_signal) - np.min(red_ac_signal)) / 2.0

    ir_dc_signal = lowpass_filter(ir_values, fs)
    red_dc_signal = lowpass_filter(red_values, fs)
    dc_ir = np.mean(ir_dc_signal)
    dc_red = np.mean(red_dc_signal)

    if dc_red == 0 or dc_ir == 0 or ac_ir == 0:
        return 0.0

    R = (ac_red / dc_red) / (ac_ir / dc_ir)
    spo2 = 110.0 - 25.0 * R
    return round(max(0.0, min(100.0, spo2)), 1)


def assess_signal_quality(raw_signal: np.ndarray,
                          filtered_signal: np.ndarray,
                          peaks: list, fs: int) -> str:
    score = 0
    noise = raw_signal - np.mean(raw_signal) - filtered_signal
    signal_power = np.mean(filtered_signal ** 2)
    noise_power = np.mean(noise ** 2)
    if noise_power > 0:
        snr = 10 * np.log10(signal_power / noise_power)
    else:
        snr = 50
    if snr > 10:
        score += 2
    elif snr > 5:
        score += 1

    duration_sec = len(filtered_signal) / fs
    if len(peaks) >= 2 and duration_sec > 0:
        detected_bpm = len(peaks) / duration_sec * 60
        if 40 <= detected_bpm <= 180:
            score += 2
        elif 30 <= detected_bpm <= 200:
            score += 1

    amplitude = np.max(filtered_signal) - np.min(filtered_signal)
    if amplitude > 0.01 * np.mean(np.abs(raw_signal)):
        score += 1

    if score >= 4:
        return "good"
    elif score >= 2:
        return "fair"
    return "poor"


# ============================================================
# Helper: chạy ML prediction
# ============================================================
def _run_ml_prediction(ir: np.ndarray, red: np.ndarray,
                       fs: int, model_name: str = "ensemble") -> dict:
    """Trích xuất feature + chạy ML model, trả dict kết quả"""
    features = extract_features(ir, red, fs)

    if model_name == "ensemble":
        result = ensemble_predictor.predict(features)
    elif model_name == "classical_ml":
        model = ClassicalMLModel("random_forest")
        preds = model.predict(features)
        conf = model.get_confidence(features)
        result = PredictionResult(
            **preds, model_used="classical_ml", confidence=conf,
            feature_summary=features.to_dict(),
        )
    elif model_name == "deep_learning":
        model = DeepLearningModel("cnn_lstm")
        preds = model.predict(features)
        conf = model.get_confidence(features)
        result = PredictionResult(
            **preds, model_used="deep_learning", confidence=conf,
            feature_summary=features.to_dict(),
        )
    elif model_name == "transformer":
        model = TransformerModel()
        preds = model.predict(features)
        conf = model.get_confidence(features)
        result = PredictionResult(
            **preds, model_used="transformer", confidence=conf,
            feature_summary=features.to_dict(),
        )
    else:
        result = ensemble_predictor.predict(features)

    return result.to_dict()


# ============================================================
# API Endpoints
# ============================================================

@app.get("/")
def root():
    """Health check"""
    return {
        "status": "running",
        "service": "PPG Processing Server + ML Pipeline",
        "version": "3.0.0",
        "models": ["classical_ml", "deep_learning", "transformer", "ensemble"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/api/ppg/upload", response_model=PPGResult)
def upload_ppg_data(reading: PPGReading):
    """
    ESP32 gửi dữ liệu thô → xử lý tín hiệu truyền thống + ML prediction
    """
    if len(reading.ir_values) < MIN_SAMPLES:
        raise HTTPException(
            400,
            f"Cần ít nhất {MIN_SAMPLES} mẫu "
            f"({MIN_SAMPLES / reading.sample_rate:.1f}s ở {reading.sample_rate}Hz)",
        )
    if len(reading.ir_values) != len(reading.red_values):
        raise HTTPException(400, "Số mẫu IR và Red phải bằng nhau")

    ir = np.array(reading.ir_values, dtype=float)
    red = np.array(reading.red_values, dtype=float)
    fs = reading.sample_rate

    if np.mean(ir) < 1000 or np.mean(red) < 1000:
        raise HTTPException(400, "Giá trị cảm biến quá thấp - kiểm tra ngón tay đặt trên cảm biến")

    # Xử lý tín hiệu truyền thống
    filtered_ir = bandpass_filter(ir, fs)
    heart_rate, peaks, hrv_sdnn = calculate_heart_rate(filtered_ir, fs)
    spo2 = calculate_spo2(ir, red, fs)
    quality = assess_signal_quality(ir, filtered_ir, peaks, fs)

    # ML prediction
    try:
        ml_preds = _run_ml_prediction(ir, red, fs, "ensemble")
    except Exception as e:
        logger.warning("ML prediction lỗi: %s", e)
        ml_preds = None

    reading_id = str(uuid.uuid4())[:8]
    result = PPGResult(
        reading_id=reading_id,
        device_id=reading.device_id,
        heart_rate=heart_rate,
        spo2=spo2,
        hrv_sdnn=hrv_sdnn,
        signal_quality=quality,
        filtered_signal=filtered_ir.tolist(),
        peaks=peaks,
        timestamp=reading.timestamp or datetime.now(timezone.utc).isoformat(),
        ml_predictions=ml_preds,
    )

    readings_db[reading_id] = result.model_dump()

    # Giới hạn bộ nhớ
    device_readings = [
        k for k, v in readings_db.items()
        if v["device_id"] == reading.device_id
    ]
    if len(device_readings) > MAX_HISTORY:
        device_readings.sort(key=lambda k: readings_db[k]["timestamp"])
        for old_key in device_readings[:-MAX_HISTORY]:
            del readings_db[old_key]

    return result


@app.post("/api/ml/predict")
def ml_predict(req: MLPredictionRequest):
    """
    Endpoint ML-only: trích xuất feature + dự đoán
    Không chạy xử lý tín hiệu truyền thống
    """
    if len(req.ir_values) < MIN_SAMPLES:
        raise HTTPException(400, f"Cần ít nhất {MIN_SAMPLES} mẫu")
    if len(req.ir_values) != len(req.red_values):
        raise HTTPException(400, "Số mẫu IR và Red phải bằng nhau")

    ir = np.array(req.ir_values, dtype=float)
    red = np.array(req.red_values, dtype=float)

    if np.mean(ir) < 1000 or np.mean(red) < 1000:
        raise HTTPException(400, "Giá trị cảm biến quá thấp")

    result = _run_ml_prediction(ir, red, req.sample_rate, req.model)

    # Lưu kết quả
    result_id = str(uuid.uuid4())[:8]
    ml_results_db[result_id] = {
        "result_id": result_id,
        "device_id": req.device_id,
        "model": req.model,
        "predictions": result,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    return {
        "result_id": result_id,
        "device_id": req.device_id,
        **result,
    }


@app.post("/api/ml/train")
def train_model(req: TrainRequest):
    """
    Train classical ML model với dữ liệu synthetic (demo)
    Trong production: dùng dataset thực từ clinical study
    """
    global ensemble_predictor

    try:
        logger.info("Đang tạo %d mẫu synthetic...", req.num_samples)
        features_list, labels = generate_synthetic_training_data(
            num_samples=req.num_samples,
        )

        logger.info("Đang train model %s...", req.model_type)
        trained_model = train_classical_models(
            features_list, labels, req.model_type,
        )

        # Cập nhật ensemble predictor
        ensemble_predictor = EnsemblePredictor([
            trained_model,
            DeepLearningModel("cnn_lstm"),
            TransformerModel(),
        ])

        return {
            "status": "success",
            "model_type": req.model_type,
            "num_samples": req.num_samples,
            "message": f"Đã train model {req.model_type} với {req.num_samples} mẫu synthetic",
        }
    except ImportError:
        raise HTTPException(
            400,
            "Cần cài scikit-learn: pip install scikit-learn --break-system-packages",
        )
    except Exception as e:
        raise HTTPException(500, f"Lỗi khi train: {str(e)}")


@app.get("/api/ml/models")
def list_models():
    """Liệt kê các model có sẵn và trạng thái"""
    models_info = []
    for model in ensemble_predictor.models:
        models_info.append({
            "name": model.name,
            "is_trained": getattr(model, "is_trained", False),
            "base_confidence": model.get_confidence(
                PPGFeatures(
                    time_domain=np.zeros(18),
                    frequency_domain=np.zeros(8),
                    morphological=np.zeros(12),
                    raw_segment=np.zeros(100),
                    sample_rate=100,
                )
            ),
        })
    return {
        "models": models_info,
        "targets": TARGET_NAMES,
    }


@app.get("/api/ppg/history/{device_id}")
def get_history(device_id: str, limit: int = 20):
    """Mobile App lấy lịch sử đo của thiết bị"""
    limit = max(1, min(limit, MAX_HISTORY))
    history = [v for v in readings_db.values() if v["device_id"] == device_id]
    history.sort(key=lambda x: x["timestamp"], reverse=True)
    return {"device_id": device_id, "readings": history[:limit]}


@app.get("/api/ppg/latest/{device_id}")
def get_latest(device_id: str):
    """Mobile App lấy kết quả đo mới nhất"""
    readings = [v for v in readings_db.values() if v["device_id"] == device_id]
    if not readings:
        raise HTTPException(404, "Chưa có dữ liệu cho thiết bị này")
    readings.sort(key=lambda x: x["timestamp"], reverse=True)
    return readings[0]


@app.delete("/api/ppg/history/{device_id}")
def clear_history(device_id: str):
    """Xóa toàn bộ lịch sử đo của thiết bị"""
    keys_to_delete = [
        k for k, v in readings_db.items()
        if v["device_id"] == device_id
    ]
    for k in keys_to_delete:
        del readings_db[k]
    return {"device_id": device_id, "deleted": len(keys_to_delete)}
