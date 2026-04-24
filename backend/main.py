"""
PPG Backend Server - UNIFIED v3.0 (tối ưu cho Render Free Tier 512MB)
====================================================================
Kết hợp tinh hoa từ:
  - main.py v2.0 (FastAPI nhẹ, chạy tốt trên Render)
  - backend_analysis.md (chiến lược Hybrid, tránh PyTorch/GAN)
  - backend_ppg_processor.ipynb v2.3 (SpO2 RMS + piecewise curve)

Các cải tiến chính so với v2.0:
  1. SpO2 v2.3 Classical: AC=RMS, DC=Hybrid (median + lowpass), piecewise curve
  2. Tight Validation: chặn AC/DC & Ratio R bất thường (không còn SpO2 ảo)
  3. Temporal Smoothing: median 5 mẫu gần nhất mỗi device
  4. HRV sâu: SDNN + RMSSD + pNN50 + MeanNN (tính thuần numpy, không nặng)
  5. HR outlier rejection bằng median thay vì mean
  6. Perfusion Index (PI) - chỉ số tưới máu
  7. Confidence score riêng cho HR và SpO2
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from typing import List, Optional, Dict, Any, Tuple
from collections import deque
from datetime import datetime, timezone
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
import uuid

app = FastAPI(title="PPG Server", version="3.0.0")

# Cho phép ESP32 và Mobile App kết nối
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# Database đơn giản (in-memory) - đủ dùng cho học tập / demo
# Production: thay bằng PostgreSQL/Firebase RTDB nếu cần
# ============================================================
readings_db: Dict[str, dict] = {}

# Lịch sử SpO2 per-device phục vụ Temporal Smoothing (median filter)
spo2_history: Dict[str, deque] = {}

# ============================================================
# CONFIG - tập trung một chỗ, dễ tinh chỉnh
# ============================================================
# Giới hạn kích thước dữ liệu
MIN_SAMPLES = 50         # Tối thiểu 50 mẫu (0.5s ở 100Hz)
MAX_SAMPLES = 10_000     # Tối đa 10,000 mẫu (100s ở 100Hz)
MIN_SAMPLE_RATE = 25     # Hz
MAX_SAMPLE_RATE = 400    # Hz
MAX_HISTORY = 100        # Số bản ghi lưu tối đa mỗi device

# HR validation
HR_MIN_BPM = 40
HR_MAX_BPM = 180

# SpO2 validation (Tight)
SPO2_MIN_VALID = 85.0
SPO2_MAX_VALID = 100.0
SPO2_RATIO_MIN = 0.4     # Ratio R không được quá thấp (lỗi tín hiệu)
SPO2_RATIO_MAX = 2.0     # Ratio R không được quá cao (tín hiệu ảo)
SPO2_SMOOTH_WINDOW = 5   # Số mẫu cho median filter

# AC/DC ratio bounds - tối ưu cho MAX30102
AC_DC_MIN_IR = 0.001
AC_DC_MAX_IR = 0.12
AC_DC_MIN_RED = 0.001
AC_DC_MAX_RED = 0.10

# Ngưỡng phát hiện ngón tay đặt đúng
MIN_FINGER_DC = 1000


# ============================================================
# Models
# ============================================================
class PPGReading(BaseModel):
    """Dữ liệu thô từ ESP32 + MAX30102"""
    device_id: str
    ir_values: List[int]
    red_values: List[int]
    sample_rate: int = 100
    timestamp: Optional[str] = None

    @field_validator("sample_rate")
    @classmethod
    def _v_sr(cls, v: int) -> int:
        if not (MIN_SAMPLE_RATE <= v <= MAX_SAMPLE_RATE):
            raise ValueError(
                f"sample_rate phải từ {MIN_SAMPLE_RATE} đến {MAX_SAMPLE_RATE} Hz"
            )
        return v

    @field_validator("ir_values", "red_values")
    @classmethod
    def _v_arr(cls, v: List[int]) -> List[int]:
        if len(v) > MAX_SAMPLES:
            raise ValueError(f"Tối đa {MAX_SAMPLES} mẫu mỗi lần gửi")
        return v

    @field_validator("device_id")
    @classmethod
    def _v_id(cls, v: str) -> str:
        v = v.strip()
        if not v or len(v) > 64:
            raise ValueError("device_id phải từ 1-64 ký tự")
        return v


class HRVMetrics(BaseModel):
    """Chỉ số HRV time-domain (không cần neurokit2)"""
    sdnn_ms: float       # Standard Deviation of NN intervals
    rmssd_ms: float      # Root Mean Square of Successive Differences
    pnn50_pct: float     # % of NN intervals differing >50ms
    mean_nn_ms: float    # Trung bình khoảng RR


class PPGResult(BaseModel):
    """Kết quả sau xử lý - gửi về Mobile App"""
    reading_id: str
    device_id: str
    heart_rate: float            # Nhịp tim (BPM)
    hr_confidence: float         # 0-1
    spo2: float                  # SpO2 (%) - đã smoothing
    spo2_raw: float              # SpO2 (%) - chưa smoothing (debug)
    spo2_confidence: float       # 0-1
    ratio_r: float               # Ratio of Ratios (R)
    perfusion_index: float       # PI (%) = AC_ir/DC_ir * 100
    hrv: HRVMetrics              # HRV time-domain
    signal_quality: str          # "good" | "fair" | "poor"
    rejection_reason: Optional[str]  # lý do SpO2 bị reject (nếu có)
    filtered_signal: List[float] # Tín hiệu IR đã lọc (cho Mobile vẽ)
    peaks: List[int]             # Chỉ số các đỉnh systolic
    timestamp: str


# ============================================================
# Bộ lọc tín hiệu
# ============================================================
def bandpass_filter(signal: np.ndarray, fs: int,
                    lowcut: float = 0.5, highcut: float = 5.0,
                    order: int = 4) -> np.ndarray:
    """
    Butterworth bandpass filter
    - 0.5Hz: loại DC offset, chuyển động chậm
    - 5Hz: loại nhiễu cao tần (nhịp tim 30-180 BPM ~ 0.5-3 Hz)
    """
    nyquist = fs / 2.0
    low = max(lowcut / nyquist, 0.001)
    high = min(highcut / nyquist, 0.999)
    if low >= high:
        return signal.copy()
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal)


def lowpass_filter(signal: np.ndarray, fs: int,
                   cutoff: float = 0.5, order: int = 4) -> np.ndarray:
    """Butterworth lowpass - trích xuất đường nền DC mượt"""
    nyquist = fs / 2.0
    normalized = min(cutoff / nyquist, 0.999)
    b, a = butter(order, normalized, btype="low")
    return filtfilt(b, a, signal)


# ============================================================
# Tính nhịp tim + HRV (robust với median & outlier rejection)
# ============================================================
def calculate_hr_and_hrv(filtered_signal: np.ndarray,
                        fs: int) -> Tuple[float, List[int], HRVMetrics, float]:
    """
    Tính HR + HRV từ tín hiệu đã lọc
    Returns: (hr_bpm, peak_indices, hrv_metrics, hr_confidence)
    """
    empty_hrv = HRVMetrics(sdnn_ms=0.0, rmssd_ms=0.0,
                           pnn50_pct=0.0, mean_nn_ms=0.0)

    # Khoảng cách tối thiểu giữa 2 đỉnh (ứng với HR max)
    min_distance = max(int(fs * 60 / HR_MAX_BPM), 1)

    peaks, _ = find_peaks(
        filtered_signal,
        distance=min_distance,
        height=np.mean(filtered_signal),
        prominence=0.1 * np.std(filtered_signal),
    )

    if len(peaks) < 2:
        return 0.0, peaks.tolist(), empty_hrv, 0.0

    # Tính RR intervals (giây)
    rr_sec = np.diff(peaks) / fs

    # Loại outlier: chỉ giữ RR ứng với HR_MIN..HR_MAX
    valid_mask = (rr_sec >= 60.0 / HR_MAX_BPM) & (rr_sec <= 60.0 / HR_MIN_BPM)
    rr_valid = rr_sec[valid_mask]

    if len(rr_valid) == 0:
        return 0.0, peaks.tolist(), empty_hrv, 0.0

    # HR dùng MEDIAN (robust hơn mean khi có outlier)
    hr_bpm = 60.0 / float(np.median(rr_valid))

    # HRV time-domain (thuần numpy - không cần neurokit2 nặng)
    rr_ms = rr_valid * 1000.0
    sdnn = float(np.std(rr_ms, ddof=1)) if len(rr_ms) > 1 else 0.0
    if len(rr_ms) > 1:
        diff_nn = np.diff(rr_ms)
        rmssd = float(np.sqrt(np.mean(diff_nn ** 2)))
        nn50 = int(np.sum(np.abs(diff_nn) > 50))
        pnn50 = float(nn50 / len(diff_nn) * 100) if len(diff_nn) > 0 else 0.0
    else:
        rmssd, pnn50 = 0.0, 0.0
    mean_nn = float(np.mean(rr_ms))

    hrv = HRVMetrics(
        sdnn_ms=round(sdnn, 2),
        rmssd_ms=round(rmssd, 2),
        pnn50_pct=round(pnn50, 2),
        mean_nn_ms=round(mean_nn, 2),
    )

    # Confidence cho HR: dựa trên tỉ lệ RR hợp lệ + mật độ đỉnh hợp lý
    valid_ratio = len(rr_valid) / len(rr_sec)
    duration_sec = len(filtered_signal) / fs
    expected_peaks = hr_bpm / 60.0 * duration_sec if hr_bpm > 0 else 0
    density_ratio = min(len(peaks) / expected_peaks, 1.0) if expected_peaks > 0 else 0
    hr_conf = round(0.6 * valid_ratio + 0.4 * density_ratio, 2)

    return round(hr_bpm, 1), peaks.tolist(), hrv, hr_conf


# ============================================================
# SpO2 v2.3 Classical - RMS + Hybrid DC + Piecewise curve
# ============================================================
def calculate_spo2_v23(ir: np.ndarray, red: np.ndarray,
                      fs: int) -> Tuple[Optional[float], float, float, float, str]:
    """
    SpO2 v2.3 Classical (tối ưu cho MAX30102):
      - AC = RMS của tín hiệu bandpass (chống nhiễu đột biến tốt hơn peak-to-peak)
      - DC = Hybrid: trung bình của median(raw) và mean(lowpass)
      - R  = (AC_red/DC_red) / (AC_ir/DC_ir)
      - Calibration piecewise 3 đoạn theo R

    Returns: (spo2_raw, ratio_r, acdc_ir, acdc_red, rejection_reason)
             spo2_raw = None nếu bị reject (không hợp lệ)
    """
    # --- AC: RMS sau bandpass (IR 0.5-5Hz, RED 0.7-3.5Hz) ---
    ir_ac_sig = bandpass_filter(ir, fs, 0.5, 5.0)
    red_ac_sig = bandpass_filter(red, fs, 0.7, 3.5)
    ac_ir = float(np.sqrt(np.mean(ir_ac_sig ** 2)))
    ac_red = float(np.sqrt(np.mean(red_ac_sig ** 2)))

    # --- DC: Hybrid (median raw + mean lowpass) / 2 ---
    ir_lp = lowpass_filter(ir, fs, cutoff=0.5)
    red_lp = lowpass_filter(red, fs, cutoff=0.5)
    dc_ir = float((np.median(ir) + np.mean(ir_lp)) / 2.0)
    dc_red = float((np.median(red) + np.mean(red_lp)) / 2.0)

    if dc_ir <= 0 or dc_red <= 0 or ac_ir <= 0 or ac_red <= 0:
        return None, 0.0, 0.0, 0.0, "ac_or_dc_invalid"

    acdc_ir = ac_ir / dc_ir
    acdc_red = ac_red / dc_red

    # --- Tight Validation: chặn SpO2 ảo ---
    if not (AC_DC_MIN_IR <= acdc_ir <= AC_DC_MAX_IR):
        return None, 0.0, acdc_ir, acdc_red, "acdc_ir_out_of_range"
    if not (AC_DC_MIN_RED <= acdc_red <= AC_DC_MAX_RED):
        return None, 0.0, acdc_ir, acdc_red, "acdc_red_out_of_range"

    # --- Ratio of Ratios ---
    R = acdc_red / acdc_ir
    if not (SPO2_RATIO_MIN <= R <= SPO2_RATIO_MAX):
        return None, R, acdc_ir, acdc_red, "ratio_r_out_of_range"

    # --- Calibration curve piecewise (tối ưu MAX30102) ---
    if R < 0.7:
        spo2 = 105.5 - 17.0 * R
    elif R < 1.1:
        spo2 = 102.0 - 19.5 * R
    else:
        spo2 = 108.0 - 23.0 * R

    spo2 = float(np.clip(spo2, SPO2_MIN_VALID, SPO2_MAX_VALID))

    return round(spo2, 1), round(R, 3), round(acdc_ir, 5), round(acdc_red, 5), "ok"


def apply_temporal_smoothing(device_id: str, spo2_raw: float) -> float:
    """Lọc median qua SPO2_SMOOTH_WINDOW mẫu gần nhất (mỗi device riêng)"""
    if device_id not in spo2_history:
        spo2_history[device_id] = deque(maxlen=SPO2_SMOOTH_WINDOW)
    spo2_history[device_id].append(spo2_raw)
    return round(float(np.median(list(spo2_history[device_id]))), 1)


def calculate_spo2_confidence(ir: np.ndarray, red: np.ndarray,
                             acdc_ir: float, acdc_red: float) -> float:
    """Confidence SpO2 dựa trên SNR của cả 2 kênh + độ ổn định AC/DC"""
    if len(ir) < 10 or len(red) < 10:
        return 0.0
    ir_snr = np.ptp(ir) / (np.std(ir) + 1e-6)
    red_snr = np.ptp(red) / (np.std(red) + 1e-6)
    snr_score = min((ir_snr + red_snr) / 20.0, 1.0)

    # AC/DC nằm giữa range là lý tưởng
    def _mid_score(v, lo, hi):
        if v <= lo or v >= hi:
            return 0.0
        mid = (lo + hi) / 2
        return 1.0 - abs(v - mid) / (hi - lo)

    acdc_score = (_mid_score(acdc_ir, AC_DC_MIN_IR, AC_DC_MAX_IR) +
                  _mid_score(acdc_red, AC_DC_MIN_RED, AC_DC_MAX_RED)) / 2

    return round(0.6 * snr_score + 0.4 * acdc_score, 2)


# ============================================================
# Đánh giá chất lượng tín hiệu tổng thể
# ============================================================
def assess_signal_quality(raw_signal: np.ndarray,
                         filtered_signal: np.ndarray,
                         peaks: list, fs: int) -> str:
    """Kết hợp SNR + mật độ đỉnh + biên độ (giữ từ main.py v2.0, vẫn rất tốt)"""
    score = 0

    # 1. SNR (dB)
    noise = raw_signal - np.mean(raw_signal) - filtered_signal
    signal_power = np.mean(filtered_signal ** 2)
    noise_power = np.mean(noise ** 2)
    snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else 50
    if snr > 10:
        score += 2
    elif snr > 5:
        score += 1

    # 2. Mật độ đỉnh hợp lý
    duration = len(filtered_signal) / fs
    if len(peaks) >= 2 and duration > 0:
        bpm = len(peaks) / duration * 60
        if 40 <= bpm <= 180:
            score += 2
        elif 30 <= bpm <= 200:
            score += 1

    # 3. Biên độ đủ lớn
    amplitude = np.max(filtered_signal) - np.min(filtered_signal)
    if amplitude > 0.01 * np.mean(np.abs(raw_signal)):
        score += 1

    if score >= 4:
        return "good"
    if score >= 2:
        return "fair"
    return "poor"


# ============================================================
# API Endpoints
# ============================================================
@app.get("/")
def root():
    """Health check"""
    return {
        "status": "running",
        "service": "PPG Processing Server",
        "version": "3.0.0",
        "features": [
            "SpO2 v2.3 (RMS + Hybrid DC + Piecewise)",
            "HRV time-domain (SDNN/RMSSD/pNN50)",
            "Temporal Smoothing (median 5)",
            "Tight Validation",
            "Perfusion Index",
        ],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/api/ppg/upload", response_model=PPGResult)
def upload_ppg_data(reading: PPGReading):
    """
    ESP32 gửi dữ liệu thô lên đây. Server xử lý và trả kết quả.
    Pipeline:
      1. Validate
      2. Bandpass lọc IR → tính HR + HRV
      3. SpO2 v2.3 (RMS/Hybrid) + Tight Validation
      4. Temporal Smoothing SpO2 (median 5)
      5. Signal Quality + Confidence
      6. Lưu DB + giới hạn history
    """
    # --- 1. Validate ---
    if len(reading.ir_values) < MIN_SAMPLES:
        raise HTTPException(
            400,
            f"Cần ít nhất {MIN_SAMPLES} mẫu "
            f"({MIN_SAMPLES / reading.sample_rate:.1f}s ở {reading.sample_rate}Hz)",
        )
    if len(reading.ir_values) != len(reading.red_values):
        raise HTTPException(400, "Số mẫu IR và Red phải bằng nhau")

    ir = np.asarray(reading.ir_values, dtype=float)
    red = np.asarray(reading.red_values, dtype=float)
    fs = reading.sample_rate

    if np.mean(ir) < MIN_FINGER_DC or np.mean(red) < MIN_FINGER_DC:
        raise HTTPException(
            400,
            "Giá trị cảm biến quá thấp - kiểm tra ngón tay đặt trên cảm biến",
        )

    # --- 2. Lọc IR & tính HR + HRV ---
    filtered_ir = bandpass_filter(ir, fs)
    hr, peaks, hrv, hr_conf = calculate_hr_and_hrv(filtered_ir, fs)

    # --- 3. SpO2 v2.3 ---
    spo2_raw, ratio_r, acdc_ir, acdc_red, reject_reason = calculate_spo2_v23(ir, red, fs)

    if spo2_raw is None:
        # SpO2 không hợp lệ - vẫn trả kết quả HR, nhưng spo2 = 0 + lý do reject
        spo2_smooth = 0.0
        spo2_raw_val = 0.0
        spo2_conf = 0.0
    else:
        # --- 4. Temporal Smoothing ---
        spo2_smooth = apply_temporal_smoothing(reading.device_id, spo2_raw)
        spo2_raw_val = spo2_raw
        spo2_conf = calculate_spo2_confidence(ir, red, acdc_ir, acdc_red)

    # Perfusion Index (PI) = AC_ir / DC_ir * 100
    perfusion_index = round(acdc_ir * 100, 2)

    # --- 5. Signal Quality tổng hợp ---
    quality = assess_signal_quality(ir, filtered_ir, peaks, fs)

    # --- 6. Tạo kết quả ---
    reading_id = str(uuid.uuid4())[:8]
    result = PPGResult(
        reading_id=reading_id,
        device_id=reading.device_id,
        heart_rate=hr,
        hr_confidence=hr_conf,
        spo2=spo2_smooth,
        spo2_raw=spo2_raw_val,
        spo2_confidence=spo2_conf,
        ratio_r=ratio_r,
        perfusion_index=perfusion_index,
        hrv=hrv,
        signal_quality=quality,
        rejection_reason=None if reject_reason == "ok" else reject_reason,
        filtered_signal=filtered_ir.tolist(),
        peaks=peaks,
        timestamp=reading.timestamp or datetime.now(timezone.utc).isoformat(),
    )

    # Lưu DB
    readings_db[reading_id] = result.model_dump()

    # Giới hạn bộ nhớ theo device
    device_keys = [k for k, v in readings_db.items()
                   if v["device_id"] == reading.device_id]
    if len(device_keys) > MAX_HISTORY:
        device_keys.sort(key=lambda k: readings_db[k]["timestamp"])
        for old_key in device_keys[:-MAX_HISTORY]:
            del readings_db[old_key]

    return result


@app.get("/api/ppg/history/{device_id}")
def get_history(device_id: str, limit: int = 20):
    """Lịch sử đo của device"""
    limit = max(1, min(limit, MAX_HISTORY))
    history = [v for v in readings_db.values() if v["device_id"] == device_id]
    history.sort(key=lambda x: x["timestamp"], reverse=True)
    return {"device_id": device_id, "count": len(history), "readings": history[:limit]}


@app.get("/api/ppg/latest/{device_id}")
def get_latest(device_id: str):
    """Kết quả mới nhất của device"""
    readings = [v for v in readings_db.values() if v["device_id"] == device_id]
    if not readings:
        raise HTTPException(404, "Chưa có dữ liệu cho thiết bị này")
    readings.sort(key=lambda x: x["timestamp"], reverse=True)
    return readings[0]


@app.delete("/api/ppg/history/{device_id}")
def clear_history(device_id: str):
    """Xóa toàn bộ lịch sử của device (bao gồm cả SpO2 smoothing buffer)"""
    keys = [k for k, v in readings_db.items() if v["device_id"] == device_id]
    for k in keys:
        del readings_db[k]
    spo2_history.pop(device_id, None)
    return {"device_id": device_id, "deleted": len(keys)}


@app.get("/api/stats")
def global_stats():
    """Thống kê toàn hệ thống"""
    devices = set(v["device_id"] for v in readings_db.values())
    return {
        "total_readings": len(readings_db),
        "unique_devices": len(devices),
        "devices": sorted(devices),
        "spo2_buffers": {d: len(buf) for d, buf in spo2_history.items()},
    }
