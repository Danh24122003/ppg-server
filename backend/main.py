"""
PPG Backend Server - UNIFIED v4.0 (HeartPy + RR Accumulator)
====================================================================
Cải tiến so với v3.0:
  1. Peak detection: HeartPy (van Gent 2019) thay thế find_peaks đơn giản
     - Moving average window = 0.75s + adaptive threshold (grid search 18 mức)
     - Sub-sample accuracy: high_precision=True, spline upsample → 1000 Hz
  2. Overlap buffer 1.5s: tránh mất peak tại biên batch
  3. Deduplication: peaks trong vùng overlap bị loại, chỉ lấy RR cross-boundary
     và RR hoàn toàn mới (không đếm đỉnh 2 lần)
  4. RR accumulator per-device: tích lũy ≥30 nhịp trước khi tính HRV time-domain
  5. Outlier rejection ±20% median(RR) — thay thế absolute threshold cũ
  6. Error handling: BadSignalWarning/Exception từ hp.process() → HR=0, không crash
  7. Fallback scipy find_peaks nếu heartpy không được cài
"""

from fastapi import FastAPI, HTTPException, Header, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from pydantic import BaseModel, field_validator
from typing import List, Optional, Dict, Tuple
from collections import deque
from datetime import datetime, timezone, timedelta
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, welch
from scipy.interpolate import interp1d
import hmac
import os
import re
import threading
import uuid
from loguru import logger

try:
    import heartpy as hp
    HEARTPY_AVAILABLE = True
except ImportError:
    HEARTPY_AVAILABLE = False

PPG_API_KEY = os.environ.get("PPG_API_KEY", "").strip()

app = FastAPI(title="PPG Server", version="4.0.0")

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)

# ============================================================
# In-memory storage
# ============================================================
readings_db: Dict[str, dict] = {}
device_index: Dict[str, List[str]] = {}
spo2_history: Dict[str, deque] = {}

# Overlap buffer: 1.5s cuối của filtered IR batch trước, per device
overlap_buffer: Dict[str, np.ndarray] = {}

# RR accumulator: chuỗi RR intervals (ms) tích lũy qua các batch, per device
rr_accumulator: Dict[str, deque] = {}

_state_lock = threading.Lock()

# Per-device lock registry: bảo vệ tuần tự hóa upload cùng device
_device_locks: Dict[str, threading.Lock] = {}
_device_locks_registry_lock = threading.Lock()


def _get_device_lock(device_id: str) -> threading.Lock:
    """Trả về lock riêng cho device (tạo mới nếu chưa có). Double-check pattern thread-safe."""
    lock = _device_locks.get(device_id)
    if lock is not None:
        return lock
    with _device_locks_registry_lock:
        lock = _device_locks.get(device_id)
        if lock is None:
            lock = threading.Lock()
            _device_locks[device_id] = lock
        return lock

# ============================================================
# CONFIG
# ============================================================
MIN_SAMPLES = 50
MAX_SAMPLES = 10_000
MIN_SAMPLE_RATE = 25
MAX_SAMPLE_RATE = 400
MAX_HISTORY = 100
MAX_DEVICES = 50
MAX_TOTAL_READINGS = 5_000

HR_MIN_BPM = 40
HR_MAX_BPM = 200

SPO2_MIN_VALID = 85.0
SPO2_MAX_VALID = 100.0
SPO2_RATIO_MIN = 0.4
SPO2_RATIO_MAX = 2.0
SPO2_SMOOTH_WINDOW = 5

AC_DC_MIN_IR = 0.001
AC_DC_MAX_IR = 0.12
AC_DC_MIN_RED = 0.001
AC_DC_MAX_RED = 0.10

MIN_FINGER_DC = 1000

# HeartPy / peak detection
OVERLAP_SECONDS = 1.5           # độ dài overlap buffer (s)
RR_MIN_FOR_HRV = 30             # tối thiểu RR intervals để tính HRV time-domain
RR_ACCUMULATOR_MAX = 240        # max RR intervals lưu (~4 phút ở 60 BPM)
RR_OUTLIER_TOLERANCE = 0.20     # ±20% median(RR)
RR_MIN_FOR_LFHF = 60    # ~1 phút ở 60 BPM
HRV_RESAMPLE_FS = 4.0   # Hz - chuẩn HRV frequency analysis


# ============================================================
# Device ID validation (dùng chung cho Pydantic + path params)
# ============================================================
_DEVICE_ID_RE = re.compile(r'^[a-zA-Z0-9_\-]{1,64}$')


def _check_device_id(v: str) -> str:
    v = v.strip()
    if not _DEVICE_ID_RE.match(v):
        raise ValueError("device_id phải 1-64 ký tự, chỉ chứa a-z A-Z 0-9 _ -")
    return v


# ============================================================
# Models
# ============================================================
class PPGReading(BaseModel):
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
        return _check_device_id(v)

    @field_validator("timestamp")
    @classmethod
    def _v_ts(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        try:
            dt = datetime.fromisoformat(v)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            if dt > now + timedelta(seconds=5) or dt < now - timedelta(hours=1):
                return None
        except (ValueError, TypeError):
            return None
        return v


class HRVMetrics(BaseModel):
    sdnn_ms: float
    rmssd_ms: float
    pnn50_pct: float
    pnn20_pct: float = 0.0
    mean_nn_ms: float
    rr_count: int = 0           # số RR tích lũy (debug)
    reliability: str = "low"    # "low" | "medium" | "high"
    lf_ms2: Optional[float] = None
    hf_ms2: Optional[float] = None
    lf_hf: Optional[float] = None


class PPGResult(BaseModel):
    reading_id: str
    device_id: str
    heart_rate: float
    hr_confidence: float
    spo2: float
    spo2_raw: float
    spo2_confidence: float
    ratio_r: float
    perfusion_index: float
    hrv: HRVMetrics
    signal_quality: str
    rejection_reason: Optional[str]
    filtered_signal: List[float]
    peaks: List[int]
    timestamp: str


# ============================================================
# Bộ lọc tín hiệu
# ============================================================
def bandpass_filter(signal: np.ndarray, fs: int,
                    lowcut: float = 0.5, highcut: float = 4.0,
                    order: int = 4) -> np.ndarray:
    nyquist = fs / 2.0
    low = max(lowcut / nyquist, 0.001)
    high = min(highcut / nyquist, 0.999)
    if low >= high:
        return signal.copy()
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal)


def lowpass_filter(signal: np.ndarray, fs: int,
                   cutoff: float = 0.5, order: int = 4) -> np.ndarray:
    nyquist = fs / 2.0
    normalized = min(cutoff / nyquist, 0.999)
    b, a = butter(order, normalized, btype="low")
    return filtfilt(b, a, signal)


# ============================================================
# HeartPy Peak Detection + RR Accumulator
# ============================================================
def _reject_rr_outliers(rr_ms: np.ndarray) -> np.ndarray:
    """Loại RR nằm ngoài ±20% median(RR). Robust hơn absolute threshold với batch ngắn."""
    if len(rr_ms) == 0:
        return rr_ms
    med = float(np.median(rr_ms))
    lo = med * (1.0 - RR_OUTLIER_TOLERANCE)
    hi = med * (1.0 + RR_OUTLIER_TOLERANCE)
    return rr_ms[(rr_ms >= lo) & (rr_ms <= hi)]


def _compute_lf_hf(rr_ms: np.ndarray, fs_rr: float = HRV_RESAMPLE_FS) -> dict:
    """
    Frequency-domain HRV (Task Force 1996):
      - Nội suy RR lên lưới đều fs_rr Hz bằng cubic spline
      - Welch PSD với Hann window
      - Tích phân dải: LF (0.04-0.15 Hz), HF (0.15-0.40 Hz)
    Trả None cho cả 3 nếu không đủ data hoặc tính toán thất bại.
    """
    if len(rr_ms) < RR_MIN_FOR_LFHF:
        return {"lf_ms2": None, "hf_ms2": None, "lf_hf": None}
    try:
        t = np.cumsum(rr_ms) / 1000.0  # seconds
        t_uniform = np.arange(t[0], t[-1], 1.0 / fs_rr)
        if len(t_uniform) < 16:
            return {"lf_ms2": None, "hf_ms2": None, "lf_hf": None}
        f = interp1d(t, rr_ms, kind="cubic", fill_value="extrapolate")
        rr_uniform = f(t_uniform)
        nperseg = min(256, len(rr_uniform))
        freqs, psd = welch(rr_uniform, fs=fs_rr, nperseg=nperseg, window="hann")
        lf_mask = (freqs >= 0.04) & (freqs < 0.15)
        hf_mask = (freqs >= 0.15) & (freqs < 0.40)
        lf = float(np.trapezoid(psd[lf_mask], freqs[lf_mask]))
        hf = float(np.trapezoid(psd[hf_mask], freqs[hf_mask]))
        lf_hf = lf / hf if hf > 0 else None
        return {
            "lf_ms2": round(lf, 2),
            "hf_ms2": round(hf, 2),
            "lf_hf": round(lf_hf, 2) if lf_hf is not None else None,
        }
    except Exception:
        return {"lf_ms2": None, "hf_ms2": None, "lf_hf": None}


def _compute_hrv_metrics(rr_ms: np.ndarray) -> HRVMetrics:
    """Tính HRV time-domain từ mảng RR (ms). Cần ít nhất 2 intervals."""
    n = len(rr_ms)
    if n < 2:
        return HRVMetrics(
            sdnn_ms=0.0, rmssd_ms=0.0,
            pnn50_pct=0.0, pnn20_pct=0.0,
            mean_nn_ms=0.0, rr_count=n,
            reliability="low",
        )
    sdnn = float(np.std(rr_ms, ddof=1))
    diff_nn = np.diff(rr_ms)
    rmssd = float(np.sqrt(np.mean(diff_nn ** 2)))
    # Task Force 1996 / Ewing 1984: chia (N-1) = len(diff_nn)
    pnn50 = float(np.sum(np.abs(diff_nn) > 50) / len(diff_nn) * 100)
    pnn20 = float(np.sum(np.abs(diff_nn) > 20) / len(diff_nn) * 100)
    mean_nn = float(np.mean(rr_ms))
    lfhf = _compute_lf_hf(rr_ms)

    if n < 10:
        reliability = "low"
    elif n < 60:
        reliability = "medium"
    else:
        reliability = "high"

    return HRVMetrics(
        sdnn_ms=round(sdnn, 2),
        rmssd_ms=round(rmssd, 2),
        pnn50_pct=round(pnn50, 2),
        pnn20_pct=round(pnn20, 2),
        mean_nn_ms=round(mean_nn, 2),
        rr_count=n,
        reliability=reliability,
        lf_ms2=lfhf["lf_ms2"],
        hf_ms2=lfhf["hf_ms2"],
        lf_hf=lfhf["lf_hf"],
    )


def _fallback_find_peaks(
    filtered_signal: np.ndarray, fs: int, rr_snap: List[float]
) -> Tuple[float, List[int], HRVMetrics, float, List[float]]:
    """
    Pure function: fallback scipy find_peaks + adaptive threshold khi heartpy không cài.
    Nhận rr_snap (snapshot từ caller), trả new_rr_list (không mutate global state).
    """
    empty_hrv = HRVMetrics(sdnn_ms=0.0, rmssd_ms=0.0,
                            pnn50_pct=0.0, mean_nn_ms=0.0)

    ma_window = max(int(0.75 * fs), 1)
    ma = np.convolve(filtered_signal, np.ones(ma_window) / ma_window, mode="same")
    threshold = float(np.mean(ma)) * 1.2
    min_distance = max(int(fs * 0.3), 1)

    peaks, _ = find_peaks(
        filtered_signal,
        distance=min_distance,
        prominence=threshold * 0.5,
    )
    if len(peaks) < 2:
        return 0.0, peaks.tolist(), empty_hrv, 0.0, list(rr_snap)

    rr_ms = np.diff(peaks).astype(float) / fs * 1000.0
    rr_ms = _reject_rr_outliers(rr_ms)
    if len(rr_ms) == 0:
        return 0.0, peaks.tolist(), empty_hrv, 0.0, list(rr_snap)

    hr_bpm = float(np.clip(60_000.0 / np.median(rr_ms), HR_MIN_BPM, HR_MAX_BPM))

    new_rr_list = list(rr_snap)
    new_rr_list.extend(float(rr) for rr in rr_ms)
    if len(new_rr_list) > RR_ACCUMULATOR_MAX:
        new_rr_list = new_rr_list[-RR_ACCUMULATOR_MAX:]

    acc_rr = np.array(new_rr_list)
    valid_ratio = len(rr_ms) / max(len(peaks) - 1, 1)
    hr_conf = round(float(np.clip(valid_ratio, 0.0, 1.0)), 2)

    if len(acc_rr) >= RR_MIN_FOR_HRV:
        hrv = _compute_hrv_metrics(acc_rr)
    elif len(rr_ms) >= 2:
        hrv = _compute_hrv_metrics(rr_ms)
    else:
        return round(hr_bpm, 1), peaks.tolist(), empty_hrv, hr_conf, new_rr_list

    return round(hr_bpm, 1), peaks.tolist(), hrv, hr_conf, new_rr_list


def calculate_hr_and_hrv(
    filtered_signal: np.ndarray,
    fs: int,
    overlap_snap: np.ndarray,
    rr_snap: List[float],
) -> Tuple[float, List[int], HRVMetrics, float, np.ndarray, List[float]]:
    """
    Pure function: HeartPy peak detection với overlap buffer + RR accumulator.
    Nhận snapshot (overlap_snap, rr_snap), trả (hr, peaks, hrv, conf, new_overlap, new_rr_list)
    để caller cập nhật global state. Không mutate bất kỳ global dict nào.

    Deduplication logic:
      - combined = [prev_tail 1.5s] + [current 5s]
      - offset = len(prev_tail) = ranh giới overlap / new
      - Peaks với index < offset → vùng overlap, đã đếm lần trước → bỏ qua
      - RR được thêm vào accumulator chỉ khi đỉnh phải (right peak) >= offset
        Điều này bắt được: RR cross-boundary (overlap_last → new_first)
        và RR thuần mới (new → new), không đếm overlap → overlap 2 lần.

    Error handling:
      - hp.process() raise Exception (BadSignalWarning, v.v.) → trả HR=0
      - new_overlap vẫn được trả ra để caller update (batch kế không bị ảnh hưởng)
    """
    empty_hrv = HRVMetrics(sdnn_ms=0.0, rmssd_ms=0.0,
                           pnn50_pct=0.0, mean_nn_ms=0.0)

    if not HEARTPY_AVAILABLE:
        hr, peaks, hrv, conf, new_rr = _fallback_find_peaks(filtered_signal, fs, rr_snap)
        # Fallback không dùng overlap → giữ nguyên snapshot
        return hr, peaks, hrv, conf, overlap_snap, new_rr

    overlap_size = int(OVERLAP_SECONDS * fs)

    # --- Ghép overlap + current ---
    if overlap_snap is not None and len(overlap_snap) > 0:
        tail = overlap_snap[-min(overlap_size, len(overlap_snap)):]
        combined = np.concatenate([tail, filtered_signal])
        offset = len(tail)
    else:
        combined = filtered_signal
        offset = 0

    # Overlap mới: luôn trả ra (dù có lỗi phía dưới, caller cần update)
    new_overlap = filtered_signal[-overlap_size:].copy() if len(filtered_signal) >= overlap_size else filtered_signal.copy()

    # --- HeartPy process ---
    use_hp = (1000 % fs == 0)
    hp_kwargs = dict(
        sample_rate=float(fs), windowsize=0.75,
        bpmmin=float(HR_MIN_BPM), bpmmax=float(HR_MAX_BPM),
        high_precision=use_hp,
    )
    if use_hp:
        hp_kwargs["high_precision_fs"] = 1000.0
    try:
        wd, _ = hp.process(combined, **hp_kwargs)
    except Exception:
        # Tín hiệu quá nhiễu, không tìm được nhịp hợp lệ
        return 0.0, [], empty_hrv, 0.0, new_overlap, list(rr_snap)

    raw_peaks = wd.get("peaklist", [])
    if len(raw_peaks) < 2:
        return 0.0, [], empty_hrv, 0.0, new_overlap, list(rr_snap)

    all_peaks = np.array(raw_peaks, dtype=float)

    # --- Phân vùng: overlap peaks vs new peaks ---
    overlap_peaks = all_peaks[all_peaks < offset]
    new_peaks = all_peaks[all_peaks >= offset]

    # Chỉ số đỉnh trong current batch (integer, cho display trên Mobile)
    display_peaks = [
        int(round(p - offset))
        for p in new_peaks
        if 0 <= int(round(p - offset)) < len(filtered_signal)
    ]

    if len(new_peaks) == 0:
        return 0.0, display_peaks, empty_hrv, 0.0, new_overlap, list(rr_snap)

    # --- Tính RR mới (chỉ những RR có right peak là new peak) ---
    sorted_all = np.sort(np.concatenate([overlap_peaks, new_peaks]))
    rr_all_ms = np.diff(sorted_all) / fs * 1000.0
    right_peaks = sorted_all[1:]
    new_rr_ms = rr_all_ms[right_peaks >= offset]

    # Outlier rejection ±20% median
    new_rr_ms = _reject_rr_outliers(new_rr_ms)

    # --- Tích lũy RR vào accumulator (new list, không mutate) ---
    new_rr_list = list(rr_snap)
    new_rr_list.extend(float(rr) for rr in new_rr_ms)
    if len(new_rr_list) > RR_ACCUMULATOR_MAX:
        new_rr_list = new_rr_list[-RR_ACCUMULATOR_MAX:]

    # --- HR từ batch hiện tại ---
    if len(new_rr_ms) > 0:
        hr_bpm = float(np.clip(60_000.0 / np.median(new_rr_ms), HR_MIN_BPM, HR_MAX_BPM))
    else:
        # Ước lượng từ mật độ đỉnh nếu không có RR hợp lệ
        duration_sec = len(filtered_signal) / fs
        hr_bpm = float(np.clip(len(new_peaks) / duration_sec * 60, HR_MIN_BPM, HR_MAX_BPM))

    # --- HRV: dùng RR tích lũy nếu đủ, không thì dùng batch hiện tại ---
    acc_rr = np.array(new_rr_list)
    if len(acc_rr) >= RR_MIN_FOR_HRV:
        hrv = _compute_hrv_metrics(acc_rr)
    elif len(new_rr_ms) >= 2:
        hrv = _compute_hrv_metrics(new_rr_ms)
    else:
        hrv = empty_hrv

    # --- Confidence score ---
    n_expected_rr = max(len(new_peaks) - 1, 1)
    valid_ratio = len(new_rr_ms) / n_expected_rr
    duration_sec = len(filtered_signal) / fs
    expected_peaks = hr_bpm / 60.0 * duration_sec
    density_ratio = min(len(new_peaks) / expected_peaks, 1.0) if expected_peaks > 0 else 0.0
    hr_conf = round(float(np.clip(0.6 * valid_ratio + 0.4 * density_ratio, 0.0, 1.0)), 2)

    return round(hr_bpm, 1), display_peaks, hrv, hr_conf, new_overlap, new_rr_list


# ============================================================
# SpO2 v2.3 Classical - RMS + Hybrid DC + Piecewise curve
# ============================================================
def calculate_spo2_v23(ir: np.ndarray, red: np.ndarray,
                       fs: int) -> Tuple[Optional[float], float, float, float, str]:
    ir  = np.atleast_1d(np.asarray(ir,  dtype=float))
    red = np.atleast_1d(np.asarray(red, dtype=float))
    ir_ac_sig  = bandpass_filter(ir,  fs, 0.5, 4.0)
    red_ac_sig = bandpass_filter(red, fs, 0.5, 4.0)
    ac_ir = float(np.sqrt(np.mean(ir_ac_sig ** 2)))
    ac_red = float(np.sqrt(np.mean(red_ac_sig ** 2)))

    ir_lp = lowpass_filter(ir, fs, cutoff=0.5)
    red_lp = lowpass_filter(red, fs, cutoff=0.5)
    dc_ir = float((np.median(ir) + np.mean(ir_lp)) / 2.0)
    dc_red = float((np.median(red) + np.mean(red_lp)) / 2.0)

    if dc_ir <= 0 or dc_red <= 0 or ac_ir <= 0 or ac_red <= 0:
        return None, 0.0, 0.0, 0.0, "ac_or_dc_invalid"

    acdc_ir = ac_ir / dc_ir
    acdc_red = ac_red / dc_red

    if not (AC_DC_MIN_IR <= acdc_ir <= AC_DC_MAX_IR):
        return None, 0.0, acdc_ir, acdc_red, "acdc_ir_out_of_range"
    if not (AC_DC_MIN_RED <= acdc_red <= AC_DC_MAX_RED):
        return None, 0.0, acdc_ir, acdc_red, "acdc_red_out_of_range"

    R = acdc_red / acdc_ir
    if not (SPO2_RATIO_MIN <= R <= SPO2_RATIO_MAX):
        return None, R, acdc_ir, acdc_red, "ratio_r_out_of_range"

    if R < 0.7:
        spo2 = 105.5 - 17.0 * R
    elif R < 1.1:
        spo2 = 102.0 - 19.5 * R
    else:
        spo2 = 108.0 - 23.0 * R

    spo2 = float(np.clip(spo2, SPO2_MIN_VALID, SPO2_MAX_VALID))
    return round(spo2, 1), round(R, 3), round(acdc_ir, 5), round(acdc_red, 5), "ok"


def apply_temporal_smoothing(
    spo2_snap: List[float], spo2_raw: float
) -> Tuple[float, List[float]]:
    """Pure: nhận snapshot spo2_history, trả (smoothed_value, new_snap)."""
    new_list = list(spo2_snap)
    new_list.append(spo2_raw)
    if len(new_list) > SPO2_SMOOTH_WINDOW:
        new_list = new_list[-SPO2_SMOOTH_WINDOW:]
    return round(float(np.median(new_list)), 1), new_list


def calculate_spo2_confidence(ir: np.ndarray, red: np.ndarray,
                              acdc_ir: float, acdc_red: float) -> float:
    if len(ir) < 10 or len(red) < 10:
        return 0.0
    # Heuristic: peak-to-peak / std — không phải SNR chuẩn, chỉ dùng làm confidence score
    ir_snr  = (np.max(ir)  - np.min(ir))  / (np.std(ir)  + 1e-6)
    red_snr = (np.max(red) - np.min(red)) / (np.std(red) + 1e-6)
    snr_score = min((ir_snr + red_snr) / 20.0, 1.0)

    def _mid_score(v, lo, hi):
        if v <= lo or v >= hi:
            return 0.0
        mid = (lo + hi) / 2
        return 1.0 - abs(v - mid) / (hi - lo)

    acdc_score = (_mid_score(acdc_ir, AC_DC_MIN_IR, AC_DC_MAX_IR) +
                  _mid_score(acdc_red, AC_DC_MIN_RED, AC_DC_MAX_RED)) / 2
    return round(0.6 * snr_score + 0.4 * acdc_score, 2)


# ============================================================
# Đánh giá chất lượng tín hiệu
# ============================================================
def assess_signal_quality(raw_signal: np.ndarray,
                          filtered_signal: np.ndarray,
                          peaks: list, fs: int) -> str:
    score = 0

    # Spectral purity: tỷ lệ công suất trong band HR [0.5-4.0 Hz] / tổng công suất
    nperseg = min(256, len(filtered_signal) // 2)
    if nperseg >= 16:
        freqs, psd = welch(filtered_signal, fs=fs, nperseg=nperseg, window="hann")
        hr_band = (freqs >= 0.5) & (freqs <= 4.0)
        total_power = float(np.trapezoid(psd, freqs)) + 1e-12
        hr_power = float(np.trapezoid(psd[hr_band], freqs[hr_band]))
        purity = hr_power / total_power
        if purity >= 0.70:
            score += 2
        elif purity >= 0.40:
            score += 1

    duration = len(filtered_signal) / fs
    if len(peaks) >= 2 and duration > 0:
        bpm = len(peaks) / duration * 60
        if 40 <= bpm <= 180:
            score += 2
        elif 30 <= bpm <= 200:
            score += 1

    amplitude = np.max(filtered_signal) - np.min(filtered_signal)
    if amplitude > 0.01 * np.mean(np.abs(raw_signal)):
        score += 1

    if score >= 4:
        return "good"
    if score >= 2:
        return "fair"
    return "poor"


# ============================================================
# Helpers
# ============================================================
def _parse_ts(ts_str):
    try:
        dt = datetime.fromisoformat(ts_str)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return datetime.min.replace(tzinfo=timezone.utc)


def _validate_device_id(device_id: str) -> str:
    try:
        return _check_device_id(device_id)
    except ValueError as e:
        raise HTTPException(400, str(e))


def _require_token(x_device_token: Optional[str] = Header(None)) -> None:
    if not PPG_API_KEY:
        return  # dev mode: auth disabled
    if not x_device_token:
        raise HTTPException(403, "Missing X-Device-Token")
    if not hmac.compare_digest(x_device_token, PPG_API_KEY):
        raise HTTPException(403, "Invalid X-Device-Token")


# ============================================================
# API Endpoints
# ============================================================
@app.get("/")
def root():
    return {
        "status": "running",
        "service": "PPG Processing Server",
        "version": "4.0.0",
        "heartpy_available": HEARTPY_AVAILABLE,
        "features": [
            "HeartPy peak detection (van Gent 2019)" if HEARTPY_AVAILABLE
            else "Fallback scipy find_peaks (adaptive threshold)",
            "Sub-sample accuracy: spline upsample 1000 Hz",
            "Overlap buffer 1.5s + peak deduplication",
            "RR accumulator (tích lũy ≥30 nhịp cho HRV)",
            "Outlier rejection ±20% median(RR)",
            "SpO2 v2.3 (RMS + Hybrid DC + Piecewise)",
            "HRV time-domain (SDNN/RMSSD/pNN50)",
            "Temporal Smoothing SpO2 (median 5)",
        ],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/api/ppg/upload", response_model=PPGResult)
@limiter.limit("20/minute")
def upload_ppg_data(
    request: Request,
    reading: PPGReading,
    _auth: None = Depends(_require_token),
):
    """
    ESP32 gửi dữ liệu thô. Pipeline:
      1. Validate
      2. Bandpass IR → HeartPy peaks → HR + HRV (với overlap dedup + RR accumulator)
      3. SpO2 v2.3 + Tight Validation
      4. Temporal Smoothing SpO2
      5. Signal Quality + Confidence
      6. Lưu DB
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

    logger.info("Upload | device={} samples={} fs={}",
                reading.device_id, len(reading.ir_values), reading.sample_rate)

    device_lock = _get_device_lock(reading.device_id)
    with device_lock:
        # --- Phase A (lock, ngắn): Check MAX_DEVICES + snapshot state ---
        with _state_lock:
            if (reading.device_id not in device_index
                    and len(device_index) >= MAX_DEVICES):
                logger.warning("Max devices reached | device={}", reading.device_id)
                raise HTTPException(429, "Đã đạt số lượng thiết bị tối đa")
            _overlap = overlap_buffer.get(reading.device_id)
            overlap_snap = _overlap.copy() if _overlap is not None else np.array([], dtype=float)
            rr_snap = list(rr_accumulator.get(reading.device_id, []))
            spo2_snap = list(spo2_history.get(reading.device_id, []))

        # --- Phase B (no lock): signal processing, pure computation ---
        filtered_ir = bandpass_filter(ir, fs)
        hr, peaks, hrv, hr_conf, new_overlap, new_rr = calculate_hr_and_hrv(
            filtered_ir, fs, overlap_snap, rr_snap
        )

        spo2_raw, ratio_r, acdc_ir, acdc_red, reject_reason = calculate_spo2_v23(ir, red, fs)

        if spo2_raw is None:
            spo2_smooth = 0.0
            spo2_raw_val = 0.0
            spo2_conf = 0.0
            new_spo2 = None  # không update spo2_history
        else:
            spo2_smooth, new_spo2 = apply_temporal_smoothing(spo2_snap, spo2_raw)
            spo2_raw_val = spo2_raw
            spo2_conf = calculate_spo2_confidence(ir, red, acdc_ir, acdc_red)

        perfusion_index = round(acdc_ir * 100, 2)
        quality = assess_signal_quality(ir, filtered_ir, peaks, fs)

        if quality == "poor":
            logger.warning("Poor signal | device={} hr={} spo2={}",
                           reading.device_id, hr, spo2_smooth)
        logger.debug("Result | device={} HR={} SpO2={} quality={}",
                     reading.device_id, hr, spo2_smooth, quality)

        reading_id = uuid.uuid4().hex
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

        # --- Phase C (lock, ngắn): commit state ---
        with _state_lock:
            # Re-check MAX_DEVICES (race: concurrent upload khác có thể đã chiếm slot)
            if (reading.device_id not in device_index
                    and len(device_index) >= MAX_DEVICES):
                raise HTTPException(429, "Đã đạt số lượng thiết bị tối đa")

            # Eviction trước khi insert (MAX_TOTAL_READINGS)
            if len(readings_db) >= MAX_TOTAL_READINGS:
                oldest_key = min(readings_db, key=lambda k: _parse_ts(readings_db[k]["timestamp"]))
                oldest_device = readings_db[oldest_key]["device_id"]
                del readings_db[oldest_key]
                if oldest_device in device_index:
                    try:
                        device_index[oldest_device].remove(oldest_key)
                    except ValueError:
                        pass
                    if not device_index[oldest_device]:
                        del device_index[oldest_device]
                        overlap_buffer.pop(oldest_device, None)
                        rr_accumulator.pop(oldest_device, None)
                        spo2_history.pop(oldest_device, None)
                        _device_locks.pop(oldest_device, None)
                        logger.info("Evict device | device={}", oldest_device)

            # Commit reading + device_index
            result_stored = result.model_dump()
            result_stored.pop("filtered_signal", None)
            result_stored.pop("peaks", None)
            readings_db[reading_id] = result_stored
            device_index.setdefault(reading.device_id, []).append(reading_id)

            # Commit buffers
            overlap_buffer[reading.device_id] = new_overlap
            rr_accumulator[reading.device_id] = deque(new_rr, maxlen=RR_ACCUMULATOR_MAX)
            if new_spo2 is not None:
                spo2_history[reading.device_id] = deque(new_spo2, maxlen=SPO2_SMOOTH_WINDOW)

            # Per-device MAX_HISTORY eviction
            device_keys = list(device_index.get(reading.device_id, []))
            if len(device_keys) > MAX_HISTORY:
                device_keys.sort(key=lambda k: _parse_ts(readings_db[k]["timestamp"]))
                for old_key in device_keys[:-MAX_HISTORY]:
                    del readings_db[old_key]
                    try:
                        device_index[reading.device_id].remove(old_key)
                    except ValueError:
                        pass

    return result


@app.get("/api/ppg/history/{device_id}")
@limiter.limit("60/minute")
def get_history(
    request: Request,
    device_id: str,
    limit: int = 20,
    _auth: None = Depends(_require_token),
):
    device_id = _validate_device_id(device_id)
    limit = max(1, min(limit, MAX_HISTORY))
    with _state_lock:
        ids = list(device_index.get(device_id, []))
        history = [readings_db[i] for i in ids if i in readings_db]
    history.sort(key=lambda x: _parse_ts(x["timestamp"]), reverse=True)
    return {"device_id": device_id, "count": len(history), "readings": history[:limit]}


@app.get("/api/ppg/latest/{device_id}")
@limiter.limit("60/minute")
def get_latest(
    request: Request,
    device_id: str,
    _auth: None = Depends(_require_token),
):
    device_id = _validate_device_id(device_id)
    with _state_lock:
        ids = list(device_index.get(device_id, []))
        readings = [readings_db[i] for i in ids if i in readings_db]
    if not readings:
        raise HTTPException(404, "Chưa có dữ liệu cho thiết bị này")
    readings.sort(key=lambda x: _parse_ts(x["timestamp"]), reverse=True)
    return readings[0]


@app.get("/api/ppg/stats/{device_id}")
@limiter.limit("60/minute")
def get_device_stats(
    request: Request,
    device_id: str,
    _auth: None = Depends(_require_token),
):
    """Thống kê HR/SpO2/HRV mean/min/max trong 24h qua cho một device."""
    device_id = _validate_device_id(device_id)
    cutoff = datetime.now(timezone.utc) - timedelta(hours=24)

    with _state_lock:
        rids = list(device_index.get(device_id, []))
        candidates = [v for v in (readings_db.get(rid) for rid in rids) if v is not None]

    readings_24h = []
    for v in candidates:
        try:
            ts = datetime.fromisoformat(v["timestamp"])
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            if ts >= cutoff:
                readings_24h.append(v)
        except (ValueError, TypeError):
            continue

    if not readings_24h:
        raise HTTPException(404, "Không có dữ liệu trong 24h qua cho thiết bị này")

    def _stats(values: list) -> dict:
        arr = [x for x in values if x is not None and x > 0]
        if not arr:
            return {"mean": None, "min": None, "max": None, "count": 0}
        return {
            "mean": round(float(np.mean(arr)), 2),
            "min": round(float(np.min(arr)), 2),
            "max": round(float(np.max(arr)), 2),
            "count": len(arr),
        }

    quality_counts: dict = {}
    for v in readings_24h:
        q = v.get("signal_quality", "unknown")
        quality_counts[q] = quality_counts.get(q, 0) + 1

    return {
        "device_id": device_id,
        "period_hours": 24,
        "total_readings": len(readings_24h),
        "signal_quality_distribution": quality_counts,
        "heart_rate_bpm": _stats([v["heart_rate"] for v in readings_24h]),
        "spo2_pct": _stats([v["spo2"] for v in readings_24h]),
        "perfusion_index_pct": _stats([v["perfusion_index"] for v in readings_24h]),
        "hrv": {
            "sdnn_ms": _stats([v["hrv"]["sdnn_ms"] for v in readings_24h]),
            "rmssd_ms": _stats([v["hrv"]["rmssd_ms"] for v in readings_24h]),
            "pnn50_pct": _stats([v["hrv"]["pnn50_pct"] for v in readings_24h]),
        },
        "oldest_reading": min(readings_24h, key=lambda v: _parse_ts(v["timestamp"]))["timestamp"],
        "newest_reading": max(readings_24h, key=lambda v: _parse_ts(v["timestamp"]))["timestamp"],
    }


@app.delete("/api/ppg/history/{device_id}")
@limiter.limit("10/minute")
def clear_history(
    request: Request,
    device_id: str,
    _auth: None = Depends(_require_token),
):
    """Xóa lịch sử, SpO2 buffer, overlap buffer và RR accumulator của device."""
    device_id = _validate_device_id(device_id)
    with _state_lock:
        keys = list(device_index.get(device_id, []))
        for k in keys:
            readings_db.pop(k, None)
        device_index.pop(device_id, None)
        spo2_history.pop(device_id, None)
        overlap_buffer.pop(device_id, None)
        rr_accumulator.pop(device_id, None)
        _device_locks.pop(device_id, None)
    return {"device_id": device_id, "deleted": len(keys)}


@app.get("/api/stats")
def global_stats():
    with _state_lock:
        total_readings = len(readings_db)
        devices = sorted(device_index.keys())
        spo2_buffers = {d: len(buf) for d, buf in spo2_history.items()}
        rr_accumulated = {d: len(buf) for d, buf in rr_accumulator.items()}
    return {
        "total_readings": total_readings,
        "unique_devices": len(devices),
        "devices": devices,
        "spo2_buffers": spo2_buffers,
        "rr_accumulated": rr_accumulated,
    }
