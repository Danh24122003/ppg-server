"""
ML Pipeline cho PPG Server
Kiến trúc: Feature Extraction → Classical ML / Deep Learning / Transformer → Dự đoán
Dự đoán: Huyết áp (BP), SpO2, Nhịp tim (HR/HRV/AFib), Stress, Glucose
"""

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, welch
from scipy.interpolate import interp1d
from scipy.stats import skew, kurtosis
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import hashlib
import os
import pickle
import logging

logger = logging.getLogger(__name__)

# ============================================================
# Hằng số
# ============================================================
FEATURE_VERSION = "1.0.0"
NUM_TIME_FEATURES = 18
NUM_FREQ_FEATURES = 8
NUM_MORPH_FEATURES = 12
NUM_TOTAL_FEATURES = NUM_TIME_FEATURES + NUM_FREQ_FEATURES + NUM_MORPH_FEATURES  # 38

# Các target dự đoán
TARGET_NAMES = ["sbp", "dbp", "spo2", "heart_rate", "hrv_sdnn", "stress_index", "glucose_estimate", "afib_prob"]

# Khoảng giá trị hợp lệ cho mỗi target
TARGET_RANGES: Dict[str, Tuple[float, float]] = {
    "sbp": (70.0, 200.0),
    "dbp": (40.0, 130.0),
    "spo2": (70.0, 100.0),
    "heart_rate": (30.0, 220.0),
    "hrv_sdnn": (5.0, 300.0),
    "stress_index": (0.0, 100.0),
    "glucose_estimate": (50.0, 400.0),
    "afib_prob": (0.0, 1.0),
}

MODEL_DIR = os.environ.get("MODEL_DIR", "models")

# Khi PPG_STRICT_PKL_VERIFY=1: từ chối load .pkl không có file .sha256 đi kèm.
# Mặc định off để backward-compat với bundle cũ (log WARNING thay vì block).
_STRICT_PKL_VERIFY = os.environ.get("PPG_STRICT_PKL_VERIFY", "0") == "1"


def _verify_pkl_checksum(pkl_path: str) -> bool:
    """Kiểm tra SHA-256 của file .pkl so với sidecar .sha256.

    Dùng cho testing và kiểm tra độc lập bên ngoài _try_load_models.
    Trả True nếu digest khớp hoặc không có sidecar (backward-compat).
    Trả False nếu digest sai hoặc lỗi đọc file.

    Lưu ý: _try_load_models KHÔNG gọi hàm này — nó dùng logic inline
    để tránh TOCTOU (đọc bytes một lần, verify và deserialize trên cùng bytes).
    """
    sha_path = pkl_path + ".sha256"
    if not os.path.exists(sha_path):
        logger.warning(
            "Không có .sha256 cho %s — bỏ qua verify (RCE risk). "
            "Đặt PPG_STRICT_PKL_VERIFY=1 để bật strict mode.",
            pkl_path,
        )
        return True
    try:
        with open(pkl_path, "rb") as f:
            payload = f.read()
        with open(sha_path, "r", encoding="utf-8") as f:
            expected = f.read().strip().split()[0].lower()
        actual = hashlib.sha256(payload).hexdigest().lower()
        if actual != expected:
            logger.error(
                "Checksum không khớp cho %s — từ chối load (RCE risk).", pkl_path
            )
            return False
        return True
    except OSError as exc:
        logger.error("Không đọc được checksum %s: %s — từ chối load.", sha_path, exc)
        return False


# ============================================================
# Data Classes
# ============================================================
@dataclass
class PPGFeatures:
    """Tập hợp tất cả feature trích xuất từ tín hiệu PPG"""
    time_domain: np.ndarray       # (NUM_TIME_FEATURES,)
    frequency_domain: np.ndarray  # (NUM_FREQ_FEATURES,)
    morphological: np.ndarray     # (NUM_MORPH_FEATURES,)
    raw_segment: np.ndarray       # Tín hiệu gốc (dùng cho DL)
    sample_rate: int

    def to_flat_array(self) -> np.ndarray:
        """Ghép tất cả feature thành 1 vector phẳng cho Classical ML"""
        return np.concatenate([
            self.time_domain,
            self.frequency_domain,
            self.morphological,
        ])

    def to_dict(self) -> dict:
        return {
            "time_domain": self.time_domain.tolist(),
            "frequency_domain": self.frequency_domain.tolist(),
            "morphological": self.morphological.tolist(),
            "num_features": len(self.to_flat_array()),
        }


@dataclass
class PredictionResult:
    """Kết quả dự đoán từ ML pipeline"""
    sbp: float = 0.0              # Huyết áp tâm thu (mmHg)
    dbp: float = 0.0              # Huyết áp tâm trương (mmHg)
    spo2: float = 0.0             # Nồng độ oxy (%)
    heart_rate: float = 0.0       # Nhịp tim (BPM)
    hrv_sdnn: float = 0.0         # Biến thiên nhịp tim (ms)
    stress_index: float = 0.0     # Chỉ số stress (0-100)
    glucose_estimate: float = 0.0 # Ước tính glucose (mg/dL)
    afib_prob: float = 0.0        # Xác suất rung nhĩ (0-1)
    model_used: str = ""          # Model nào được dùng
    confidence: float = 0.0       # Độ tin cậy tổng thể (0-1)
    feature_summary: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "blood_pressure": {
                "systolic": round(self.sbp, 1),
                "diastolic": round(self.dbp, 1),
                "unit": "mmHg",
            },
            "spo2": round(self.spo2, 1),
            "heart_rate": round(self.heart_rate, 1),
            "hrv_sdnn": round(self.hrv_sdnn, 1),
            "stress_index": round(self.stress_index, 1),
            # Glucose không trả về giá trị số vì PPG-based glucose cần dataset
            # chuyên biệt và calibration cá nhân — output rule-based sẽ gây hiểu nhầm.
            "glucose_estimate": None,
            "glucose_note": "Chưa hỗ trợ — cần dataset chuyên biệt và calibration cá nhân",
            "afib_probability": round(self.afib_prob, 3),
            "model_used": self.model_used,
            "confidence": round(self.confidence, 3),
            "feature_summary": self.feature_summary,
        }


# ============================================================
# Feature Extraction
# ============================================================
def _safe_bandpass(signal: np.ndarray, fs: int,
                   lowcut: float = 0.5, highcut: float = 4.0,
                   order: int = 4) -> np.ndarray:
    """Bộ lọc thông dải Butterworth an toàn"""
    nyquist = fs / 2.0
    low = max(lowcut / nyquist, 0.001)
    high = min(highcut / nyquist, 0.999)
    if low >= high:
        return signal
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)


def _find_ppg_peaks(signal: np.ndarray, fs: int) -> np.ndarray:
    """Tìm đỉnh systolic trên tín hiệu PPG đã lọc"""
    min_distance = max(int(fs * 60 / 180), 1)
    peaks, _ = find_peaks(
        signal,
        distance=min_distance,
        height=np.mean(signal),
        prominence=0.1 * np.std(signal),
    )
    return peaks


def extract_time_domain_features(signal: np.ndarray, peaks: np.ndarray,
                                  fs: int) -> np.ndarray:
    """
    Trích xuất 18 feature miền thời gian:
    - Thống kê cơ bản: mean, std, skewness, kurtosis, min, max, range, rms
    - Peak-based: avg_ppi, std_ppi, rmssd, pnn50, num_peaks, peak_rate
    - Zero-crossing rate, slope_mean, slope_std, energy
    """
    features = np.zeros(NUM_TIME_FEATURES)

    # Thống kê cơ bản (8 features)
    features[0] = np.mean(signal)
    features[1] = np.std(signal)
    features[2] = float(skew(signal)) if len(signal) > 2 else 0.0
    features[3] = float(kurtosis(signal)) if len(signal) > 3 else 0.0
    features[4] = np.min(signal)
    features[5] = np.max(signal)
    features[6] = np.ptp(signal)  # range = max - min
    features[7] = np.sqrt(np.mean(signal ** 2))  # RMS

    # Peak-based features (6 features)
    if len(peaks) >= 2:
        intervals = np.diff(peaks) / fs  # PPI tính bằng giây
        valid = intervals[(intervals > 60.0 / 180) & (intervals < 60.0 / 30)]
        if len(valid) > 0:
            features[8] = np.mean(valid)            # avg PPI
            features[9] = np.std(valid)              # std PPI
            # RMSSD
            if len(valid) > 1:
                diffs = np.diff(valid)
                features[10] = np.sqrt(np.mean(diffs ** 2))
                # pNN50: tỷ lệ interval liên tiếp chênh > 50ms
                features[11] = np.mean(np.abs(diffs) > 0.05)
    features[12] = len(peaks)                         # num_peaks
    duration = len(signal) / fs
    features[13] = len(peaks) / duration if duration > 0 else 0.0  # peak_rate

    # Zero-crossing rate (1 feature)
    zero_mean = signal - np.mean(signal)
    crossings = np.sum(np.abs(np.diff(np.sign(zero_mean))) > 0)
    features[14] = crossings / len(signal) if len(signal) > 0 else 0.0

    # Slope features (2 features)
    slopes = np.diff(signal)
    features[15] = np.mean(slopes)
    features[16] = np.std(slopes)

    # Energy (1 feature)
    features[17] = np.sum(signal ** 2) / len(signal) if len(signal) > 0 else 0.0

    return features


def _hrv_lf_hf_from_rr(ir_filtered: np.ndarray, fs: float) -> Tuple[float, float, float]:
    """Compute HRV VLF/LF/HF from RR tachogram per Task Force 1996.

    Steps:
    1. Detect peaks in filtered IR signal (distance=fs*0.4 → max ~150 BPM)
    2. Compute RR intervals (ms)
    3. Cubic spline interpolate to uniform 4 Hz grid
    4. Welch PSD (Hann window, nperseg=256)
    5. Trapezoidal integration over Task Force frequency bands:
       - VLF: 0.0033-0.04 Hz
       - LF:  0.04-0.15 Hz
       - HF:  0.15-0.4 Hz

    Reference: Task Force 1996, Circulation 93:1043, PMID 8598068.
    """
    peaks, _ = find_peaks(ir_filtered, distance=int(fs * 0.4))
    if len(peaks) < 10:
        return 0.0, 0.0, 0.0
    rr_ms = np.diff(peaks) / fs * 1000.0

    # Stage 1: Physiological range filter [300-2000 ms] = [30-200 BPM]
    # Per Task Force 1996 (Circulation 93:1043, PMID 8598068)
    physio_mask = (rr_ms >= 300) & (rr_ms <= 2000)
    if physio_mask.sum() < 9:  # cần ít nhất 9 valid RR (= 10 peaks)
        return 0.0, 0.0, 0.0
    rr_ms = rr_ms[physio_mask]

    # Stage 2: HeartPy quotient filter ±20% successive
    # Per van Gent et al. 2019 + Piskorski & Guzik 2005
    if len(rr_ms) >= 2:
        ratios = rr_ms[1:] / rr_ms[:-1]
        quotient_mask = np.concatenate([[True], (ratios >= 0.8) & (ratios <= 1.2)])
        rr_ms = rr_ms[quotient_mask]

    if len(rr_ms) < 9:
        return 0.0, 0.0, 0.0

    t = np.cumsum(rr_ms) / 1000.0
    fs_rr = 4.0
    if t[-1] - t[0] < 1.0 / fs_rr:
        return 0.0, 0.0, 0.0
    t_uniform = np.arange(t[0], t[-1], 1.0 / fs_rr)
    if len(t_uniform) < 16:
        return 0.0, 0.0, 0.0
    rr_uniform = interp1d(
        t, rr_ms, kind='cubic',
        bounds_error=False,
        fill_value=(float(rr_ms[0]), float(rr_ms[-1])),
    )(t_uniform)
    nperseg = min(256, len(rr_uniform))
    freqs_rr, psd_rr = welch(rr_uniform, fs=fs_rr, nperseg=nperseg, window='hann')

    def _band_power(low: float, high: float) -> float:
        mask = (freqs_rr >= low) & (freqs_rr < high)
        if mask.sum() < 2:
            return 0.0
        return float(np.trapezoid(psd_rr[mask], freqs_rr[mask]))

    vlf = _band_power(0.0033, 0.04)
    lf = _band_power(0.04, 0.15)
    hf = _band_power(0.15, 0.4)
    return vlf, lf, hf


def extract_frequency_domain_features(signal: np.ndarray, fs: int) -> np.ndarray:
    """
    Trích xuất 8 feature miền tần số bằng Welch PSD:
    - VLF/LF/HF computed from RR tachogram per Task Force 1996 (NOT raw PPG signal)
    - LF/HF ratio, total power, dominant frequency, spectral entropy, bandwidth
    """
    features = np.zeros(NUM_FREQ_FEATURES)

    # Welch PSD
    nperseg = min(len(signal), fs * 4)  # window 4 giây hoặc toàn bộ
    if nperseg < 16:
        return features
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg)

    if len(psd) == 0:
        return features

    # Band powers helper for non-HRV bands (total, dominant freq, etc.)
    def band_power(f_low: float, f_high: float) -> float:
        mask = (freqs >= f_low) & (freqs <= f_high)
        return float(np.trapezoid(psd[mask], freqs[mask])) if np.any(mask) else 0.0

    # VLF/LF/HF từ RR tachogram per Task Force 1996 (PMID 8598068):
    # PPG signal đã bandpass [0.5, 4.0Hz] nên 3 dải VLF/LF/HF (0.003-0.4Hz)
    # luôn ≈ 0 nếu tính trên PPG. HRV LF/HF phải tính trên chuỗi RR (tachogram).
    vlf, lf, hf = _hrv_lf_hf_from_rr(signal, fs)
    total = band_power(freqs[0], freqs[-1])

    features[0] = vlf
    features[1] = lf
    features[2] = hf
    features[3] = lf / hf if hf > 1e-10 else 0.0   # LF/HF ratio
    features[4] = total

    # Dominant frequency
    features[5] = float(freqs[np.argmax(psd)])

    # Spectral entropy
    psd_norm = psd / (np.sum(psd) + 1e-10)
    psd_norm = psd_norm[psd_norm > 0]
    features[6] = float(-np.sum(psd_norm * np.log2(psd_norm))) if len(psd_norm) > 0 else 0.0

    # Spectral bandwidth (tần số trung bình gia quyền theo PSD)
    features[7] = float(np.sum(freqs * psd) / (np.sum(psd) + 1e-10))

    return features


def extract_morphological_features(signal: np.ndarray, peaks: np.ndarray,
                                    fs: int) -> np.ndarray:
    """
    Trích xuất 12 feature hình thái sóng PPG:
    - Systolic/diastolic amplitude, rise/fall time, pulse width
    - Augmentation index, dicrotic notch, area ratios
    - 1st & 2nd derivative features, waveform symmetry
    """
    features = np.zeros(NUM_MORPH_FEATURES)

    if len(peaks) < 2:
        return features

    # Phân tích từng chu kỳ PPG giữa các đỉnh liên tiếp
    systolic_amps = []
    rise_times = []
    fall_times = []
    pulse_widths = []
    area_ratios = []

    for i in range(len(peaks) - 1):
        start = peaks[i]
        end = peaks[i + 1]
        if end - start < 5:
            continue

        cycle = signal[start:end]
        peak_idx_in_cycle = np.argmax(cycle)

        # Systolic amplitude
        systolic_amps.append(cycle[peak_idx_in_cycle] - cycle[0])

        # Rise time (từ đầu chu kỳ đến đỉnh)
        rise_times.append(peak_idx_in_cycle / fs)

        # Fall time (từ đỉnh đến cuối chu kỳ)
        fall_times.append((len(cycle) - peak_idx_in_cycle) / fs)

        # Pulse width tại 50% biên độ
        half_amp = (cycle[peak_idx_in_cycle] + cycle[0]) / 2
        above_half = np.where(cycle >= half_amp)[0]
        if len(above_half) > 0:
            pulse_widths.append((above_half[-1] - above_half[0]) / fs)

        # Area ratio: systolic area / diastolic area
        if peak_idx_in_cycle > 0 and peak_idx_in_cycle < len(cycle) - 1:
            sys_area = np.trapezoid(cycle[:peak_idx_in_cycle + 1])
            dia_area = np.trapezoid(cycle[peak_idx_in_cycle:])
            if dia_area != 0:
                area_ratios.append(sys_area / dia_area)

    # Gán feature từ thống kê các chu kỳ
    if systolic_amps:
        features[0] = np.mean(systolic_amps)
        features[1] = np.std(systolic_amps)
    if rise_times:
        features[2] = np.mean(rise_times)
    if fall_times:
        features[3] = np.mean(fall_times)
    if pulse_widths:
        features[4] = np.mean(pulse_widths)
        features[5] = np.std(pulse_widths)
    if area_ratios:
        features[6] = np.mean(area_ratios)

    # 1st derivative (velocity) features
    deriv1 = np.diff(signal)
    features[7] = np.max(deriv1)   # max upslope
    features[8] = np.min(deriv1)   # max downslope

    # 2nd derivative (acceleration) features
    if len(signal) > 2:
        deriv2 = np.diff(signal, n=2)
        features[9] = np.max(deriv2)
        features[10] = np.min(deriv2)

    # Waveform symmetry: rise_time / total_cycle_time trung bình
    if rise_times and fall_times:
        total_times = [r + f for r, f in zip(rise_times, fall_times)]
        ratios = [r / t for r, t in zip(rise_times, total_times) if t > 0]
        if ratios:
            features[11] = np.mean(ratios)

    return features


def extract_features(ir_signal: np.ndarray, red_signal: np.ndarray,
                     fs: int) -> PPGFeatures:
    """
    Pipeline trích xuất toàn bộ feature từ tín hiệu PPG thô
    Input: tín hiệu IR + Red raw, sample rate
    Output: PPGFeatures chứa 38 features + raw segment
    """
    # Lọc nhiễu
    filtered_ir = _safe_bandpass(ir_signal, fs)

    # Tìm đỉnh
    peaks = _find_ppg_peaks(filtered_ir, fs)

    # Trích xuất 3 nhóm feature
    time_feats = extract_time_domain_features(filtered_ir, peaks, fs)
    freq_feats = extract_frequency_domain_features(filtered_ir, fs)
    morph_feats = extract_morphological_features(filtered_ir, peaks, fs)

    # Chuẩn bị raw segment cho DL (normalize về [0, 1])
    raw_norm = (ir_signal - np.min(ir_signal)) / (np.ptp(ir_signal) + 1e-10)

    return PPGFeatures(
        time_domain=time_feats,
        frequency_domain=freq_feats,
        morphological=morph_feats,
        raw_segment=raw_norm,
        sample_rate=fs,
    )


# ============================================================
# Base Model Interface
# ============================================================
class BasePPGModel:
    """Interface chung cho tất cả model PPG"""
    name: str = "base"

    def predict(self, features: PPGFeatures) -> Dict[str, float]:
        raise NotImplementedError

    def get_confidence(self, features: PPGFeatures) -> float:
        return 0.0


# ============================================================
# Classical ML Models (SVM, Random Forest, XGBoost)
# ============================================================
class ClassicalMLModel(BasePPGModel):
    """
    Classical ML sử dụng feature vector phẳng (38 features)
    Hỗ trợ: SVM, Random Forest, XGBoost
    Khi chưa train: dùng rule-based estimation từ feature
    """
    name = "classical_ml"

    def __init__(self, model_type: str = "random_forest"):
        self.model_type = model_type  # "svm", "random_forest", "xgboost"
        self.models: Dict[str, object] = {}  # target_name -> trained model
        self.scalers: Dict[str, object] = {}
        self.is_trained = False
        self._try_load_models()

    def _try_load_models(self):
        """Tải model đã train từ disk (nếu có).

        Đọc toàn bộ bytes một lần → verify SHA-256 trên bytes đó →
        pickle.loads(bytes). Không mở file lần thứ hai sau khi verify,
        tránh TOCTOU race condition.
        """
        model_path = os.path.join(MODEL_DIR, f"{self.model_type}_models.pkl")
        if not os.path.exists(model_path):
            return

        # Đọc bytes một lần duy nhất — dùng cho cả verify lẫn deserialize
        try:
            with open(model_path, "rb") as f:
                payload = f.read()
        except OSError as exc:
            logger.warning("Không đọc được %s: %s", model_path, exc)
            return

        # Verify SHA-256 trên bytes đã đọc (không mở lại file)
        sha_path = model_path + ".sha256"
        if os.path.exists(sha_path):
            try:
                with open(sha_path, "r", encoding="utf-8") as f:
                    expected = f.read().strip().split()[0].lower()
                actual = hashlib.sha256(payload).hexdigest().lower()
                if actual != expected:
                    logger.error(
                        "Checksum không khớp cho %s — từ chối load (RCE risk).",
                        model_path,
                    )
                    return
            except OSError as exc:
                logger.error(
                    "Không đọc được checksum %s: %s — từ chối load.", sha_path, exc
                )
                return
        elif _STRICT_PKL_VERIFY:
            logger.error(
                "STRICT mode: không có .sha256 cho %s — từ chối load (RCE risk).",
                model_path,
            )
            return
        else:
            logger.warning(
                "Không có .sha256 cho %s — bỏ qua verify (RCE risk). "
                "Đặt PPG_STRICT_PKL_VERIFY=1 để bật strict mode.",
                model_path,
            )

        try:
            data = pickle.loads(payload)  # deserialize từ bytes đã verify
            self.models = data.get("models", {})
            self.scalers = data.get("scalers", {})
            self.is_trained = True
            logger.info("Đã tải model %s từ %s", self.model_type, model_path)
        except Exception as e:
            logger.warning("Không tải được model: %s", e)

    def predict(self, features: PPGFeatures) -> Dict[str, float]:
        """Dự đoán bằng model đã train hoặc rule-based fallback"""
        X = features.to_flat_array().reshape(1, -1)

        if self.is_trained:
            return self._predict_trained(X)
        return self._predict_rule_based(features)

    def _predict_trained(self, X: np.ndarray) -> Dict[str, float]:
        """Dự đoán bằng model sklearn/xgboost đã train"""
        results = {}
        for target in TARGET_NAMES:
            if target in self.models:
                model = self.models[target]
                scaler = self.scalers.get(target)
                X_scaled = scaler.transform(X) if scaler else X
                pred = float(model.predict(X_scaled)[0])
                lo, hi = TARGET_RANGES[target]
                results[target] = max(lo, min(hi, pred))
            else:
                results[target] = 0.0
        return results

    def _predict_rule_based(self, features: PPGFeatures) -> Dict[str, float]:
        """
        Rule-based estimation khi chưa có model train
        Dùng các mối quan hệ sinh lý đã biết giữa feature PPG và chỉ số sức khỏe
        """
        t = features.time_domain
        f = features.frequency_domain
        m = features.morphological

        # Heart Rate từ peak rate
        peak_rate = t[13]  # peaks per second
        hr = peak_rate * 60 if peak_rate > 0 else 75.0
        hr = max(30.0, min(220.0, hr))

        # HRV SDNN từ PPI std
        ppi_std = t[9]  # std of peak-to-peak intervals
        hrv = ppi_std * 1000 if ppi_std > 0 else 50.0
        hrv = max(5.0, min(300.0, hrv))

        # SpO2: ước tính từ biên độ tín hiệu và tần số features
        # (Trong thực tế cần ratio IR/Red, đây là approximation)
        spo2 = 97.0  # baseline
        if m[0] > 0:  # systolic amplitude
            # Biên độ lớn hơn → tín hiệu tốt → SpO2 có xu hướng cao
            amp_factor = min(m[0] / (np.abs(t[0]) + 1e-10), 0.5)
            spo2 = 95.0 + amp_factor * 5.0
        spo2 = max(70.0, min(100.0, spo2))

        # Huyết áp: ước tính từ pulse transit time proxy (rise time, morphology)
        rise_time = m[2]  # mean rise time
        fall_time = m[3]
        # Rise time ngắn → huyết áp cao (mạch cứng)
        if rise_time > 0:
            sbp = 120.0 - (rise_time - 0.1) * 200  # baseline ± adjustment
        else:
            sbp = 120.0
        sbp = max(70.0, min(200.0, sbp))

        # DBP tương quan với area ratio
        area_ratio = m[6]
        if area_ratio > 0:
            dbp = 80.0 - (area_ratio - 1.0) * 20
        else:
            dbp = 80.0
        dbp = max(40.0, min(130.0, dbp))
        # Đảm bảo SBP > DBP
        if sbp <= dbp:
            sbp = dbp + 20

        # Stress index: LF/HF ratio cao → stress cao
        lf_hf = f[3]
        stress = 30.0 + lf_hf * 15  # baseline 30, scale by LF/HF
        stress = max(0.0, min(100.0, stress))

        # Glucose: ước tính rất thô (PPG-based glucose là research topic)
        glucose = 100.0  # normal fasting baseline
        # Spectral entropy cao → variability cao → có thể liên quan glucose thay đổi
        spectral_entropy = f[6]
        glucose += (spectral_entropy - 3.0) * 5
        glucose = max(50.0, min(400.0, glucose))

        # AFib probability: dựa trên irregularity của PPI
        pnn50 = t[11]  # tỷ lệ interval chênh > 50ms
        rmssd = t[10]
        afib = 0.0
        if pnn50 > 0.3 and rmssd > 0.08:
            afib = min(1.0, pnn50 * 0.8 + rmssd * 2)
        afib = max(0.0, min(1.0, afib))

        return {
            "sbp": round(sbp, 1),
            "dbp": round(dbp, 1),
            "spo2": round(spo2, 1),
            "heart_rate": round(hr, 1),
            "hrv_sdnn": round(hrv, 1),
            "stress_index": round(stress, 1),
            "glucose_estimate": round(glucose, 1),
            "afib_prob": round(afib, 3),
        }

    def get_confidence(self, features: PPGFeatures) -> float:
        if self.is_trained:
            return 0.75
        return 0.4  # rule-based → confidence thấp hơn


# ============================================================
# Deep Learning Models (1D-CNN, LSTM) - NumPy Implementation
# ============================================================
class DeepLearningModel(BasePPGModel):
    """
    1D-CNN + LSTM inference bằng NumPy
    Kiến trúc: Conv1D → ReLU → Pool → LSTM-cell → Dense → Output
    Khi chưa train: dùng random weights (demo) hoặc load pretrained
    """
    name = "deep_learning"

    def __init__(self, model_type: str = "cnn_lstm", hidden_size: int = 64):
        self.model_type = model_type  # "cnn", "lstm", "cnn_lstm"
        self.hidden_size = hidden_size
        self.weights: Optional[dict] = None
        self.is_trained = False
        self._try_load_weights()

    def _try_load_weights(self):
        """Tải weights đã train từ disk"""
        weight_path = os.path.join(MODEL_DIR, f"{self.model_type}_weights.npz")
        if os.path.exists(weight_path):
            try:
                self.weights = dict(np.load(weight_path, allow_pickle=True))
                self.is_trained = True
                logger.info("Đã tải weights %s", self.model_type)
            except Exception as e:
                logger.warning("Không tải được weights: %s", e)

    def _relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        x = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x))

    def _conv1d(self, x: np.ndarray, kernel: np.ndarray, bias: np.ndarray) -> np.ndarray:
        """Conv1D forward pass: x shape (seq_len,), kernel shape (kernel_size, out_channels)"""
        kernel_size, out_ch = kernel.shape
        seq_len = len(x)
        out_len = seq_len - kernel_size + 1
        if out_len <= 0:
            return np.zeros(out_ch)
        output = np.zeros((out_len, out_ch))
        for i in range(out_len):
            segment = x[i:i + kernel_size]
            output[i] = segment @ kernel + bias
        return output

    def _max_pool1d(self, x: np.ndarray, pool_size: int = 2) -> np.ndarray:
        """Max pooling 1D: x shape (seq_len, channels)"""
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        seq_len, ch = x.shape
        out_len = seq_len // pool_size
        if out_len == 0:
            return np.max(x, axis=0, keepdims=True)
        output = np.zeros((out_len, ch))
        for i in range(out_len):
            output[i] = np.max(x[i * pool_size:(i + 1) * pool_size], axis=0)
        return output

    def _lstm_cell(self, x_seq: np.ndarray, h_size: int) -> np.ndarray:
        """
        LSTM forward pass đơn giản
        x_seq: (seq_len, input_size)
        Returns: hidden state cuối cùng (h_size,)
        """
        if x_seq.ndim == 1:
            x_seq = x_seq.reshape(-1, 1)
        seq_len, input_size = x_seq.shape
        total_size = input_size + h_size

        # Khởi tạo hoặc load weights
        if self.weights and "lstm_W" in self.weights:
            W = self.weights["lstm_W"]
            b = self.weights["lstm_b"]
        else:
            # Random initialization (demo mode)
            rng = np.random.RandomState(42)
            W = rng.randn(total_size, 4 * h_size) * 0.1
            b = np.zeros(4 * h_size)

        h = np.zeros(h_size)
        c = np.zeros(h_size)

        for t in range(seq_len):
            combined = np.concatenate([x_seq[t], h])
            gates = combined @ W[:total_size, :4 * h_size] + b[:4 * h_size]

            i_gate = self._sigmoid(gates[:h_size])
            f_gate = self._sigmoid(gates[h_size:2 * h_size])
            g_gate = np.tanh(gates[2 * h_size:3 * h_size])
            o_gate = self._sigmoid(gates[3 * h_size:4 * h_size])

            c = f_gate * c + i_gate * g_gate
            h = o_gate * np.tanh(c)

        return h

    def _forward(self, raw_signal: np.ndarray) -> np.ndarray:
        """
        Forward pass: Raw PPG → CNN → LSTM → Dense → predictions
        Output: (num_targets,) raw scores
        """
        # Resample tín hiệu về 256 điểm cố định
        target_len = 256
        if len(raw_signal) != target_len:
            indices = np.linspace(0, len(raw_signal) - 1, target_len).astype(int)
            x = raw_signal[indices]
        else:
            x = raw_signal.copy()

        h_size = self.hidden_size

        if self.model_type in ("cnn", "cnn_lstm"):
            # Conv1D layer
            if self.weights and "conv_kernel" in self.weights:
                kernel = self.weights["conv_kernel"]
                conv_bias = self.weights["conv_bias"]
            else:
                rng = np.random.RandomState(42)
                kernel = rng.randn(5, 16) * 0.1  # kernel_size=5, out_ch=16
                conv_bias = np.zeros(16)

            conv_out = self._conv1d(x, kernel, conv_bias)
            conv_out = self._relu(conv_out)
            conv_out = self._max_pool1d(conv_out, pool_size=4)
        else:
            conv_out = x.reshape(-1, 1)

        if self.model_type in ("lstm", "cnn_lstm"):
            # LSTM layer
            lstm_out = self._lstm_cell(conv_out, h_size)
        else:
            # Global average pooling cho CNN-only
            lstm_out = np.mean(conv_out, axis=0)
            if len(lstm_out) < h_size:
                lstm_out = np.pad(lstm_out, (0, h_size - len(lstm_out)))

        # Dense output layer
        if self.weights and "dense_W" in self.weights:
            W_out = self.weights["dense_W"]
            b_out = self.weights["dense_b"]
        else:
            rng = np.random.RandomState(123)
            W_out = rng.randn(h_size, len(TARGET_NAMES)) * 0.1
            b_out = np.array([120, 80, 97, 75, 50, 30, 100, 0.05])  # baseline biases

        output = lstm_out[:h_size] @ W_out[:h_size] + b_out
        return output

    def predict(self, features: PPGFeatures) -> Dict[str, float]:
        raw_scores = self._forward(features.raw_segment)
        results = {}
        for i, target in enumerate(TARGET_NAMES):
            lo, hi = TARGET_RANGES[target]
            val = float(raw_scores[i]) if i < len(raw_scores) else 0.0
            results[target] = round(max(lo, min(hi, val)), 1)
        # AFib là probability → clamp [0, 1]
        results["afib_prob"] = round(max(0.0, min(1.0, results["afib_prob"])), 3)
        return results

    def get_confidence(self, features: PPGFeatures) -> float:
        if self.is_trained:
            return 0.8
        return 0.3  # random weights → low confidence


# ============================================================
# Transformer Model (Self-Attention)
# ============================================================
class TransformerModel(BasePPGModel):
    """
    Transformer với Self-Attention cho chuỗi PPG
    Kiến trúc: Positional Encoding → Multi-Head Attention → FFN → Dense
    """
    name = "transformer"

    def __init__(self, d_model: int = 32, num_heads: int = 4, seq_len: int = 128):
        self.d_model = d_model
        self.num_heads = num_heads
        self.seq_len = seq_len
        self.weights: Optional[dict] = None
        self.is_trained = False
        self._try_load_weights()

    def _try_load_weights(self):
        weight_path = os.path.join(MODEL_DIR, "transformer_weights.npz")
        if os.path.exists(weight_path):
            try:
                self.weights = dict(np.load(weight_path, allow_pickle=True))
                self.is_trained = True
            except Exception:
                pass

    def _positional_encoding(self, seq_len: int, d_model: int) -> np.ndarray:
        """Sinusoidal positional encoding"""
        pe = np.zeros((seq_len, d_model))
        position = np.arange(seq_len).reshape(-1, 1)
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term[:d_model // 2])
        return pe

    def _scaled_dot_product_attention(self, Q: np.ndarray, K: np.ndarray,
                                       V: np.ndarray) -> np.ndarray:
        """
        Scaled dot-product attention
        Q, K, V: (seq_len, d_k)
        """
        d_k = Q.shape[-1]
        scores = (Q @ K.T) / np.sqrt(d_k)
        # Softmax
        scores_max = scores - np.max(scores, axis=-1, keepdims=True)
        exp_scores = np.exp(scores_max)
        attn_weights = exp_scores / (np.sum(exp_scores, axis=-1, keepdims=True) + 1e-10)
        return attn_weights @ V

    def _multi_head_attention(self, x: np.ndarray) -> np.ndarray:
        """Multi-head self-attention"""
        seq_len, d_model = x.shape
        d_k = d_model // self.num_heads
        if d_k == 0:
            return x

        rng = np.random.RandomState(42)

        heads = []
        for h in range(self.num_heads):
            key = f"head_{h}"
            if self.weights and f"{key}_Wq" in self.weights:
                Wq = self.weights[f"{key}_Wq"]
                Wk = self.weights[f"{key}_Wk"]
                Wv = self.weights[f"{key}_Wv"]
            else:
                Wq = rng.randn(d_model, d_k) * 0.1
                Wk = rng.randn(d_model, d_k) * 0.1
                Wv = rng.randn(d_model, d_k) * 0.1

            Q = x @ Wq
            K = x @ Wk
            V = x @ Wv
            head_out = self._scaled_dot_product_attention(Q, K, V)
            heads.append(head_out)

        concat = np.concatenate(heads, axis=-1)

        # Output projection
        if self.weights and "Wo" in self.weights:
            Wo = self.weights["Wo"]
        else:
            out_dim = concat.shape[-1]
            Wo = rng.randn(out_dim, d_model) * 0.1
        return concat @ Wo

    def _feed_forward(self, x: np.ndarray) -> np.ndarray:
        """Position-wise feed-forward network"""
        d_model = x.shape[-1]
        d_ff = d_model * 4
        rng = np.random.RandomState(99)

        if self.weights and "ff_W1" in self.weights:
            W1 = self.weights["ff_W1"]
            b1 = self.weights["ff_b1"]
            W2 = self.weights["ff_W2"]
            b2 = self.weights["ff_b2"]
        else:
            W1 = rng.randn(d_model, d_ff) * 0.1
            b1 = np.zeros(d_ff)
            W2 = rng.randn(d_ff, d_model) * 0.1
            b2 = np.zeros(d_model)

        hidden = np.maximum(0, x @ W1 + b1)  # ReLU
        return hidden @ W2 + b2

    def _forward(self, raw_signal: np.ndarray) -> np.ndarray:
        """
        Transformer forward pass
        Raw PPG → Embed → Positional Encoding → Self-Attention → FFN → Pool → Dense
        """
        # Resample về seq_len cố định
        if len(raw_signal) != self.seq_len:
            indices = np.linspace(0, len(raw_signal) - 1, self.seq_len).astype(int)
            x = raw_signal[indices]
        else:
            x = raw_signal.copy()

        # Input embedding: (seq_len,) → (seq_len, d_model)
        rng = np.random.RandomState(7)
        if self.weights and "embed_W" in self.weights:
            embed_W = self.weights["embed_W"]
        else:
            embed_W = rng.randn(1, self.d_model) * 0.1
        x_embed = x.reshape(-1, 1) @ embed_W  # (seq_len, d_model)

        # Positional encoding
        pe = self._positional_encoding(self.seq_len, self.d_model)
        x_embed = x_embed + pe

        # Self-Attention + residual
        attn_out = self._multi_head_attention(x_embed)
        x_embed = x_embed + attn_out  # residual connection

        # Feed-forward + residual
        ff_out = self._feed_forward(x_embed)
        x_embed = x_embed + ff_out

        # Global average pooling → (d_model,)
        pooled = np.mean(x_embed, axis=0)

        # Dense output
        if self.weights and "out_W" in self.weights:
            W_out = self.weights["out_W"]
            b_out = self.weights["out_b"]
        else:
            W_out = rng.randn(self.d_model, len(TARGET_NAMES)) * 0.1
            b_out = np.array([120, 80, 97, 75, 50, 30, 100, 0.05])

        return pooled @ W_out + b_out

    def predict(self, features: PPGFeatures) -> Dict[str, float]:
        raw_scores = self._forward(features.raw_segment)
        results = {}
        for i, target in enumerate(TARGET_NAMES):
            lo, hi = TARGET_RANGES[target]
            val = float(raw_scores[i]) if i < len(raw_scores) else 0.0
            results[target] = round(max(lo, min(hi, val)), 1)
        results["afib_prob"] = round(max(0.0, min(1.0, results["afib_prob"])), 3)
        return results

    def get_confidence(self, features: PPGFeatures) -> float:
        if self.is_trained:
            return 0.85
        return 0.25


# ============================================================
# Ensemble / Multi-Model Fusion
# ============================================================
class EnsemblePredictor:
    """
    Kết hợp kết quả từ nhiều model bằng weighted average
    Weights dựa trên confidence của từng model
    """

    def __init__(self, models: Optional[List[BasePPGModel]] = None):
        if models is None:
            # Classical luôn có mặt (có rule-based fallback khi chưa train)
            self.models = [ClassicalMLModel("random_forest")]
            # DL và Transformer chỉ vào ensemble KHI đã có trained weights.
            # Nếu chưa train, weights random sẽ kéo kết quả ensemble lệch hẳn.
            dl = DeepLearningModel("cnn_lstm")
            if dl.is_trained:
                self.models.append(dl)
                logger.info("Ensemble: thêm DeepLearningModel (đã trained)")
            transformer = TransformerModel()
            if transformer.is_trained:
                self.models.append(transformer)
                logger.info("Ensemble: thêm TransformerModel (đã trained)")
            logger.info("Ensemble khởi tạo với %d model", len(self.models))
        else:
            self.models = models

    def predict(self, features: PPGFeatures) -> PredictionResult:
        """
        Chạy tất cả model, lấy weighted average theo confidence
        """
        predictions = []
        confidences = []

        for model in self.models:
            try:
                pred = model.predict(features)
                conf = model.get_confidence(features)
                predictions.append(pred)
                confidences.append(conf)
            except Exception as e:
                logger.warning("Model %s lỗi: %s", model.name, e)

        if not predictions:
            return PredictionResult(model_used="none", confidence=0.0)

        # Weighted average
        total_conf = sum(confidences)
        if total_conf == 0:
            weights = [1.0 / len(predictions)] * len(predictions)
        else:
            weights = [c / total_conf for c in confidences]

        final = {}
        for target in TARGET_NAMES:
            weighted_sum = sum(
                w * p.get(target, 0.0) for w, p in zip(weights, predictions)
            )
            lo, hi = TARGET_RANGES[target]
            final[target] = round(max(lo, min(hi, weighted_sum)), 1)

        # AFib clamp
        final["afib_prob"] = round(max(0.0, min(1.0, final["afib_prob"])), 3)

        model_names = [m.name for m in self.models[:len(predictions)]]
        avg_confidence = sum(confidences) / len(confidences)

        return PredictionResult(
            sbp=final["sbp"],
            dbp=final["dbp"],
            spo2=final["spo2"],
            heart_rate=final["heart_rate"],
            hrv_sdnn=final["hrv_sdnn"],
            stress_index=final["stress_index"],
            glucose_estimate=final["glucose_estimate"],
            afib_prob=final["afib_prob"],
            model_used="+".join(model_names),
            confidence=round(avg_confidence, 3),
            feature_summary=features.to_dict(),
        )


# ============================================================
# Training Utilities
# ============================================================
def generate_synthetic_training_data(num_samples: int = 500,
                                      fs: int = 100,
                                      duration: float = 5.0) -> Tuple[List[PPGFeatures], Dict[str, np.ndarray]]:
    """
    Tạo dữ liệu PPG giả lập để demo training pipeline
    Mỗi mẫu: tín hiệu PPG tổng hợp với HR, BP, SpO2 biết trước
    """
    rng = np.random.RandomState(42)
    num_points = int(fs * duration)
    t = np.linspace(0, duration, num_points)

    all_features = []
    labels = {target: [] for target in TARGET_NAMES}

    for _ in range(num_samples):
        # Random physiological parameters
        hr = rng.uniform(50, 120)           # BPM
        sbp = rng.uniform(90, 170)          # mmHg
        dbp = rng.uniform(50, 110)          # mmHg
        spo2_val = rng.uniform(88, 100)     # %
        stress = rng.uniform(10, 80)
        glucose = rng.uniform(70, 200)
        hrv_val = rng.uniform(15, 120)

        # Tạo tín hiệu PPG tổng hợp
        freq = hr / 60.0
        # Sóng cơ bản
        ppg = np.sin(2 * np.pi * freq * t)
        # Thêm harmonics (hình dạng sóng PPG thực tế)
        ppg += 0.3 * np.sin(4 * np.pi * freq * t + 0.5)
        ppg += 0.1 * np.sin(6 * np.pi * freq * t + 1.0)
        # Thêm DC offset (phụ thuộc SpO2)
        dc_level = 50000 + (spo2_val - 95) * 1000
        ppg = ppg * 500 + dc_level
        # Thêm nhiễu
        ppg += rng.randn(num_points) * 50
        # Red signal (cho SpO2 calculation)
        red = ppg * (0.8 + (100 - spo2_val) * 0.01) + rng.randn(num_points) * 30

        ir = ppg.astype(float)
        red = red.astype(float)

        features = extract_features(ir, red, fs)
        all_features.append(features)

        labels["sbp"].append(sbp)
        labels["dbp"].append(dbp)
        labels["spo2"].append(spo2_val)
        labels["heart_rate"].append(hr)
        labels["hrv_sdnn"].append(hrv_val)
        labels["stress_index"].append(stress)
        labels["glucose_estimate"].append(glucose)
        labels["afib_prob"].append(rng.uniform(0, 0.3))

    labels = {k: np.array(v) for k, v in labels.items()}
    return all_features, labels


def train_classical_models(features_list: List[PPGFeatures],
                            labels: Dict[str, np.ndarray],
                            model_type: str = "random_forest") -> ClassicalMLModel:
    """
    Train classical ML models cho tất cả target
    Yêu cầu: scikit-learn đã cài
    """
    try:
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.svm import SVR
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import cross_val_score
    except ImportError:
        logger.error("Cần cài scikit-learn: pip install scikit-learn")
        raise

    # Tạo feature matrix
    X = np.array([f.to_flat_array() for f in features_list])

    model_cls_map = {
        "random_forest": lambda: RandomForestRegressor(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
        ),
        "xgboost": lambda: GradientBoostingRegressor(
            n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
        ),
        "svm": lambda: SVR(kernel="rbf", C=10, epsilon=0.1),
    }

    model_obj = ClassicalMLModel(model_type)

    for target in TARGET_NAMES:
        y = labels[target]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        create_fn = model_cls_map.get(model_type, model_cls_map["random_forest"])
        reg = create_fn()
        reg.fit(X_scaled, y)

        # Cross-validation score
        scores = cross_val_score(reg, X_scaled, y, cv=5, scoring="r2")
        logger.info("Target %s: R² = %.3f ± %.3f", target, scores.mean(), scores.std())

        model_obj.models[target] = reg
        model_obj.scalers[target] = scaler

    model_obj.is_trained = True

    # Lưu model với SHA-256 sidecar (đồng bộ với cơ chế verify trong _try_load_models)
    import tempfile
    os.makedirs(MODEL_DIR, exist_ok=True)
    save_path = os.path.join(MODEL_DIR, f"{model_type}_models.pkl")
    sha_path = save_path + ".sha256"
    payload = pickle.dumps({"models": model_obj.models, "scalers": model_obj.scalers})
    digest = hashlib.sha256(payload).hexdigest()
    with tempfile.NamedTemporaryFile(dir=MODEL_DIR, suffix=".tmp", delete=False) as tf:
        tmp_path = tf.name
        tf.write(payload)
    try:
        os.replace(tmp_path, save_path)
        with open(sha_path, "w", encoding="utf-8") as sf:
            sf.write(digest + "\n")
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
    logger.info("Đã lưu model %s vào %s (sha256=%s...)", model_type, save_path, digest[:12])

    return model_obj
