"""
Test ML server response structure tại fs=100Hz (sau firmware v4.1.0 upgrade).

Mục đích: Kiểm tra structure thực tế của ml_predictions.blood_pressure
khi gửi data fs=100Hz lên ML server LIVE (ppg-ml.onrender.com).

Confirm/refute:
- BP=0/0 cả systolic + diastolic, hay chỉ 1 trong 2?
- ml_predictions structure đầy đủ ra sao?
- model_used + confidence khớp với firmware log?
"""

import json
import sys

import numpy as np
import requests

ML_SERVER = "https://ppg-ml.onrender.com"
ENDPOINT = f"{ML_SERVER}/api/ppg/upload"
DEVICE_ID = "test-fs100-replay"
FS = 100
DURATION_S = 5
N_SAMPLES = FS * DURATION_S  # 500

# ----------------------------------------------------------------------
# Synthetic PPG generator — realistic shape, HR ~75 BPM
# ----------------------------------------------------------------------
def generate_synthetic_ppg(fs: int, n_samples: int, hr_bpm: float = 75.0):
    """Tạo synthetic PPG IR + RED giống MAX30102 reflectance.

    - DC offset ~85,000 (giống live test logs)
    - Pulsatile component: gaussian peaks at HR rate
    - SNR cao, không noise nặng (best-case scenario)
    """
    t = np.arange(n_samples) / fs
    period = 60.0 / hr_bpm
    phase = (t % period) / period

    # PPG morphology: skewed pulse (rise nhanh, decay chậm)
    pulse = np.exp(-((phase - 0.15) ** 2) / 0.005)
    pulse += 0.4 * np.exp(-((phase - 0.32) ** 2) / 0.012)  # dicrotic notch echo

    # IR channel: DC ~85k, AC ~3k (PI ~3.5%)
    ir_dc = 85000.0
    ir_ac = 3000.0
    ir = ir_dc + ir_ac * pulse + np.random.normal(0, 50, n_samples)

    # RED channel: DC ~83k, AC ~2.4k → R ratio ~0.6 → SpO2 ~95-98%
    red_dc = 83000.0
    red_ac = 2400.0
    red = red_dc + red_ac * pulse + np.random.normal(0, 50, n_samples)

    return ir.astype(int).tolist(), red.astype(int).tolist()


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    print(f"[INFO] ML server: {ML_SERVER}")
    print(f"[INFO] Endpoint:  {ENDPOINT}")
    print(f"[INFO] fs={FS}Hz, n_samples={N_SAMPLES} ({DURATION_S}s batch)")
    print()

    np.random.seed(42)
    ir, red = generate_synthetic_ppg(FS, N_SAMPLES, hr_bpm=75.0)

    payload = {
        "device_id": DEVICE_ID,
        "ir_values": ir,
        "red_values": red,
        "sample_rate": FS,
    }

    print(f"[POST] payload bytes ~{len(json.dumps(payload))}")
    print()

    try:
        # Render free tier có thể cold start — first call mất 30-60s
        resp = requests.post(ENDPOINT, json=payload, timeout=120)
    except requests.RequestException as exc:
        print(f"[ERROR] Request failed: {exc}")
        sys.exit(1)

    print(f"[RESP] HTTP {resp.status_code}")
    print()

    if resp.status_code != 200:
        print(f"[BODY] {resp.text[:500]}")
        sys.exit(1)

    data = resp.json()

    # Trim noisy fields for readability
    summary = {k: v for k, v in data.items() if k not in ("filtered_signal", "peaks")}

    print("=" * 70)
    print("FULL RESPONSE (filtered_signal + peaks omitted)")
    print("=" * 70)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print()

    # Focused BP analysis
    print("=" * 70)
    print("BP-FOCUSED ANALYSIS")
    print("=" * 70)
    ml = data.get("ml_predictions")
    if ml is None:
        print("ml_predictions = None  ← ML pipeline KHÔNG chạy")
    else:
        print(f"model_used  : {ml.get('model_used')}")
        print(f"confidence  : {ml.get('confidence')}")
        bp = ml.get("blood_pressure", {})
        print(f"BP systolic : {bp.get('systolic')}")
        print(f"BP diastolic: {bp.get('diastolic')}")
        print(f"BP unit     : {bp.get('unit')}")

        # Diagnosis
        sbp = bp.get("systolic", 0) or 0
        dbp = bp.get("diastolic", 0) or 0
        print()
        if sbp == 0 and dbp == 0:
            print("[DIAG] ❌ BOTH zero → likely BOTH models not loaded, hoặc")
            print("       extract_features fail, hoặc PredictionResult.to_dict()")
            print("       zero-out cả 2 fields.")
        elif sbp == 0 and dbp > 0:
            print("[DIAG] ⚠️ SBP=0 only → khớp dự đoán: random_forest_models.pkl")
            print(f"       chỉ chứa DBP. SVR (chứa SBP) không được load.")
            print(f"       DBP={dbp} là realistic, fs=100 match training.")
        elif sbp > 0 and dbp == 0:
            print("[DIAG] ⚠️ DBP=0 only → ngược lại với hypothesis trước")
        else:
            print(f"[DIAG] ✅ Both predicted: {sbp}/{dbp} mmHg")
            print("       Models loaded đúng. Accuracy cần BP cuff để verify.")


if __name__ == "__main__":
    main()
