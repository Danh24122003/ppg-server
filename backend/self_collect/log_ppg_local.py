"""
Laptop FastAPI server tạm thay Render khi thu paired (PPG, BP) data local.

WORKFLOW (mỗi subject ~12-15 phút):
  1. Chạy script: `python log_ppg_local.py`
  2. Server listen 0.0.0.0:8080 — note IP laptop in ra console
  3. Đổi firmware ESP32 serverUrl → "http://<laptop_ip>:8080/api/ppg/upload"
     (chỉ cần làm 1 LẦN, rebuild + flash)
  4. Per subject: nhập metadata + BP cuff Omron qua console → ENTER
  5. ESP32 stream PPG ~5 phút → server save CSV
  6. ENTER lần nữa để stop session → tự động save file
  7. Lặp cho subject tiếp theo

OUTPUT:
  collected_data/<subject_id>_<timestamp>.csv với metadata header + raw IR/Red

CSV FORMAT:
  # subject_id=S001,age=25,sex=M,height_cm=175,weight_kg=70,bmi=22.9
  # sbp_baseline=118,dbp_baseline=76,sbp_post=120,dbp_post=78
  # hypertension=n,medication=n,smoking=n,finger=middle,cuff_arm=left
  # session_date=2026-04-28T17:30:00,fs=100,duration_s=300
  # notes=bạn cùng phòng
  timestamp_ms,ir,red
  1714299600000,85432,83122
  1714299600010,85445,83135
  ...

REQUIREMENTS: pip install fastapi uvicorn (đã có sẵn từ backend project)
"""

from __future__ import annotations

import csv
import datetime
import os
import socket
import sys
import threading
import time

# Windows console mặc định CP1252 không in được tiếng Việt — force UTF-8.
# Phải làm SỚM, trước khi any print() chạy.
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

HERE = Path(__file__).resolve().parent
COLLECTED_DIR = HERE.parent.parent / "collected_data"
COLLECTED_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Import pipeline production từ Backend/main.py để hiển thị
# HR/SpO2/HRV/quality THẬT khi demo với thầy hướng dẫn.
# Cùng code production → số liệu defensible.
# ============================================================
_BACKEND_DIR = HERE.parent  # .../backend/
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))
os.environ.setdefault("PPG_API_KEY", "")  # tắt auth khi import main.py

try:
    from main import (  # type: ignore
        bandpass_filter,
        calculate_hr_and_hrv,
        calculate_spo2_v23,
        apply_temporal_smoothing,
        calculate_spo2_confidence,
        assess_signal_quality,
        SPO2_SMOOTH_WINDOW,
        RR_ACCUMULATOR_MAX,
        MIN_FINGER_DC,
    )
    try:
        # Tắt loguru info/debug spam của main.py — chỉ giữ WARNING+
        from loguru import logger as _bk_logger
        _bk_logger.remove()
        _bk_logger.add(sys.stderr, level="WARNING")
    except Exception:
        pass
    _PIPELINE_OK = True
    print("[INIT] Pipeline production loaded OK (HR/SpO2/HRV real-time)")
except Exception as _exc:
    print(f"[WARN] Không import được pipeline: {_exc}")
    print("       Chạy mode raw-only (không hiển thị HR/SpO2/HRV).")
    _PIPELINE_OK = False
    MIN_FINGER_DC = 1000
    SPO2_SMOOTH_WINDOW = 5
    RR_ACCUMULATOR_MAX = 240


# ============================================================
# Schema khớp ESP32 firmware v4.1.0 payload
# ============================================================
class PPGBatch(BaseModel):
    device_id: str = Field(min_length=1, max_length=64)
    ir_values: List[int]
    red_values: List[int]
    sample_rate: int = Field(ge=25, le=400)


# ============================================================
# Trạng thái session (shared giữa FastAPI + console thread)
# ============================================================
@dataclass
class SubjectSession:
    """Snapshot 1 phiên đo của 1 subject."""
    subject_id: str
    age: int
    sex: str
    height_cm: float
    weight_kg: float
    sbp_baseline: int
    dbp_baseline: int
    hypertension: str = "n"
    medication: str = "n"
    smoking: str = "n"
    finger: str = "middle"
    cuff_arm: str = "left"
    notes: str = ""
    sbp_post: Optional[int] = None
    dbp_post: Optional[int] = None
    started_at: datetime.datetime = field(default_factory=datetime.datetime.now)

    # Buffer raw samples — append từng batch
    fs: Optional[int] = None
    samples_ts: List[int] = field(default_factory=list)  # ms
    samples_ir: List[int] = field(default_factory=list)
    samples_red: List[int] = field(default_factory=list)

    @property
    def bmi(self) -> float:
        if self.height_cm <= 0:
            return 0.0
        h = self.height_cm / 100.0
        return round(self.weight_kg / (h * h), 1)

    def n_samples(self) -> int:
        return len(self.samples_ir)


# Global state (single-subject single-session at a time)
_state_lock = threading.Lock()
_current: Optional[SubjectSession] = None
_batch_count = 0

# Pending-batch warning state — track khi ESP32 đang gửi mà chưa có subject active.
# Reset mỗi khi user nhập metadata xong (collect_subject return → console_loop set _current).
_pending_lock = threading.Lock()
_pending_count = 0  # số batch bị reject 503
_pending_first_seen: Optional[float] = None  # time.time() khi POST đầu bị reject
_pending_warned_30s = False  # đã in cảnh báo 30s timeout chưa

# Per-device pipeline state (tách biệt với main.py globals — main.py không chạy uvicorn ở đây).
# Mirror cấu trúc Backend/main.py: overlap buffer 1.5s + RR accumulator + SpO2 smoothing window.
_pipeline_lock = threading.Lock()
_overlap_buffer_demo: Dict[str, np.ndarray] = {}
_rr_accumulator_demo: Dict[str, list] = {}
_spo2_history_demo: Dict[str, deque] = {}


# ============================================================
# FastAPI server
# ============================================================
app = FastAPI(title="PPG Local Collector", version="1.0.0")


@app.get("/")
def health():
    with _state_lock:
        sub = _current.subject_id if _current else None
        n = _current.n_samples() if _current else 0
    return {"status": "ok", "current_subject": sub, "samples_collected": n}


@app.post("/api/ppg/upload")
def upload(batch: PPGBatch):
    """Nhận batch raw IR/Red từ ESP32 → append vào session hiện tại."""
    global _batch_count, _pending_count, _pending_first_seen, _pending_warned_30s

    with _state_lock:
        current_active = _current is not None

    if not current_active:
        # ESP32 đang gửi nhưng chưa có subject active → in cảnh báo rõ ràng + reject
        with _pending_lock:
            now = time.time()
            _pending_count += 1
            if _pending_first_seen is None:
                _pending_first_seen = now
            elapsed = now - _pending_first_seen

            # In cảnh báo theo mốc tăng dần (tránh spam mỗi batch)
            if _pending_count == 1:
                print()
                print(">>> [WARN] ESP32 đang gửi batch nhưng S01ƯA có subject active!")
                print(">>>        Quay lại Terminal này, gõ Subject ID + ENTER để bắt đầu.")
                print()
            elif _pending_count == 5:
                print()
                print(f">>> [WARN] Đã reject {_pending_count} batches — ESP32 vẫn đang đợi.")
                print(">>>        Nhập metadata NGAY hoặc batch sẽ tiếp tục bị drop.")
                print()
            elif elapsed > 30 and not _pending_warned_30s:
                _pending_warned_30s = True
                print()
                print("=" * 64)
                print(">>> [TIMEOUT 30s] ESP32 đã chờ 30 giây không có subject!")
                print(f">>>               {_pending_count} batches đã bị drop.")
                print(">>> Kiểm tra Terminal: bạn đang ở prompt 'Subject ID:' phải không?")
                print(">>>   - NẾU có: gõ Subject ID + ENTER ngay")
                print(">>>   - NẾU không: Ctrl+C exit + run lại log_ppg_local.py")
                print("=" * 64)
                print()

        raise HTTPException(
            status_code=503,
            detail="No active subject session — nhập metadata trên console",
        )

    # Có subject active → reset pending warning state cho lần next idle
    with _pending_lock:
        if _pending_count > 0:
            print(f">>> [INFO] Subject active, resume nhận batch (đã drop {_pending_count} batches trước đó)")
            _pending_count = 0
            _pending_first_seen = None
            _pending_warned_30s = False

    with _state_lock:
        # Re-check vì có thể subject đã được reset giữa 2 block (race với console_loop)
        if _current is None:
            raise HTTPException(
                status_code=503,
                detail="Subject session ended trước khi append samples",
            )
        if len(batch.ir_values) != len(batch.red_values):
            raise HTTPException(status_code=400, detail="ir/red length mismatch")

        if _current.fs is None:
            _current.fs = batch.sample_rate

        # Append samples với timestamp tăng dần theo fs.
        # Float division + round tránh integer truncation:
        # vd fs=111 → period chính xác 9.009ms thay vì 9 (truncate khi dùng //).
        # Fix bug 2026-05-03: trước dùng `1000 // fs` gây drift fs về 111Hz cho
        # firmware claim 100Hz (period_ms=10 OK với fs=100 nhưng nếu firmware
        # gửi rate khác do timing artifact thì truncate sai).
        ts0 = int(datetime.datetime.now().timestamp() * 1000)
        period_ms_float = 1000.0 / batch.sample_rate
        for i, (ir, red) in enumerate(zip(batch.ir_values, batch.red_values)):
            ts = int(round(ts0 + i * period_ms_float))
            _current.samples_ts.append(ts)
            _current.samples_ir.append(ir)
            _current.samples_red.append(red)

        _batch_count += 1
        n = _current.n_samples()
        sub = _current.subject_id

    # ============================================================
    # Demo: chạy pipeline production trên batch này → show metrics
    # ============================================================
    hr = 0.0
    hr_conf = 0.0
    spo2_smooth = 0.0
    spo2_raw_val = 0.0
    spo2_conf = 0.0
    ratio_r = 0.0
    perfusion_index = 0.0
    quality = "unknown"
    reject_reason: Optional[str] = None
    hrv_dict = {
        "sdnn_ms": 0.0, "rmssd_ms": 0.0, "pnn50_pct": 0.0, "pnn20_pct": 0.0,
        "mean_nn_ms": 0.0, "rr_count": 0, "reliability": "low",
        "lf_ms2": None, "hf_ms2": None, "lf_hf": None,
    }

    ir_arr = np.asarray(batch.ir_values, dtype=float)
    red_arr = np.asarray(batch.red_values, dtype=float)
    fs = batch.sample_rate
    finger_ok = float(np.mean(ir_arr)) >= MIN_FINGER_DC

    if _PIPELINE_OK and len(ir_arr) >= 50 and finger_ok:
        try:
            did = batch.device_id
            with _pipeline_lock:
                overlap_snap = _overlap_buffer_demo.get(
                    did, np.array([], dtype=float)
                ).copy()
                rr_snap = list(_rr_accumulator_demo.get(did, []))
                spo2_snap = list(_spo2_history_demo.get(did, []))

            # Pipeline pure-compute (không cần lock)
            filtered_ir = bandpass_filter(ir_arr, fs)
            hr, peaks, hrv_obj, hr_conf, new_overlap, new_rr = calculate_hr_and_hrv(
                filtered_ir, fs, overlap_snap, rr_snap
            )
            spo2_raw, ratio_r_v, acdc_ir, acdc_red, reject = calculate_spo2_v23(
                ir_arr, red_arr, fs
            )
            ratio_r = float(ratio_r_v)
            reject_reason = None if reject == "ok" else reject

            new_spo2 = None
            if spo2_raw is not None:
                spo2_smooth, new_spo2 = apply_temporal_smoothing(spo2_snap, spo2_raw)
                spo2_raw_val = float(spo2_raw)
                spo2_conf = calculate_spo2_confidence(
                    ir_arr, red_arr, acdc_ir, acdc_red
                )

            perfusion_index = round(float(acdc_ir) * 100, 2)
            quality = assess_signal_quality(ir_arr, filtered_ir, peaks, fs)
            hrv_dict = hrv_obj.model_dump()

            # Commit state
            with _pipeline_lock:
                _overlap_buffer_demo[did] = new_overlap
                _rr_accumulator_demo[did] = list(new_rr)[-RR_ACCUMULATOR_MAX:]
                if new_spo2 is not None:
                    _spo2_history_demo[did] = deque(
                        new_spo2, maxlen=SPO2_SMOOTH_WINDOW
                    )
        except Exception as exc:
            # Advance overlap buffer ngay cả khi pipeline throw — nếu không, stale
            # buffer cũ ghép với batch sau gây HR display stuck cross-batch.
            # Production main.py:457-459 luôn return new_overlap kể cả lỗi HeartPy.
            print(f"  [WARN] Pipeline error: {exc}")
            try:
                with _pipeline_lock:
                    overlap_size = int(1.5 * fs)
                    _overlap_buffer_demo[did] = (
                        ir_arr[-overlap_size:].copy()
                        if len(ir_arr) >= overlap_size
                        else ir_arr.copy()
                    )
            except Exception:
                pass
    elif _PIPELINE_OK and not finger_ok:
        quality = "no_finger"

    # ============================================================
    # Compact display cho demo (3 dòng/batch khi đủ data)
    # ============================================================
    print(
        f"  [BATCH #{_batch_count:03d}] {sub} fs={batch.sample_rate} "
        f"+{len(batch.ir_values)} → total={n} samples"
    )
    if _PIPELINE_OK:
        spo2_disp = f"{spo2_smooth:5.1f}%" if spo2_smooth > 0 else " --.- "
        rej_disp = f" rej={reject_reason}" if reject_reason else ""
        print(
            f"    HR={hr:5.1f} BPM (conf={hr_conf:.2f}) | "
            f"SpO2={spo2_disp} (raw={spo2_raw_val:.1f} R={ratio_r:.2f} "
            f"PI={perfusion_index:.2f}%){rej_disp} | quality={quality}"
        )
        if hrv_dict["rr_count"] > 0:
            lfhf_str = (
                f" LF/HF={hrv_dict['lf_hf']:.2f}"
                if hrv_dict.get("lf_hf") is not None
                else ""
            )
            print(
                f"    HRV: RR={hrv_dict['rr_count']} ({hrv_dict['reliability']}) "
                f"SDNN={hrv_dict['sdnn_ms']:.1f} RMSSD={hrv_dict['rmssd_ms']:.1f} "
                f"pNN50={hrv_dict['pnn50_pct']:.1f}% "
                f"meanNN={hrv_dict['mean_nn_ms']:.0f}ms{lfhf_str}"
            )

    # Response cho firmware — giá trị thật thay vì mock
    return {
        "reading_id": f"local-{_batch_count:04d}",
        "device_id": batch.device_id,
        "heart_rate": hr,
        "hr_confidence": hr_conf,
        "spo2": spo2_smooth,
        "spo2_raw": spo2_raw_val,
        "spo2_confidence": spo2_conf,
        "ratio_r": ratio_r,
        "perfusion_index": perfusion_index,
        "hrv": hrv_dict,
        "signal_quality": quality,
        "rejection_reason": reject_reason,
        "ml_predictions": {
            "blood_pressure": {"systolic": 0, "diastolic": 0, "unit": "mmHg"},
            "model_used": "local_collect",
            "confidence": 0.0,
        },
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }


# ============================================================
# Console-driven session control
# ============================================================
def _input_int(prompt: str, lo: int = 0, hi: int = 999) -> int:
    while True:
        try:
            v = int(input(prompt).strip())
            if lo <= v <= hi:
                return v
            print(f"  ! ngoài range [{lo}, {hi}]")
        except ValueError:
            print("  ! không phải số nguyên")
        except EOFError:
            raise  # let caller handle (vd console_loop catch -> save session)


def _input_int_optional(prompt: str, lo: int = 0, hi: int = 999) -> Optional[int]:
    """Như _input_int nhưng ENTER trống = None (skip)."""
    while True:
        try:
            raw = input(prompt).strip()
            if raw == "":
                return None
            v = int(raw)
            if lo <= v <= hi:
                return v
            print(f"  ! ngoài range [{lo}, {hi}]")
        except ValueError:
            print("  ! không phải số nguyên (ENTER trống để skip)")
        except EOFError:
            return None  # pipe closed -> treat as skip


def _input_float(prompt: str, lo: float = 0, hi: float = 999) -> float:
    while True:
        try:
            v = float(input(prompt).strip())
            if lo <= v <= hi:
                return v
            print(f"  ! ngoài range [{lo}, {hi}]")
        except ValueError:
            print("  ! không phải số")


def _input_choice(prompt: str, choices: List[str], default: str = "") -> str:
    cs = "/".join(choices)
    full = f"{prompt} [{cs}]"
    if default:
        full += f" (default={default})"
    full += ": "
    while True:
        v = input(full).strip().lower()
        if v == "" and default:
            return default
        if v in [c.lower() for c in choices]:
            return v
        print(f"  ! chọn 1 trong {choices}")


def collect_subject() -> Optional[SubjectSession]:
    """Prompt user nhập metadata + BP cuff cho 1 subject."""
    print()
    print("=" * 60)
    sid = input("Subject ID (vd S001, gõ 'q' để thoát): ").strip()
    if sid.lower() == "q":
        return None

    age = _input_int("Age (year): ", 1, 120)
    sex = _input_choice("Sex", ["M", "F"]).upper()
    height = _input_float("Height (cm): ", 100, 220)
    weight = _input_float("Weight (kg): ", 30, 200)

    print("\n--- Đo BP cuff Omron BASELINE (mean của 2 lần) ---")
    sbp_baseline = _input_int("  SBP baseline (mmHg): ", 70, 220)
    dbp_baseline = _input_int("  DBP baseline (mmHg): ", 40, 140)

    print("\n--- Subject info ---")
    hypertension = _input_choice("Hypertension history", ["y", "n"], "n")
    medication = _input_choice("On BP medication", ["y", "n"], "n")
    smoking = _input_choice("Smoking", ["y", "n"], "n")
    finger = _input_choice("Recording finger", ["index", "middle", "ring"], "middle")
    cuff_arm = _input_choice("Cuff arm", ["left", "right"], "left")
    notes = input("Notes (optional): ").strip()

    return SubjectSession(
        subject_id=sid,
        age=age,
        sex=sex,
        height_cm=height,
        weight_kg=weight,
        sbp_baseline=sbp_baseline,
        dbp_baseline=dbp_baseline,
        hypertension=hypertension,
        medication=medication,
        smoking=smoking,
        finger=finger,
        cuff_arm=cuff_arm,
        notes=notes,
    )


def save_session(sess: SubjectSession) -> Path:
    """Save session ra CSV với metadata header + raw samples."""
    ts = sess.started_at.strftime("%Y%m%d_%H%M%S")
    out_path = COLLECTED_DIR / f"{sess.subject_id}_{ts}.csv"

    # FIX 2026-05-01: duration_s tính từ TIMESTAMPS THỰC của samples, không phải
    # wall-clock (datetime.now() - started_at). Wall-clock kèm thời gian user thao
    # tác console (nhập BP_post, etc.) nên dài hơn streaming time 19-54%.
    # Bug cũ làm misleading khi compute fs từ duration metadata.
    if len(sess.samples_ts) >= 2:
        duration_s = round((sess.samples_ts[-1] - sess.samples_ts[0]) / 1000.0, 1)
    else:
        # Fallback wall-clock cho session quá ngắn (vd error trước khi có samples)
        duration_s = round((datetime.datetime.now() - sess.started_at).total_seconds(), 1)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        # Metadata header (CSV comments — pandas read_csv với comment='#' parse được)
        f.write(
            f"# subject_id={sess.subject_id},age={sess.age},sex={sess.sex},"
            f"height_cm={sess.height_cm},weight_kg={sess.weight_kg},bmi={sess.bmi}\n"
        )
        f.write(
            f"# sbp_baseline={sess.sbp_baseline},dbp_baseline={sess.dbp_baseline},"
            f"sbp_post={sess.sbp_post or 'NA'},dbp_post={sess.dbp_post or 'NA'}\n"
        )
        f.write(
            f"# hypertension={sess.hypertension},medication={sess.medication},"
            f"smoking={sess.smoking},finger={sess.finger},cuff_arm={sess.cuff_arm}\n"
        )
        f.write(
            f"# session_date={sess.started_at.isoformat()},fs={sess.fs or 'NA'},"
            f"duration_s={duration_s}\n"
        )
        # Notes (escape comma/newline to keep on 1 line)
        safe_notes = sess.notes.replace("\n", " ").replace(",", ";")
        f.write(f"# notes={safe_notes}\n")

        # Body
        writer = csv.writer(f)
        writer.writerow(["timestamp_ms", "ir", "red"])
        for ts_ms, ir, red in zip(sess.samples_ts, sess.samples_ir, sess.samples_red):
            writer.writerow([ts_ms, ir, red])

    return out_path


def get_local_ip() -> str:
    """Tìm IP LAN của laptop để hiển thị cho user (firmware ESP32 cần URL này).

    LƯU Ý: socket.connect("8.8.8.8") có thể pick IP của VPN/Docker/Hyper-V
    (vd 172.16.0.2) thay vì IP WiFi thật (vd 192.168.1.x). User cần check
    bằng `ipconfig` để chọn IP cùng subnet với ESP32.
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def list_all_ipv4() -> List[Tuple[str, str]]:
    """Liệt kê TẤT CẢ IPv4 của laptop, kèm hostname interface (best-effort).

    Trả về [(interface_label, ip), ...] để user thấy hết các option và
    chọn đúng IP cùng subnet với ESP32 (thường là 192.168.x.x WiFi LAN).
    """
    out: List[Tuple[str, str]] = []
    try:
        hostname = socket.gethostname()
        # getaddrinfo trả nhiều entry cho mỗi IP của host
        for entry in socket.getaddrinfo(hostname, None, socket.AF_INET):
            ip = entry[4][0]
            label = "host"
            if ip.startswith("192.168."):
                label = "WiFi/LAN (likely)"
            elif ip.startswith("10."):
                label = "private 10.x"
            elif ip.startswith("172."):
                label = "private 172.x (VPN/Docker?)"
            elif ip.startswith("127."):
                continue  # skip loopback
            if (label, ip) not in out:
                out.append((label, ip))
    except Exception:
        pass
    return out


def console_loop():
    """Main loop: nhận subject → wait recording → ENTER stop → save."""
    global _current, _batch_count

    while True:
        sess = collect_subject()
        if sess is None:
            # Trước khi exit, kiểm tra session đang active để auto-save
            # (user có thể gõ 'q' nhầm sau khi đo xong nhưng quên ENTER)
            with _state_lock:
                active_sess = _current
                _current = None
            if active_sess is not None and active_sess.n_samples() > 0:
                print(f"\n[AUTO-SAVE] Phát hiện session {active_sess.subject_id} "
                      f"đang active với {active_sess.n_samples()} samples — save trước khi exit...")
                try:
                    out = save_session(active_sess)
                    print(f"  [SAVE] {out.name}")
                except Exception as exc:
                    print(f"  [WARN] Save fail: {exc}")
            print("\n[EXIT] Stopping server...")
            os._exit(0)  # force exit toàn bộ process kể cả uvicorn thread

        with _state_lock:
            _current = sess
            _batch_count = 0

        print()
        print(f"  → READY: {sess.subject_id} ({sess.bmi=})")
        print(f"  → ESP32 bắt đầu POST samples lên server...")
        print(f"  → Press ENTER khi đã đo xong (~5 phút) để dừng + save")
        try:
            input()  # block đến khi ENTER
        except KeyboardInterrupt:
            print("\n[INTERRUPT] saving anyway...")

        # Prompt BP post-PPG (optional — ENTER trống để skip)
        print("\n--- Đo BP cuff Omron POST (sau khi xong PPG) ---")
        try:
            sbp_post = _input_int_optional("  SBP post (mmHg, ENTER = skip): ", 70, 220)
            dbp_post = _input_int_optional("  DBP post (mmHg, ENTER = skip): ", 40, 140)
            with _state_lock:
                if _current is not None:
                    _current.sbp_post = sbp_post
                    _current.dbp_post = dbp_post
        except (KeyboardInterrupt, EOFError):
            pass

        # Save
        with _state_lock:
            sess = _current
            _current = None  # ngừng nhận batch

        if sess.n_samples() == 0:
            print(f"  [WARN] {sess.subject_id}: 0 samples, KHÔNG save")
            continue

        out = save_session(sess)
        print(f"  [SAVE] {out.name} — {sess.n_samples()} samples ({sess.fs}Hz)")
        print(f"         duration={(datetime.datetime.now() - sess.started_at).total_seconds():.0f}s")


# ============================================================
# Entry point
# ============================================================
def main():
    port = int(os.environ.get("PORT", 8080))
    primary_ip = get_local_ip()
    all_ips = list_all_ipv4()

    print()
    print("=" * 64)
    print("  PPG Local Collector — laptop FastAPI server")
    print("=" * 64)
    print()
    print("  TẤT CẢ IP của laptop (chọn IP CÙNG SUBNET với ESP32):")
    if all_ips:
        for label, ip in all_ips:
            mark = " ⭐ DEFAULT" if ip == primary_ip else ""
            print(f"    http://{ip}:{port}/api/ppg/upload   ({label}){mark}")
    else:
        print(f"    http://{primary_ip}:{port}/api/ppg/upload")
    print()
    print("  ⚠️  ESP32 thường ở subnet 192.168.x.x (WiFi nhà)")
    print("      KHÔNG dùng IP 172.16.x.x hoặc 10.x.x.x trừ khi biết chắc")
    print("      cùng subnet — đó thường là VPN/Docker/Hyper-V virtual.")
    print()
    print(f"  Output CSV: {COLLECTED_DIR}")
    print(f"  Gõ 'q' để thoát.")
    print("=" * 64)
    print()

    # FastAPI server in background thread
    config = uvicorn.Config(
        app, host="0.0.0.0", port=port, log_level="warning"
    )
    server = uvicorn.Server(config)
    server_thread = threading.Thread(target=server.run, daemon=True)
    server_thread.start()

    # Console loop in main thread
    try:
        console_loop()
    except (KeyboardInterrupt, EOFError):
        print("\n[EXIT]")
        sys.exit(0)


if __name__ == "__main__":
    main()
