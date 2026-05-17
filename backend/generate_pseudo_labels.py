"""
Algorithmic pseudo-labeling per Orphanidou et al. 2015 IEEE JBHI 19(3):832-838
PMID 25069129. DOI 10.1109/JBHI.2014.2338351.

Rules per PPG segment label = 'excellent' / 'acceptable' / 'unfit':
  Rule 1: HR mean ∈ [40, 180] BPM
  Rule 2: max(RR_gap) < 3.0 seconds
  Rule 3: max(RR) / min(RR) < 2.2
  Rule 4: template correlation r >= 0.86

  excellent : ALL 4 rules pass
  acceptable: rules 1-3 pass + r ∈ [0.66, 0.86)  ← PROJECT EXTENSION (see note)
  unfit     : ANY rule fail OR r < 0.66

NOTE — "acceptable" tier is a PROJECT EXTENSION, NOT in Orphanidou 2015 paper.
Orphanidou's scheme is binary: accept (r >= 0.86 PPG) vs reject. The 0.66 floor
in this code is Orphanidou's ECG threshold repurposed as a PPG "degraded-but-
usable" tier for ML classifier training (provides middle class examples).
Thesis Section Ch5 SQA must state this design choice explicitly.

Reference threshold 0.86 cho PPG (vs 0.66 ECG). Sensitivity 91% / specificity 95%
(Orphanidou 2015 Table V; project researcher round  verified).

Sliding window: 10s window / 5s stride (50% overlap). Orphanidou recommends
window ≥ 10s for stable HR estimate.

Input datasets (4):
  1. BP-protocol cleaned       collected_data/cleaned/*.csv      fs=100 Hz (the resampling design)
  2. S09 SQA-dedicated         collected_data/sqa_experiment/    fs=100 Hz
  3. BIDMC ICU                  external_data/bidmc/*.csv         fs=125 Hz
  4. PPG-DaLiA wrist BVP        external_data/ppg_dalia/...pkl    fs=64 Hz

Output: backend/pseudo_labels.csv
"""
from __future__ import annotations

import gc
import pickle
import sys
import time
from pathlib import Path

import warnings

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks

try:
    import heartpy as hp  # type: ignore
    HEARTPY_AVAILABLE = True
except Exception:
    HEARTPY_AVAILABLE = False

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from sqa_per_beat import compute_tsqi_per_beat  # noqa: E402

PROJECT_ROOT = ROOT.parent
BP_CLEANED_DIR = PROJECT_ROOT / "collected_data" / "cleaned"
SQA_DIR = PROJECT_ROOT / "collected_data" / "sqa_experiment"
BIDMC_DIR = PROJECT_ROOT / "external_data" / "bidmc"
DALIA_DIR = PROJECT_ROOT / "external_data" / "ppg_dalia" / "PPG_FieldStudy"
OUT_CSV = ROOT / "pseudo_labels.csv"

WINDOW_S = 10
STRIDE_S = 5

# Orphanidou 2015 thresholds
HR_MIN_BPM = 40.0
HR_MAX_BPM = 180.0
RR_GAP_MAX_S = 3.0
RR_RATIO_MAX = 2.2
TEMPLATE_R_EXCELLENT = 0.86
TEMPLATE_R_ACCEPTABLE = 0.66

FS_BP = 100  # the resampling design
FS_BIDMC = 125
FS_DALIA = 64


# ============================================================
# Peak detection — scipy.find_peaks primary (preserves the motion-protocol stress test ground truth)
# HeartPy used as HR validator only (Rule 5 peak-doubling cross-check)
# ============================================================
def compute_hr_heartpy_validator(filtered: np.ndarray, fs: int) -> float | None:
    """
    Validator-only HR estimate via HeartPy 2-stage RR rejection (van Gent 2019).

    Returns float BPM if HeartPy succeeds and HR ∈ [30, 200], else None.

    NOT used as primary peak detector — only as cross-check for Rule 5
    (peak-doubling artifact detector). Rationale: scipy.find_peaks counts
    dicrotic notch as a separate peak → HR doubling on resting signals
    (S01_p1: HR_scipy 141 vs HR_heartpy 75). HeartPy's Stage 1 physiological
    gate [300, 2000] ms rejects sub-300ms RR. Disagreement between scipy
    and HeartPy is the signature of peak-doubling.
    """
    if not HEARTPY_AVAILABLE:
        return None

    use_hp = (1000 % fs == 0)
    hp_kwargs = dict(
        sample_rate=float(fs), windowsize=0.75,
        bpmmin=float(HR_MIN_BPM), bpmmax=float(HR_MAX_BPM),
        high_precision=use_hp,
    )
    if use_hp:
        hp_kwargs["high_precision_fs"] = 1000.0
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, m = hp.process(filtered, **hp_kwargs)
    except Exception:
        return None

    bpm = m.get("bpm")
    if bpm is None:
        return None
    try:
        bpm_f = float(bpm)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(bpm_f):
        return None
    if 30.0 <= bpm_f <= 200.0:
        return bpm_f
    return None


# ============================================================
# Core labeling
# ============================================================
def label_segment(filtered: np.ndarray, peaks: np.ndarray, fs: int) -> tuple[str, dict]:
    """Apply 4 Orphanidou rules + Rule 5 peak-doubling cross-check.

    Returns (label, info_dict). Rule 5 fires when scipy and HeartPy HR
    estimates disagree in the doubling direction — forcing label="unfit"
    with reason="peak_doubling_artifact".
    """
    info: dict = {
        "n_peaks": int(len(peaks)),
        "hr_mean": float("nan"),
        "rr_gap_s": float("nan"),
        "rr_ratio": float("nan"),
        "template_r_median": 0.0,
        "rule1_hr": False,
        "rule2_gap": False,
        "rule3_ratio": False,
        "hr_heartpy": float("nan"),
        "rule5_peak_doubling": False,
    }

    if len(peaks) < 6:
        info["reason"] = "too_few_peaks"
        return "unfit", info

    rr_ms = np.diff(peaks) / float(fs) * 1000.0
    rr_min = float(np.min(rr_ms))
    if rr_min <= 0.0:
        info["reason"] = "nonpositive_rr"
        return "unfit", info

    hr_mean = 60_000.0 / float(np.mean(rr_ms))
    rr_gap_s = float(np.max(rr_ms)) / 1000.0
    rr_ratio = float(np.max(rr_ms)) / rr_min

    info["hr_mean"] = round(hr_mean, 2)
    info["rr_gap_s"] = round(rr_gap_s, 3)
    info["rr_ratio"] = round(rr_ratio, 3)

    # Rule 5 cross-check (peak-doubling detector) — run BEFORE rules 1-4 short-
    # circuit so the flag is captured for diagnostics on every window with peaks.
    # Asymmetric thresholds:
    #   hr_diff > 30 BPM: half of typical adult HR — distinguishes doubling
    #     (HR ~2x) from drift / minor disagreement.
    #   hr_scipy > 100 BPM: implausible resting → only affects resting/normal
    #     protocols. Stress tests at genuinely high true HR are not flagged
    #     because HeartPy will also report high HR (no scipy >> heartpy gap).
    #   scipy - heartpy > 30 (asymmetric): only flag DOUBLING direction,
    #     never HeartPy over-rejection direction.
    hr_heartpy = compute_hr_heartpy_validator(filtered, fs)
    if hr_heartpy is not None:
        info["hr_heartpy"] = round(hr_heartpy, 2)
        if (hr_mean - hr_heartpy) > 30.0 and hr_mean > 100.0:
            info["rule5_peak_doubling"] = True

    rule1 = HR_MIN_BPM <= hr_mean <= HR_MAX_BPM
    rule2 = rr_gap_s < RR_GAP_MAX_S
    rule3 = rr_ratio < RR_RATIO_MAX
    info["rule1_hr"] = rule1
    info["rule2_gap"] = rule2
    info["rule3_ratio"] = rule3

    # Rule 4: template correlation (median per-beat tSQI)
    tsqi_list, _ = compute_tsqi_per_beat(filtered, peaks, fs, 600, 5)
    if not tsqi_list:
        template_r = 0.0
    else:
        template_r = float(np.median(tsqi_list))
    info["template_r_median"] = round(template_r, 4)

    # Rule 5 short-circuit: peak-doubling artifact forces "unfit" regardless of
    # template / RR rules (those would otherwise mark spuriously "excellent").
    if info["rule5_peak_doubling"]:
        info["reason"] = "peak_doubling_artifact"
        return "unfit", info

    rules_pass = rule1 and rule2 and rule3
    if rules_pass and template_r >= TEMPLATE_R_EXCELLENT:
        info["reason"] = "all_pass"
        return "excellent", info
    if rules_pass and template_r >= TEMPLATE_R_ACCEPTABLE:
        info["reason"] = "borderline_template"
        return "acceptable", info
    # Compose unfit reason from failing rules
    fail_reasons = []
    if not rule1:
        fail_reasons.append("hr_oor")
    if not rule2:
        fail_reasons.append("gap_too_long")
    if not rule3:
        fail_reasons.append("rr_ratio_high")
    if rules_pass and template_r < TEMPLATE_R_ACCEPTABLE:
        fail_reasons.append("template_low")
    info["reason"] = "+".join(fail_reasons) if fail_reasons else "unknown"
    return "unfit", info


def label_signal_sliding(
    signal: np.ndarray,
    fs: int,
    window_s: int = WINDOW_S,
    stride_s: int = STRIDE_S,
) -> list[dict]:
    """Apply sliding-window labeling. Returns list of dicts (one per window)."""
    n_win = int(window_s * fs)
    n_stride = int(stride_s * fs)
    if n_win <= 0 or n_stride <= 0 or len(signal) < n_win:
        return []

    # Filter once on the whole signal — bandpass [0.5, 4.0] Hz.
    # Cheaper than per-window; filtfilt handles edge effects.
    b, a = butter(4, [0.5, 4.0], btype="band", fs=fs)
    filtered_full = filtfilt(b, a, signal)

    rows: list[dict] = []
    win_idx = 0
    for start in range(0, len(signal) - n_win + 1, n_stride):
        end = start + n_win
        seg = filtered_full[start:end]
        # Primary peak detector: scipy.find_peaks (preserves the motion-protocol stress test ground
        # truth design — peak-doubling artifacts will be caught by Rule 5
        # cross-check inside label_segment, not by replacing the detector).
        peaks, _ = find_peaks(seg, distance=int(fs * 0.3))
        label, info = label_segment(seg, peaks, fs)
        row = {
            "window_idx": win_idx,
            "t_start_s": round(start / fs, 2),
            "t_end_s": round(end / fs, 2),
            "label": label,
            "n_peaks": info["n_peaks"],
            "hr_mean": info["hr_mean"],
            "rr_gap_s": info["rr_gap_s"],
            "rr_ratio": info["rr_ratio"],
            "template_r_median": info["template_r_median"],
            "rule1_hr": info["rule1_hr"],
            "rule2_gap": info["rule2_gap"],
            "rule3_ratio": info["rule3_ratio"],
            "hr_heartpy": info["hr_heartpy"],
            "rule5_peak_doubling": info["rule5_peak_doubling"],
            "reason": info.get("reason", ""),
        }
        rows.append(row)
        win_idx += 1
    return rows


# ============================================================
# Loaders per dataset
# ============================================================
def _parse_ppg_csv_metadata(path: Path) -> dict:
    """Read leading '# key=value' header lines into a dict."""
    meta: dict = {}
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if not line.startswith("#"):
                    break
                body = line.lstrip("#").strip()
                if "=" not in body:
                    continue
                for tok in body.split(","):
                    tok = tok.strip()
                    if "=" in tok:
                        k, v = tok.split("=", 1)
                        meta[k.strip()] = v.strip()
    except OSError:
        pass
    return meta


def load_bp_or_quoc(path: Path) -> tuple[np.ndarray, int, str]:
    """Load BP-cleaned or S09 SQA CSV. Returns (ir_signal, fs, subject_id)."""
    meta = _parse_ppg_csv_metadata(path)
    df = pd.read_csv(
        path,
        comment="#",
        usecols=lambda c: c in {"timestamp_ms", "ir", "red"},
    )
    if "ir" not in df.columns or df.empty:
        raise ValueError("missing ir column or empty dataframe")
    ir = df["ir"].to_numpy(dtype=float)
    subj = meta.get("subject_id") or path.stem.split("_")[0]
    return ir, FS_BP, subj


def load_bidmc(path: Path) -> tuple[np.ndarray, int, str]:
    """Load BIDMC CSV PLETH column. Returns (signal, fs, subject_id)."""
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    if "PLETH" not in df.columns:
        raise ValueError(f"PLETH column not found in {path.name}")
    sig = df["PLETH"].to_numpy(dtype=float)
    # subject id = bidmc_NN
    subj = path.stem.replace("_Signals", "")
    return sig, FS_BIDMC, subj


def load_dalia(pkl_path: Path) -> tuple[np.ndarray, int, str]:
    """Load PPG-DaLiA pickle, extract BVP only, free rest. Returns (bvp, fs, subj)."""
    with open(pkl_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    bvp = np.asarray(data["signal"]["wrist"]["BVP"], dtype=float).reshape(-1).copy()
    del data
    gc.collect()
    subj = pkl_path.stem  # e.g. "S1"
    return bvp, FS_DALIA, subj


# ============================================================
# Per-dataset processing
# ============================================================
def _process_csv_files(files: list[Path], dataset: str, loader) -> list[dict]:
    rows: list[dict] = []
    for f in files:
        try:
            sig, fs, subj = loader(f)
        except Exception as e:
            print(f"  [skip] {dataset} {f.name}: {type(e).__name__}: {e}")
            continue
        win_rows = label_signal_sliding(sig, fs)
        for r in win_rows:
            r["dataset"] = dataset
            r["file"] = f.name
            r["subject_id"] = subj
            r["fs"] = fs
        rows.extend(win_rows)
        print(f"  {dataset} {f.name}: {len(win_rows)} windows")
    return rows


def process_bp_protocol() -> list[dict]:
    if not BP_CLEANED_DIR.is_dir():
        print(f"[skip] BP cleaned dir not found: {BP_CLEANED_DIR}")
        return []
    files = sorted(BP_CLEANED_DIR.glob("*.csv"))
    print(f"\n[BP-protocol] {len(files)} files")
    return _process_csv_files(files, "BP-protocol", load_bp_or_quoc)


def process_quoc_sqa() -> list[dict]:
    if not SQA_DIR.is_dir():
        print(f"[skip] S09 SQA dir not found: {SQA_DIR}")
        return []
    files = sorted([p for p in SQA_DIR.glob("[Qq]uoc_p?_*.csv")])
    print(f"\n[S09 SQA-dedicated] {len(files)} files")
    return _process_csv_files(files, "S09-SQA", load_bp_or_quoc)


def process_ch_sqa() -> list[dict]:
    if not SQA_DIR.is_dir():
        print(f"[skip] S01 SQA dir not found: {SQA_DIR}")
        return []
    files = sorted([p for p in SQA_DIR.glob("S01_p?_*.csv")])
    print(f"\n[S01 SQA-dedicated] {len(files)} files")
    return _process_csv_files(files, "S01-SQA", load_bp_or_quoc)


def process_duy_sqa() -> list[dict]:
    if not SQA_DIR.is_dir():
        print(f"[skip] Duy SQA dir not found: {SQA_DIR}")
        return []
    files = sorted([p for p in SQA_DIR.glob("[Dd]uy_p?_*.csv")])
    print(f"\n[Duy SQA-dedicated] {len(files)} files")
    return _process_csv_files(files, "Duy-SQA", load_bp_or_quoc)


def process_bidmc() -> list[dict]:
    if not BIDMC_DIR.is_dir():
        print(f"[skip] BIDMC dir not found: {BIDMC_DIR}")
        return []
    files = sorted(BIDMC_DIR.glob("bidmc_*_Signals.csv"))
    print(f"\n[BIDMC ICU] {len(files)} files")
    return _process_csv_files(files, "BIDMC", load_bidmc)


def process_dalia() -> list[dict]:
    if not DALIA_DIR.is_dir():
        print(f"[skip] PPG-DaLiA dir not found: {DALIA_DIR}")
        return []
    pkls = sorted(
        [p / f"{p.name}.pkl" for p in DALIA_DIR.iterdir() if p.is_dir()]
    )
    pkls = [p for p in pkls if p.exists()]
    print(f"\n[PPG-DaLiA] {len(pkls)} subject pickles")
    rows: list[dict] = []
    for pkl in pkls:
        try:
            sig, fs, subj = load_dalia(pkl)
        except Exception as e:
            print(f"  [skip] DaLiA {pkl.name}: {type(e).__name__}: {e}")
            continue
        win_rows = label_signal_sliding(sig, fs)
        for r in win_rows:
            r["dataset"] = "PPG-DaLiA"
            r["file"] = pkl.name
            r["subject_id"] = subj
            r["fs"] = fs
        rows.extend(win_rows)
        print(f"  PPG-DaLiA {pkl.name}: {len(win_rows)} windows")
        # Free signal explicitly after each subject (~150 MB BVP arrays)
        del sig
        gc.collect()
    return rows


# ============================================================
# Reporting
# ============================================================
def print_distribution_table(df: pd.DataFrame) -> None:
    print()
    print("=" * 80)
    print("PSEUDO-LABEL DISTRIBUTION PER DATASET")
    print("=" * 80)
    print(
        f"{'DATASET':<22}{'N_windows':>10}  "
        f"{'excellent%':>12}{'acceptable%':>13}{'unfit%':>10}"
    )
    print("-" * 80)
    overall = {"excellent": 0, "acceptable": 0, "unfit": 0}
    for ds, grp in df.groupby("dataset"):
        n = len(grp)
        n_exc = int((grp["label"] == "excellent").sum())
        n_acc = int((grp["label"] == "acceptable").sum())
        n_unf = int((grp["label"] == "unfit").sum())
        overall["excellent"] += n_exc
        overall["acceptable"] += n_acc
        overall["unfit"] += n_unf
        print(
            f"{ds:<22}{n:>10d}  "
            f"{n_exc/max(n,1)*100:>11.1f}%"
            f"{n_acc/max(n,1)*100:>12.1f}%"
            f"{n_unf/max(n,1)*100:>9.1f}%"
        )
    n_tot = len(df)
    print("-" * 80)
    print(
        f"{'OVERALL':<22}{n_tot:>10d}  "
        f"{overall['excellent']/max(n_tot,1)*100:>11.1f}%"
        f"{overall['acceptable']/max(n_tot,1)*100:>12.1f}%"
        f"{overall['unfit']/max(n_tot,1)*100:>9.1f}%"
    )


def print_quoc_per_protocol(df: pd.DataFrame) -> None:
    quoc = df[df["dataset"] == "S09-SQA"].copy()
    if quoc.empty:
        return
    # Map filenames to protocol id
    def _proto(fname: str) -> str:
        low = fname.lower()
        for p in ("p1", "p2", "p3", "p4"):
            if f"_{p}_" in low:
                return p.upper()
        return "?"
    quoc["protocol"] = quoc["file"].map(_proto)
    print()
    print("=" * 80)
    print("S09 4-protocol monotonic degradation check")
    print("=" * 80)
    print(
        f"{'protocol':<10}{'N':>5}  "
        f"{'excellent%':>12}{'acceptable%':>13}{'unfit%':>10}"
    )
    print("-" * 80)
    for p, grp in sorted(quoc.groupby("protocol")):
        n = len(grp)
        n_exc = int((grp["label"] == "excellent").sum())
        n_acc = int((grp["label"] == "acceptable").sum())
        n_unf = int((grp["label"] == "unfit").sum())
        print(
            f"{p:<10}{n:>5d}  "
            f"{n_exc/max(n,1)*100:>11.1f}%"
            f"{n_acc/max(n,1)*100:>12.1f}%"
            f"{n_unf/max(n,1)*100:>9.1f}%"
        )


def print_dalia_per_subject(df: pd.DataFrame) -> None:
    dalia = df[df["dataset"] == "PPG-DaLiA"].copy()
    if dalia.empty:
        return
    print()
    print("=" * 80)
    print("PPG-DaLiA per-subject distribution")
    print("=" * 80)
    print(
        f"{'subject':<10}{'N':>6}  "
        f"{'excellent%':>12}{'acceptable%':>13}{'unfit%':>10}"
    )
    print("-" * 80)
    for s, grp in sorted(dalia.groupby("subject_id")):
        n = len(grp)
        n_exc = int((grp["label"] == "excellent").sum())
        n_acc = int((grp["label"] == "acceptable").sum())
        n_unf = int((grp["label"] == "unfit").sum())
        print(
            f"{s:<10}{n:>6d}  "
            f"{n_exc/max(n,1)*100:>11.1f}%"
            f"{n_acc/max(n,1)*100:>12.1f}%"
            f"{n_unf/max(n,1)*100:>9.1f}%"
        )


# ============================================================
# Main
# ============================================================
def main() -> int:
    t0 = time.time()
    all_rows: list[dict] = []

    all_rows.extend(process_bp_protocol())
    all_rows.extend(process_quoc_sqa())
    all_rows.extend(process_ch_sqa())
    all_rows.extend(process_duy_sqa())
    all_rows.extend(process_bidmc())
    all_rows.extend(process_dalia())

    if not all_rows:
        print("[error] no rows produced")
        return 1

    cols = [
        "file", "dataset", "subject_id", "fs",
        "window_idx", "t_start_s", "t_end_s", "label",
        "n_peaks", "hr_mean", "rr_gap_s", "rr_ratio", "template_r_median",
        "rule1_hr", "rule2_gap", "rule3_ratio",
        "hr_heartpy", "rule5_peak_doubling", "reason",
    ]
    df = pd.DataFrame(all_rows)
    df = df[[c for c in cols if c in df.columns]]
    df.to_csv(OUT_CSV, index=False)

    elapsed = time.time() - t0
    print(f"\n[ok] wrote {len(df)} rows -> {OUT_CSV}")
    print(f"[time] {elapsed:.1f}s")

    print_distribution_table(df)
    print_quoc_per_protocol(df)
    print_dalia_per_subject(df)
    return 0


if __name__ == "__main__":
    sys.exit(main())
