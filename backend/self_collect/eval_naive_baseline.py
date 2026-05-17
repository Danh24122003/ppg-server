"""
Naive baseline comparison cho per-subject calibration.

Chứng minh hoặc bác bỏ "fool's gold" hypothesis:
  Khi model degenerate (predict const 118.3), per-subject calibration
  KHÔNG đóng góp accuracy thực sự — chỉ tương đương memorize cuff value.

3 baselines compare với MAE Calibrated:
  1. Population mean predictor: predict 118.3 (= mean PPG-BP training) cho mọi subject
  2. Per-subject anchor predictor: predict cuff_session1 (anchor) cho session sau
  3. Calibrated model: model_predict + offset

Hypothesis:
  - Nếu MAE_calibrated >> MAE_naive_anchor → model có signal info, calibration cải thiện
  - Nếu MAE_calibrated ≈ MAE_naive_anchor → model degenerate, calibration là "fool's gold"
  - Nếu MAE_naive_population >> MAE_calibrated ≈ MAE_naive_anchor → calibration ăn cắp accuracy từ anchor

Sử dụng:
  - Hiện tại S01Ỉ có session 1 cho 8 subjects → simulate session 2 bằng leave-one-out:
    cho mỗi subject, dùng cuff_baseline làm "ground truth" và compare với:
      a) population mean predictor (118.3)
      b) leave-one-out subject mean (loại subject này, mean còn lại)
      c) model predict (degenerate)
  - Khi có session 2 thực: compute MAE post-calibration vs MAE_naive_anchor

Reference:
  - Sharman et al. 2023 (Hypertension PMC9931644): "calibration-induced pseudo-accuracy"
  - Mukkamala et al. 2023 (PMC9971997): naive baseline comparison required for cuffless validation
"""
from __future__ import annotations
import os, sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ML_DIR = PROJECT_ROOT / "ML code" / "ml"
COLLECTED_DIR = PROJECT_ROOT / "collected_data"

os.environ.setdefault("MODEL_DIR", str(ML_DIR / "models"))
sys.path.insert(0, str(ML_DIR))

from ml_models import EnsemblePredictor, extract_features  # noqa: E402
from train_ppg_bp import _parse_self_collected_metadata, resample_to_target  # noqa: E402

FS_TARGET = 100
PPG_BP_TRAINING_MEAN_SBP = 118.3  # = mean(PPG-BP dataset 219 subjects, training set)
PPG_BP_TRAINING_MEAN_DBP = 75.0


def realized_fs(timestamp_ms: np.ndarray) -> float:
    """fs from MEDIAN sample interval (gap-resistant).

    FIXED 2026-05-03: previous formula `(n-1)/total_dur` bị pollute bởi gaps —
    cho subjects có disconnects (S10, S15_01, S07_01, etc.) tính fs sai
    (80-88Hz thay vì 111Hz thực), pipeline resample sai → features lệch.

    Median of diff(timestamps) ignore outlier gaps (chỉ ~0.02% của intervals).
    Verified by code reviewer round 2 + alignment với eval_quality_tiers.py,
    clean_gap_csv.py, eval_cleaned_vs_raw.py (đều đã dùng median).
    """
    if len(timestamp_ms) < 2:
        return float("nan")
    diffs = np.diff(timestamp_ms)
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return float("nan")
    median_diff = float(np.median(diffs))
    if median_diff <= 0:
        return float("nan")
    return 1000.0 / median_diff


def main() -> int:
    csv_files = sorted(COLLECTED_DIR.glob("*.csv"))
    if not csv_files:
        print(f"Khong co CSV nao trong {COLLECTED_DIR}")
        return 1

    print("=" * 78)
    print("NAIVE BASELINE COMPARISON — fool's gold check cho calibration")
    print("=" * 78)
    print()

    ensemble = EnsemblePredictor()
    print(f"[init] Models: {[m.name for m in ensemble.models]}")
    print()

    rows = []
    for csv_path in csv_files:
        meta = _parse_self_collected_metadata(csv_path)
        df = pd.read_csv(csv_path, comment="#")
        ir = df["ir"].to_numpy(dtype=np.float64)
        red = df["red"].to_numpy(dtype=np.float64)
        ts = df["timestamp_ms"].to_numpy(dtype=np.float64)

        sid = meta.get("subject_id", csv_path.stem)
        fs_meta = int(meta.get("fs", FS_TARGET))
        fs_real = realized_fs(ts)

        # Use fs từ timestamps (fix bug S10)
        fs_use = int(round(fs_real)) if 25 <= fs_real <= 400 else fs_meta

        ir_proc = ir.copy()
        red_proc = red.copy()
        # APPROACH B FIX (3/5/2026): Skip resample, treat as fs=100Hz.
        # Bias constant 11% cho fs=111 → calibration absorb. Verified eval_fs_fix_compare.py.

        try:
            feats = extract_features(ir_proc, red_proc, FS_TARGET)
            pred = ensemble.predict(feats)
            sbp_pred = pred.sbp
            dbp_pred = pred.dbp
        except Exception as exc:
            print(f"[warn] {csv_path.name}: {exc}")
            continue

        rows.append({
            "subject": sid,
            "sbp_real": float(meta.get("sbp_baseline", "nan")),
            "dbp_real": float(meta.get("dbp_baseline", "nan")),
            "sbp_pred_model": sbp_pred,
            "dbp_pred_model": dbp_pred,
        })

    df = pd.DataFrame(rows)
    n = len(df)
    print(f"Loaded {n} subjects: {df['subject'].tolist()}")
    print()

    # ==================== 3 BASELINES ====================

    # Baseline 1: Population mean (predict 118.3 cho mọi subject — = degenerate model output)
    df["sbp_pred_pop_mean"] = PPG_BP_TRAINING_MEAN_SBP
    df["dbp_pred_pop_mean"] = PPG_BP_TRAINING_MEAN_DBP

    # Baseline 2: Leave-one-out subject mean
    # Cho mỗi subject, predict bằng mean BP của các subjects KHÁC
    sbp_loo_means = []
    dbp_loo_means = []
    for i in range(n):
        others_sbp = df["sbp_real"].drop(i).mean()
        others_dbp = df["dbp_real"].drop(i).mean()
        sbp_loo_means.append(others_sbp)
        dbp_loo_means.append(others_dbp)
    df["sbp_pred_loo_mean"] = sbp_loo_means
    df["dbp_pred_loo_mean"] = dbp_loo_means

    # Baseline 3: Per-subject calibrated (= model + offset)
    # Với model degenerate, sbp_calibrated = sbp_pred_model + (sbp_real - sbp_pred_model) = sbp_real
    # → MAE_calibrated cho session 1 = 0 (trivial, vì calibrate trên chính reading đó)
    # → KHÔNG fair comparison — phải dùng session 2 hoặc LOO
    # → Implement Per-Subject-LOO-Calibrated:
    #   Treat current subject's BP như "session 2", offset = mean(others' real - others' pred)
    sbp_calib_loo = []
    dbp_calib_loo = []
    for i in range(n):
        # Average offset từ N-1 subjects khác
        others_offset_sbp = (df["sbp_real"].drop(i) - df["sbp_pred_model"].drop(i)).mean()
        others_offset_dbp = (df["dbp_real"].drop(i) - df["dbp_pred_model"].drop(i)).mean()
        sbp_calib_loo.append(df.loc[i, "sbp_pred_model"] + others_offset_sbp)
        dbp_calib_loo.append(df.loc[i, "dbp_pred_model"] + others_offset_dbp)
    df["sbp_pred_calib_loo"] = sbp_calib_loo
    df["dbp_pred_calib_loo"] = dbp_calib_loo

    # ==================== Compute MAE for each ====================
    methods = [
        ("model_degenerate", "sbp_pred_model", "dbp_pred_model",
         "Model raw (predict const 118.3)"),
        ("naive_pop_mean", "sbp_pred_pop_mean", "dbp_pred_pop_mean",
         "Naive baseline 1: predict population mean (118.3/75)"),
        ("naive_loo_mean", "sbp_pred_loo_mean", "dbp_pred_loo_mean",
         "Naive baseline 2: predict LOO subject mean"),
        ("calibrated_loo", "sbp_pred_calib_loo", "dbp_pred_calib_loo",
         "Per-subject calibrated (LOO offset average)"),
    ]

    print("=" * 78)
    print("RESULTS — Mean Absolute Error per method")
    print("=" * 78)
    print(f"{'Method':30s} | {'SBP MAE':>9s} | {'SBP ME':>8s} | {'DBP MAE':>9s} | {'DBP ME':>8s}")
    print("-" * 78)

    metrics_table = []
    for name, sbp_col, dbp_col, desc in methods:
        sbp_err = df[sbp_col] - df["sbp_real"]
        dbp_err = df[dbp_col] - df["dbp_real"]
        sbp_mae = sbp_err.abs().mean()
        dbp_mae = dbp_err.abs().mean()
        sbp_me = sbp_err.mean()
        dbp_me = dbp_err.mean()
        print(f"{name:30s} | {sbp_mae:>9.2f} | {sbp_me:>+8.2f} | {dbp_mae:>9.2f} | {dbp_me:>+8.2f}")
        metrics_table.append({
            "method": name,
            "description": desc,
            "sbp_mae": sbp_mae,
            "sbp_me": sbp_me,
            "dbp_mae": dbp_mae,
            "dbp_me": dbp_me,
        })
    print()

    # ==================== Interpretation ====================
    print("=" * 78)
    print("INTERPRETATION — Has model added value beyond naive baseline?")
    print("=" * 78)
    mae_model = metrics_table[0]["sbp_mae"]
    mae_pop_mean = metrics_table[1]["sbp_mae"]
    mae_loo_mean = metrics_table[2]["sbp_mae"]
    mae_calib = metrics_table[3]["sbp_mae"]

    print()
    print(f"SBP MAE comparison:")
    print(f"  Model output         = {mae_model:.2f}  <- what we have")
    print(f"  Naive (pop mean)     = {mae_pop_mean:.2f}  <- simplest baseline")
    print(f"  Naive (LOO mean)     = {mae_loo_mean:.2f}  <- per-subject baseline (no PPG)")
    print(f"  Calibrated (LOO)     = {mae_calib:.2f}  <- what calibration gets")
    print()

    # Test 1: Model vs population mean
    delta_1 = mae_pop_mean - mae_model
    print(f"Test 1: |Model - Population mean| = {abs(delta_1):.3f} mmHg")
    if abs(delta_1) < 0.1:
        print("  → Model EQUALS population mean predictor (degenerate confirmed)")
    elif delta_1 > 1:
        print("  → Model BETTER than naive — has some signal info")
    else:
        print("  → Model marginally different; low signal info")

    # Test 2: Calibration vs LOO mean baseline
    delta_2 = mae_loo_mean - mae_calib
    print(f"\nTest 2: |Calibrated - LOO mean| = {abs(delta_2):.3f} mmHg")
    if abs(delta_2) < 1:
        print("  → Calibration KHÔNG cải thiện đáng kể vs naive LOO mean")
        print("    (Calibration chỉ là 'pop mean + offset' khi model degenerate)")
    elif delta_2 > 2:
        print("  → Calibration thực sự cải thiện — model có signal info")
    else:
        print("  → Calibration cải thiện nhẹ; signal info limited")

    print()

    # ==================== Output for thesis ====================
    print("=" * 78)
    print("THESIS-READY CONCLUSIONS")
    print("=" * 78)
    if abs(delta_1) < 0.5 and abs(delta_2) < 1:
        print("""
⚠️  FOOL'S GOLD CONFIRMED:
    Model PPG hiện tại KHÔNG có signal info đáng kể.
    Calibration accuracy đến từ subject anchor, KHÔNG phải PPG signal.

Honest framing for thesis (Chapter 4 / 5):
    "MAE post-calibration = X mmHg, comparable to naive per-subject baseline.
     Per-subject offset functions as anchor correction, not PPG-based prediction.
     This matches Sharman et al. 2023 'calibration-induced pseudo-accuracy' finding.
     Future work: train on reflectance dataset with broader BP coverage to enable
     true PPG-driven prediction beyond per-subject baseline."
""")
    else:
        print(f"""
✓  Model có signal info modest:
    Calibration cải thiện {delta_2:.2f} mmHg vs naive baseline.
    Đáng để document trong thesis.
""")

    # Save report
    out = Path(__file__).parent / "eval_naive_baseline_report.csv"
    pd.DataFrame(metrics_table).to_csv(out, index=False)
    print(f"[save] Metrics table: {out}")

    out_per_subject = Path(__file__).parent / "eval_naive_baseline_per_subject.csv"
    df.to_csv(out_per_subject, index=False)
    print(f"[save] Per-subject predictions: {out_per_subject}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
