"""
Wrapper: load cleaned/ files, predict via EnsemblePredictor (uncalibrated),
save predictions CSV with schema required by eval_bp_metrics.py, then run full
AAMI/BHS/Bland-Altman evaluation.

Output:
  Thesis/eval_results/predictions_uncalibrated.csv
  Thesis/eval_results/report_uncalibrated/
    - metrics.json
    - metrics_table.md
    - bland_altman_sbp.png + bland_altman_dbp.png
    - scatter_sbp.png + scatter_dbp.png
    - error_histogram.png
"""
from __future__ import annotations
import os, sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ML_DIR = PROJECT_ROOT / "ML code" / "ml"
CLEANED_DIR = PROJECT_ROOT / "collected_data" / "cleaned"
OUT_DIR = PROJECT_ROOT / "Thesis" / "eval_results"

os.environ.setdefault("MODEL_DIR", str(ML_DIR / "models"))
sys.path.insert(0, str(ML_DIR))
sys.path.insert(0, str(Path(__file__).parent))

from ml_models import EnsemblePredictor, extract_features  # noqa: E402
from train_ppg_bp import _parse_self_collected_metadata  # noqa: E402
from eval_bp_metrics import evaluate  # noqa: E402

FS_TARGET = 100


def main() -> int:
    csv_files = sorted(CLEANED_DIR.glob("*_cleaned.csv"))
    if not csv_files:
        print(f"[ERROR] No cleaned CSV in {CLEANED_DIR}")
        return 1

    print(f"[init] Loading {len(csv_files)} cleaned files from {CLEANED_DIR}")
    ensemble = EnsemblePredictor()
    print(f"[init] Models: {[m.name for m in ensemble.models]}")
    print()

    rows = []
    for csv_path in csv_files:
        meta = _parse_self_collected_metadata(csv_path)
        df = pd.read_csv(csv_path, comment="#")
        ir = df["ir"].to_numpy(dtype=np.float64)
        red = df["red"].to_numpy(dtype=np.float64)

        sbp_true = float(meta.get("sbp_baseline", "nan"))
        dbp_true = float(meta.get("dbp_baseline", "nan"))
        subject_id = meta.get("subject_id", csv_path.stem)

        try:
            feats = extract_features(ir, red, FS_TARGET)
            pred = ensemble.predict(feats)
            sbp_pred = float(pred.sbp)
            dbp_pred = float(pred.dbp)
        except Exception as exc:
            print(f"[warn] {csv_path.name}: predict failed: {exc}")
            sbp_pred = dbp_pred = float("nan")

        if not (np.isfinite(sbp_true) and np.isfinite(dbp_true)
                and np.isfinite(sbp_pred) and np.isfinite(dbp_pred)):
            print(f"[skip] {csv_path.name}: NaN in true or pred")
            continue

        rows.append({
            "subject_id": subject_id,
            "file": csv_path.name,
            "sbp_true": sbp_true,
            "sbp_pred": round(sbp_pred, 2),
            "dbp_true": dbp_true,
            "dbp_pred": round(dbp_pred, 2),
        })
        print(f"  {subject_id:12} sbp {sbp_true:.0f} -> {sbp_pred:.1f} | "
              f"dbp {dbp_true:.0f} -> {dbp_pred:.1f}")

    if len(rows) < 5:
        print(f"[ERROR] Only {len(rows)} valid rows, need >=5")
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pred_csv = OUT_DIR / "predictions_uncalibrated.csv"
    pd.DataFrame(rows).to_csv(pred_csv, index=False, encoding="utf-8")
    print(f"\n[save] Predictions CSV: {pred_csv}")
    print(f"       N = {len(rows)} subjects (no per-subject calibration)")
    print()

    df = pd.read_csv(pred_csv)
    report_dir = OUT_DIR / "report_uncalibrated"
    print(f"[eval] Running AAMI/BHS/Bland-Altman evaluation...")
    print(f"[eval] Output: {report_dir}/")
    print()
    evaluate(df, report_dir, note="Uncalibrated, N=14 cleaned self-collected (post-K. fix retrain N=233)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
