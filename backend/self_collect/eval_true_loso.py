"""TRUE Leave-One-Subject-Out (LOSO) evaluation — defensible thesis-grade.

Methodology (Cheung et al. 2024 IEEE TBME 71(12):3569-3592, PMC11799359):
  For each unique self-collected person:
    1. Retrain SBP + DBP regressors EXCLUDING that person's data
    2. Predict that person's BP from their averaged feature row
    3. Compute |pred - cuff_truth|
  Aggregate: LOSO MAE, ME, SD across all 13 persons.

Key difference vs eval_naive_baseline.py:
  - eval_naive_baseline.py: bundle trained on ALL 13 → eval is in-sample (optimistic)
  - eval_true_loso.py:      bundle retrained per fold WITHOUT held person → true held-out

Comparison cohort:
  - Held-out subject features are aggregated across multi-session files (same as
    train_ppg_bp.load_self_collected_features) → 1 prediction per person.
  - Cuff truth = first-session BP (per loader convention).

Time: ~3-5 min total (13 folds × 10-15s SVR refit).

Output:
  - Console table: per-subject error + aggregate MAE
  - eval_true_loso_report.csv: per-person details
  - Comparison vs in-sample / naive baselines summary
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

ROOT = Path(__file__).resolve().parents[2]
ML_DIR = ROOT / "ML code" / "ml"
COLLECTED_DIR = ROOT / "collected_data"
sys.path.insert(0, str(ML_DIR))

from train_ppg_bp import (  # noqa: E402
    load_features_and_labels,
    load_labels,
    load_self_collected_features,
)

# Population mean baseline from PPG-BP training set (computed by train_ppg_bp.py).
PPG_BP_MEAN_SBP = 118.3
PPG_BP_MEAN_DBP = 75.0


def build_svr_pipeline() -> Pipeline:
    """SVR with RBF kernel — best CV algo for both SBP & DBP per K. retrain (5/5)."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svr", SVR(kernel="rbf", C=10.0, epsilon=1.5, gamma="scale")),
    ])


def main() -> int:
    print("=" * 78)
    print("TRUE LOSO EVALUATION — defensible thesis-grade BP estimation")
    print("=" * 78)
    print()

    # ----- Load PPG-BP features (cached once) -----
    print("[1/4] Loading PPG-BP labels + features (219 subjects)...")
    labels_df = load_labels()
    X_bp, y_bp, sid_bp = load_features_and_labels(labels_df, aggregate="mean")
    print(f"      X_bp.shape = {X_bp.shape}")

    # ----- Load self-collected features (aggregated per person) -----
    print("\n[2/4] Loading self-collected features...")
    X_sc, y_sc, sid_sc = load_self_collected_features(COLLECTED_DIR)
    n_persons = len(sid_sc)
    print(f"      X_sc.shape = {X_sc.shape}, persons = {sid_sc}")
    print(f"      y_sbp = {y_sc['sbp'].tolist()}")
    print(f"      y_dbp = {y_sc['dbp'].tolist()}")

    if n_persons < 3:
        print("ERROR: need >= 3 persons for LOSO")
        return 1

    # ----- LOSO loop -----
    print(f"\n[3/4] Running {n_persons}-fold LOSO (refit SVR each fold)...")
    rows = []
    for i, held_person in enumerate(sid_sc):
        keep_mask = np.array([j != i for j in range(n_persons)])

        # Combined training set: all PPG-BP + (n-1) other self-collected
        X_train = np.vstack([X_bp, X_sc[keep_mask]])
        y_sbp_train = np.concatenate([y_bp["sbp"], y_sc["sbp"][keep_mask]])
        y_dbp_train = np.concatenate([y_bp["dbp"], y_sc["dbp"][keep_mask]])

        # Fit fresh SVR per fold
        pipe_sbp = build_svr_pipeline().fit(X_train, y_sbp_train)
        pipe_dbp = build_svr_pipeline().fit(X_train, y_dbp_train)

        # Predict held-out person
        X_held = X_sc[i:i + 1]
        sbp_pred = float(pipe_sbp.predict(X_held)[0])
        dbp_pred = float(pipe_dbp.predict(X_held)[0])
        sbp_true = float(y_sc["sbp"][i])
        dbp_true = float(y_sc["dbp"][i])

        rows.append({
            "person": held_person,
            "n_train": len(X_train),
            "sbp_true": sbp_true,
            "sbp_pred": sbp_pred,
            "sbp_err": sbp_pred - sbp_true,
            "dbp_true": dbp_true,
            "dbp_pred": dbp_pred,
            "dbp_err": dbp_pred - dbp_true,
        })
        print(f"      [{i+1:2d}/{n_persons}] {held_person:8s} | "
              f"SBP {sbp_true:5.1f} -> {sbp_pred:6.2f} (err {sbp_pred-sbp_true:+6.2f}) | "
              f"DBP {dbp_true:4.1f} -> {dbp_pred:5.2f} (err {dbp_pred-dbp_true:+5.2f})")

    df = pd.DataFrame(rows)

    # ----- Aggregate metrics -----
    print(f"\n[4/4] Aggregate LOSO metrics")
    print()
    print("=" * 78)
    print("LOSO RESULTS — held-out (defensible)")
    print("=" * 78)

    sbp_mae = df["sbp_err"].abs().mean()
    sbp_me = df["sbp_err"].mean()
    sbp_sd = df["sbp_err"].std(ddof=1)
    dbp_mae = df["dbp_err"].abs().mean()
    dbp_me = df["dbp_err"].mean()
    dbp_sd = df["dbp_err"].std(ddof=1)

    print(f"\nSBP:  MAE = {sbp_mae:.2f} mmHg")
    print(f"      ME  = {sbp_me:+.2f} mmHg  (bias)")
    print(f"      SD  = {sbp_sd:.2f} mmHg  (precision)")
    print(f"\nDBP:  MAE = {dbp_mae:.2f} mmHg")
    print(f"      ME  = {dbp_me:+.2f} mmHg")
    print(f"      SD  = {dbp_sd:.2f} mmHg")

    # ----- Naive baselines for comparison -----
    print()
    print("=" * 78)
    print("BASELINE COMPARISON")
    print("=" * 78)

    # Naive baseline 1: predict population mean for everyone
    sbp_pop_err = df["sbp_true"] - PPG_BP_MEAN_SBP
    dbp_pop_err = df["dbp_true"] - PPG_BP_MEAN_DBP
    sbp_pop_mae = sbp_pop_err.abs().mean()
    dbp_pop_mae = dbp_pop_err.abs().mean()

    # Naive baseline 2: LOO subject mean (no model, just BP labels of others)
    sbp_loo, dbp_loo = [], []
    for i in range(n_persons):
        sbp_loo.append(df["sbp_true"].drop(i).mean())
        dbp_loo.append(df["dbp_true"].drop(i).mean())
    sbp_loo_err = pd.Series(sbp_loo) - df["sbp_true"]
    dbp_loo_err = pd.Series(dbp_loo) - df["dbp_true"]
    sbp_loo_mae = sbp_loo_err.abs().mean()
    dbp_loo_mae = dbp_loo_err.abs().mean()

    print()
    print(f"{'Method':35s} | {'SBP MAE':>9s} | {'DBP MAE':>9s} | {'Verdict':s}")
    print("-" * 78)
    print(f"{'Naive pop mean (118.3 / 75.0)':35s} | "
          f"{sbp_pop_mae:9.2f} | {dbp_pop_mae:9.2f} | dumbest baseline")
    print(f"{'Naive LOO subject mean':35s} | "
          f"{sbp_loo_mae:9.2f} | {dbp_loo_mae:9.2f} | knows others labels")
    print(f"{'TRUE LOSO (held-out, this run)':35s} | "
          f"{sbp_mae:9.2f} | {dbp_mae:9.2f} | model + held-out")

    # ----- Verdict -----
    print()
    print("=" * 78)
    print("VERDICT")
    print("=" * 78)
    sbp_delta_naive = sbp_loo_mae - sbp_mae
    dbp_delta_naive = dbp_loo_mae - dbp_mae

    if sbp_delta_naive > 1.0:
        print(f"  SBP: LOSO ({sbp_mae:.2f}) BEATS naive LOO ({sbp_loo_mae:.2f}) "
              f"by {sbp_delta_naive:.2f} mmHg")
        print(f"       -> Model has signal info beyond population stats. THESIS-DEFENSIBLE.")
    elif sbp_delta_naive > 0:
        print(f"  SBP: LOSO ({sbp_mae:.2f}) marginally better than naive ({sbp_loo_mae:.2f})")
        print(f"       -> Weak signal, frame as 'preliminary, N=13' in thesis.")
    else:
        print(f"  SBP: LOSO ({sbp_mae:.2f}) WORSE than naive ({sbp_loo_mae:.2f})")
        print(f"       -> Model adds noise. Per-subject calibration MUST handle this.")

    print()
    if dbp_delta_naive > 1.0:
        print(f"  DBP: LOSO ({dbp_mae:.2f}) BEATS naive LOO ({dbp_loo_mae:.2f}) "
              f"by {dbp_delta_naive:.2f} mmHg")
        print(f"       -> Model has signal info. THESIS-DEFENSIBLE.")
    elif dbp_delta_naive > 0:
        print(f"  DBP: LOSO ({dbp_mae:.2f}) marginally better than naive ({dbp_loo_mae:.2f})")
    else:
        print(f"  DBP: LOSO ({dbp_mae:.2f}) WORSE than naive ({dbp_loo_mae:.2f})")

    # ----- Save -----
    out = ROOT / "Backend code" / "self_collect" / "eval_true_loso_report.csv"
    df.to_csv(out, index=False)
    print(f"\n[saved] {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
