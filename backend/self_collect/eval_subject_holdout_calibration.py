"""
Subject-Held-Out Calibration Validation (the LOSO protocol — Cheung et al. 2024 causality constraint).

Methodology:
  - Load self-collected sessions as SEPARATE rows (no aggregate mean per subject)
  - 5-fold subject-level CV via GroupKFold
  - Trong mỗi fold:
      * Train: ALL sessions của train subjects
      * Test: per-subject calibration:
          - Anchor = earliest session by date (causality)
          - Drift sessions = remaining sessions
          - Compute offset từ anchor, apply lên drift sessions
  - Output: dataframe drift per (subject × session × fold) + visualizations

Reference:
  - Cheung et al. 2024, IEEE TBME 71(12):3569-3592, PMC11799359
    "Subject data must stay in one partition" + "personalization with causality constraint"
  - PracticalBP IMWUT 2025 doi:10.1145/3749486 — 1-reading anchor approach
  - Stergiou et al. 2023, ESH cuffless validation — recalibration test

Run:
  python eval_subject_holdout_calibration.py [--ppgbp / --no-ppgbp]
"""
from __future__ import annotations
import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ML_DIR = PROJECT_ROOT / "ML code" / "ml"
COLLECTED_DIR = PROJECT_ROOT / "collected_data"

os.environ.setdefault("MODEL_DIR", str(ML_DIR / "models"))
sys.path.insert(0, str(ML_DIR))

from ml_models import extract_features  # noqa: E402
from train_ppg_bp import (  # noqa: E402
    _parse_self_collected_metadata,
    load_features_and_labels,
    load_labels,
    NUM_TOTAL_FEATURES,
    SIGNAL_DIR,
)

FS_TARGET = 100  # APPROACH B: skip resample, treat all signals as 100Hz


# ============================================================
# Loader: each session = separate row (NO aggregate mean)
# ============================================================
def load_self_sessions_separate(folder: Path) -> pd.DataFrame:
    """Load tất cả CSV self-collected, mỗi session = 1 row.

    KHÔNG aggregate. Returns DataFrame với cols:
      subject_id, csv_path, session_date, sbp_baseline, dbp_baseline, features (38-array)
    """
    cleaned_dir = folder / "cleaned"
    cleaned_map: Dict[str, Path] = {}
    if cleaned_dir.exists():
        for cf in cleaned_dir.glob("*_cleaned.csv"):
            stem_orig = cf.stem[:-len("_cleaned")]
            cleaned_map[stem_orig] = cf

    raw_files = sorted(folder.glob("*.csv"))
    csv_files: List[Path] = []
    for raw in raw_files:
        if raw.parent == cleaned_dir:
            continue
        replacement = cleaned_map.get(raw.stem)
        csv_files.append(replacement if replacement else raw)

    rows = []
    for csv_path in csv_files:
        try:
            meta = _parse_self_collected_metadata(csv_path)
        except Exception as exc:
            print(f"[WARN] {csv_path.name}: parse metadata failed: {exc}")
            continue

        subject_id = meta.get("subject_id", "?").strip()
        if subject_id == "?":
            continue

        try:
            df = pd.read_csv(
                csv_path, comment="#",
                usecols=["timestamp_ms", "ir", "red"],  # P0 fix: ignore trailing commas (S10_02)
            )
            ir = df["ir"].to_numpy(dtype=np.float64)
            red = df["red"].to_numpy(dtype=np.float64)
            feats_obj = extract_features(ir, red, FS_TARGET)
            feats_array = feats_obj.to_flat_array()  # 38-vec
        except Exception as exc:
            print(f"[WARN] {csv_path.name}: feature extraction failed: {exc}")
            continue

        # Parse session_date from metadata
        session_date_str = meta.get("session_date", "")
        try:
            session_date = datetime.fromisoformat(session_date_str)
        except (ValueError, TypeError):
            session_date = None

        try:
            sbp = float(meta.get("sbp_baseline", "nan"))
            dbp = float(meta.get("dbp_baseline", "nan"))
        except (ValueError, TypeError):
            continue
        if not (np.isfinite(sbp) and np.isfinite(dbp)):
            continue

        rows.append({
            "subject_id": subject_id,
            "csv_path": str(csv_path.relative_to(PROJECT_ROOT)),
            "session_date": session_date,
            "sbp_baseline": sbp,
            "dbp_baseline": dbp,
            "features": feats_array,
        })

    df_sessions = pd.DataFrame(rows)
    if df_sessions.empty:
        return df_sessions

    # Sort by subject_id then session_date for consistent ordering
    df_sessions = df_sessions.sort_values(
        ["subject_id", "session_date"]
    ).reset_index(drop=True)
    return df_sessions


# ============================================================
# Custom CV procedure with within-fold calibration
# ============================================================
def build_pipeline(target: str) -> Pipeline:
    """Build sklearn pipeline cho regression. SVR cho cả SBP và DBP per K. fix retrain."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVR(kernel="rbf", C=10.0, epsilon=1.5, gamma="scale")),
    ])


def subject_holdout_cv(
    X_self: np.ndarray,
    df_self: pd.DataFrame,
    X_ppgbp: np.ndarray,
    y_ppgbp: Dict[str, np.ndarray],
    n_folds: int = 5,
    target: str = "sbp",
) -> pd.DataFrame:
    """
    Custom 5-fold subject-level CV với within-fold per-subject calibration.

    Cho mỗi fold:
      - Train trên: PPG-BP 219 + self-collected sessions của train subjects (KHÔNG aggregate)
      - Test trên: self-collected sessions của test subjects
        * Mỗi test subject có ≥2 sessions: earliest = anchor, rest = drift test
        * Test subjects với 1 session: chỉ compute calibration-free MAE (raw)
    """
    assert target in ("sbp", "dbp")
    cuff_col = f"{target}_baseline"
    y_self = df_self[cuff_col].to_numpy(dtype=np.float64)
    subject_ids = df_self["subject_id"].tolist()

    unique_subjects = sorted(set(subject_ids))
    n_unique = len(unique_subjects)

    if n_unique < n_folds:
        n_folds = max(2, n_unique)
        print(f"[INFO] Reducing n_folds to {n_folds} (only {n_unique} unique subjects)")

    gkf = GroupKFold(n_splits=n_folds)
    results = []

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X_self, y_self, groups=subject_ids)):
        train_subs = sorted(set(np.array(subject_ids)[train_idx]))
        test_subs = sorted(set(np.array(subject_ids)[test_idx]))
        print(f"\n=== Fold {fold_idx+1}/{n_folds} ===")
        print(f"  Train subjects ({len(train_subs)}): {train_subs}")
        print(f"  Test subjects ({len(test_subs)}): {test_subs}")

        # Combine PPG-BP + self-collected train rows
        X_train_combined = np.vstack([X_ppgbp, X_self[train_idx]])
        y_train_combined = np.concatenate([y_ppgbp[target], y_self[train_idx]])

        pipeline = build_pipeline(target)
        pipeline.fit(X_train_combined, y_train_combined)

        # Per-subject calibration in test fold
        for sid in test_subs:
            sub_mask = np.array(subject_ids) == sid
            sub_indices = np.where(sub_mask & np.isin(np.arange(len(subject_ids)), test_idx))[0]
            sub_rows = df_self.iloc[sub_indices].copy().sort_values("session_date").reset_index(drop=True)
            n_sessions = len(sub_rows)

            if n_sessions < 1:
                continue

            # Predict for all sessions
            sub_X = np.vstack(sub_rows["features"].to_list())
            sub_pred = pipeline.predict(sub_X)
            sub_truth = sub_rows[cuff_col].to_numpy()

            # Anchor = earliest session
            anchor_pred = sub_pred[0]
            anchor_truth = sub_truth[0]
            offset = anchor_truth - anchor_pred  # offset = truth - raw_pred
            anchor_date = sub_rows["session_date"].iloc[0]

            for i in range(n_sessions):
                cur_pred = sub_pred[i]
                cur_truth = sub_truth[i]
                cur_date = sub_rows["session_date"].iloc[i]

                pred_calibrated = cur_pred + offset
                err_raw = cur_pred - cur_truth
                err_calibrated = pred_calibrated - cur_truth

                if cur_date is not None and anchor_date is not None:
                    days_from_anchor = (cur_date - anchor_date).total_seconds() / 86400
                else:
                    days_from_anchor = float("nan")

                results.append({
                    "fold": fold_idx + 1,
                    "subject_id": sid,
                    "session_idx": i,
                    "is_anchor": (i == 0),
                    "included_in_drift_eval": (i != 0),  # P1 fix: explicit flag
                    "session_date": cur_date,
                    "days_from_anchor": days_from_anchor,
                    "csv_path": sub_rows["csv_path"].iloc[i],
                    "cuff_truth": cur_truth,
                    "pred_raw": cur_pred,
                    "pred_calibrated": pred_calibrated,
                    "offset": offset,
                    "err_raw": err_raw,
                    "err_calibrated": err_calibrated,
                    "abs_err_raw": abs(err_raw),
                    "abs_err_calibrated": abs(err_calibrated),
                })

    return pd.DataFrame(results)


# ============================================================
# Reporting
# ============================================================
def report_metrics(results_df: pd.DataFrame, target: str = "sbp"):
    """Print summary metrics + drift analysis."""
    if results_df.empty:
        print("[WARN] No results to report")
        return

    print(f"\n{'='*64}")
    print(f"=== TARGET: {target.upper()} ===")
    print(f"{'='*64}")

    n_sessions = len(results_df)
    n_subjects = results_df["subject_id"].nunique()
    n_drift = (~results_df["is_anchor"]).sum()
    print(f"Sessions: {n_sessions} (anchor: {n_sessions - n_drift}, drift: {n_drift})")
    print(f"Subjects: {n_subjects}")

    # Calibration-free metrics (all sessions)
    print(f"\n--- Calibration-FREE (raw pred, all {n_sessions} sessions) ---")
    print(f"  MAE: {results_df['abs_err_raw'].mean():.2f} mmHg")
    print(f"  ME (bias): {results_df['err_raw'].mean():+.2f} mmHg")
    print(f"  SD: {results_df['err_raw'].std(ddof=1):.2f} mmHg")
    print(f"  pred_std: {results_df['pred_raw'].std(ddof=1):.2f} mmHg")

    # Calibrated metrics (drift sessions only — anchor has zero error by construction)
    drift_df = results_df[~results_df["is_anchor"]]
    if len(drift_df) > 0:
        print(f"\n--- After 1-reading calibration ({len(drift_df)} drift sessions) ---")
        print(f"  MAE: {drift_df['abs_err_calibrated'].mean():.2f} mmHg")
        print(f"  ME (bias): {drift_df['err_calibrated'].mean():+.2f} mmHg")
        print(f"  SD: {drift_df['err_calibrated'].std(ddof=1):.2f} mmHg")
        print(f"  Min drift: {drift_df['abs_err_calibrated'].min():.2f}, "
              f"Max drift: {drift_df['abs_err_calibrated'].max():.2f}")

        # Drift vs days_from_anchor
        print(f"\n--- Drift vs days from anchor ---")
        if drift_df["days_from_anchor"].notna().any():
            print(f"  Days range: {drift_df['days_from_anchor'].min():.1f} to "
                  f"{drift_df['days_from_anchor'].max():.1f}")
            # Linear regression
            valid = drift_df.dropna(subset=["days_from_anchor"])
            if len(valid) >= 3:
                from scipy.stats import linregress
                lr = linregress(valid["days_from_anchor"], valid["abs_err_calibrated"])
                print(f"  Linear regression: drift = {lr.intercept:.2f} + {lr.slope:.3f} × days "
                      f"(R={lr.rvalue:.3f}, p={lr.pvalue:.3f})")

    # Per-subject MAE
    print(f"\n--- Per-subject calibrated MAE ---")
    per_sub = drift_df.groupby("subject_id").agg(
        n_drift=("abs_err_calibrated", "size"),
        mae=("abs_err_calibrated", "mean"),
        me=("err_calibrated", "mean"),
    ).reset_index()
    print(per_sub.to_string(index=False))

    # P1 fix: note 1-session subjects excluded from drift table
    all_subjects = set(results_df["subject_id"].unique())
    drift_subjects = set(drift_df["subject_id"].unique()) if len(drift_df) > 0 else set()
    single_session = sorted(all_subjects - drift_subjects)
    if single_session:
        print(f"\n  (1-session subjects excluded from drift table — anchor only, no drift to test):")
        print(f"   {single_session}")

    # AAMI / industry context
    print(f"\n--- Industry context ---")
    print(f"  AAMI/ISO 81060-2 |ME|<=5, SD<=8 (N=85): N/A (this study N={n_subjects} unique)")
    print(f"  PracticalBP IMWUT 2025: SBP ME=-1.49, SD=7.89 (N=43)")
    print(f"  Samsung Galaxy Watch BP recalibration interval: 28 days")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-ppgbp", action="store_true",
                        help="Skip PPG-BP, train chỉ trên self-collected (debug)")
    parser.add_argument("--target", choices=["sbp", "dbp", "both"], default="both")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--out", type=str, default=None,
                        help="Save results CSV (default: _drift_results_{target}.csv)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed cho reproducibility (default: 42)")
    args = parser.parse_args()

    # P1 fix: determinism for thesis reproducibility
    np.random.seed(args.seed)

    print(f"{'='*64}")
    print(f"Subject-Held-Out Calibration Validation (the LOSO protocol)")
    print(f"Random seed: {args.seed}")
    print(f"{'='*64}\n")

    # Load PPG-BP base dataset
    if not args.no_ppgbp:
        print("[1] Loading PPG-BP dataset...")
        labels_df = load_labels()
        X_ppgbp, y_ppgbp, _ = load_features_and_labels(labels_df, aggregate="mean")
        print(f"    PPG-BP: {X_ppgbp.shape[0]} subjects, {X_ppgbp.shape[1]} features")
    else:
        X_ppgbp = np.zeros((0, NUM_TOTAL_FEATURES))
        y_ppgbp = {"sbp": np.array([]), "dbp": np.array([])}
        print("[1] Skipping PPG-BP (--no-ppgbp)")

    # Load self-collected as separate rows
    print("\n[2] Loading self-collected sessions (separate rows)...")
    df_self = load_self_sessions_separate(COLLECTED_DIR)
    if df_self.empty:
        print("[ERROR] No self-collected sessions loaded.")
        return 1

    print(f"    Total sessions: {len(df_self)}")
    print(f"    Unique subjects: {df_self['subject_id'].nunique()}")
    sub_counts = df_self.groupby("subject_id").size().sort_values(ascending=False)
    print(f"    Sessions per subject:")
    for sid, n in sub_counts.items():
        print(f"      {sid}: {n} session(s)")

    X_self = np.vstack(df_self["features"].to_list())

    # Run CV per target
    targets = ["sbp", "dbp"] if args.target == "both" else [args.target]
    for target in targets:
        print(f"\n[3] Running CV for {target.upper()}...")
        results_df = subject_holdout_cv(
            X_self, df_self, X_ppgbp, y_ppgbp,
            n_folds=args.n_folds, target=target,
        )
        report_metrics(results_df, target=target)

        # Save
        out_path = args.out or f"_drift_results_{target}.csv"
        results_df.to_csv(Path(__file__).parent / out_path, index=False)
        print(f"\n  Saved: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
