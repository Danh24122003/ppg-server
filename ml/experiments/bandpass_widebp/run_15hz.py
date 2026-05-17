"""Bandpass cutoff ablation experiment for BP LOSO — 0.5-15 Hz variant.

Variant of run_bandpass_experiment.py for reviewer C1 hypothesis: 0.5-15 Hz
bandpass preserves morphological BP features (faster derivatives / 2nd derivative
peaks at higher frequencies). Compares:
  - Baseline: highcut=4.0 Hz (production current, ~30-240 BPM band)
  - Variant : highcut=15.0 Hz (wide, near kamal1989 upper limit)

Reuses existing `per_subject_baseline_4hz.csv` + `results_baseline_4hz.json` in
the parent experiments dir (4Hz features unchanged) to avoid re-running baseline.

Isolation: monkeypatch `ml_models._safe_bandpass` only during 15Hz feature
extraction. Production files NOT modified.

Output dir: ./bandpass_15hz_2026-05-14/
  - results_baseline_4hz.json       (copied from parent)
  - results_widebp_15hz.json
  - per_subject_baseline_4hz.csv    (copied from parent)
  - per_subject_widebp_15hz.csv
  - bootstrap_delta.json

Usage:
  cd "ml/ml"
  python experiments/bandpass_widebp/run_15hz.py

Time: ~5-10 min (only 15Hz feature extraction + 15 LOSO folds for variant).
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# Resolve paths relative to this file. HERE = ml/experiments/bandpass_widebp/
HERE = Path(__file__).resolve().parent
ML_DIR = HERE.parent.parent          # ml/
ROOT = ML_DIR.parent.parent          # repo root
COLLECTED_DIR = ROOT / "collected_data"

sys.path.insert(0, str(ML_DIR))

# Import ml_models FIRST so we can monkeypatch before train_ppg_bp imports it.
import ml_models  # noqa: E402

_ORIGINAL_SAFE_BANDPASS = ml_models._safe_bandpass


def _make_bandpass_with_highcut(highcut: float):
    """Return a wrapper that calls the original _safe_bandpass with a fixed highcut."""
    def _wrapped(signal: np.ndarray, fs: int,
                 lowcut: float = 0.5,
                 highcut: float = highcut,
                 order: int = 4) -> np.ndarray:
        return _ORIGINAL_SAFE_BANDPASS(signal, fs, lowcut=lowcut,
                                        highcut=highcut, order=order)
    _wrapped.__name__ = f"_safe_bandpass_highcut_{highcut}"
    return _wrapped


def patch_bandpass(highcut: float) -> None:
    ml_models._safe_bandpass = _make_bandpass_with_highcut(highcut)


def restore_bandpass() -> None:
    ml_models._safe_bandpass = _ORIGINAL_SAFE_BANDPASS


from train_ppg_bp import (  # noqa: E402
    load_features_and_labels,
    load_labels,
    load_self_collected_features,
)


def build_svr_pipeline() -> Pipeline:
    """SVR with RBF kernel — matches eval_true_loso.py canonical setup (epsilon=1.5)."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svr", SVR(kernel="rbf", C=10.0, epsilon=1.5, gamma="scale")),
    ])


def extract_features_for_highcut(highcut: float) -> Dict[str, object]:
    print(f"\n[extract] highcut={highcut} Hz — patching _safe_bandpass...")
    patch_bandpass(highcut)
    try:
        t0 = time.time()
        print(f"[extract] Loading PPG-BP labels + 657 signal files (fs 1000->100 Hz)...")
        labels_df = load_labels()
        X_bp, y_bp, sid_bp = load_features_and_labels(labels_df, aggregate="mean")
        print(f"[extract]   X_bp.shape = {X_bp.shape}")

        print(f"[extract] Loading self-collected reflectance...")
        X_sc, y_sc, sid_sc = load_self_collected_features(COLLECTED_DIR)
        print(f"[extract]   X_sc.shape = {X_sc.shape}, persons = {sid_sc}")
        dt = time.time() - t0
        print(f"[extract] Done in {dt:.1f}s")
    finally:
        restore_bandpass()

    return {
        "X_bp": X_bp,
        "y_bp_sbp": y_bp["sbp"],
        "y_bp_dbp": y_bp["dbp"],
        "sid_bp": sid_bp,
        "X_sc": X_sc,
        "y_sc_sbp": y_sc["sbp"],
        "y_sc_dbp": y_sc["dbp"],
        "sid_sc": sid_sc,
    }


def run_loso(data: Dict[str, object], label: str) -> Tuple[pd.DataFrame, Dict[str, float]]:
    X_bp = data["X_bp"]
    y_bp_sbp = data["y_bp_sbp"]
    y_bp_dbp = data["y_bp_dbp"]
    X_sc = data["X_sc"]
    y_sc_sbp = data["y_sc_sbp"]
    y_sc_dbp = data["y_sc_dbp"]
    sid_sc = data["sid_sc"]
    n_persons = len(sid_sc)

    print(f"\n[loso:{label}] Running {n_persons}-fold LOSO (refit SVR each fold)...")
    rows = []
    t0 = time.time()
    for i, held_person in enumerate(sid_sc):
        keep_mask = np.array([j != i for j in range(n_persons)])

        X_train = np.vstack([X_bp, X_sc[keep_mask]])
        y_sbp_train = np.concatenate([y_bp_sbp, y_sc_sbp[keep_mask]])
        y_dbp_train = np.concatenate([y_bp_dbp, y_sc_dbp[keep_mask]])

        pipe_sbp = build_svr_pipeline().fit(X_train, y_sbp_train)
        pipe_dbp = build_svr_pipeline().fit(X_train, y_dbp_train)

        X_held = X_sc[i:i + 1]
        sbp_pred = float(pipe_sbp.predict(X_held)[0])
        dbp_pred = float(pipe_dbp.predict(X_held)[0])
        sbp_true = float(y_sc_sbp[i])
        dbp_true = float(y_sc_dbp[i])

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
        print(f"  [{i+1:2d}/{n_persons}] {held_person:10s} | "
              f"SBP {sbp_true:5.1f} -> {sbp_pred:6.2f} (err {sbp_pred-sbp_true:+6.2f}) | "
              f"DBP {dbp_true:4.1f} -> {dbp_pred:5.2f} (err {dbp_pred-dbp_true:+5.2f})")
    dt = time.time() - t0
    print(f"[loso:{label}] Done in {dt:.1f}s")

    df = pd.DataFrame(rows)
    metrics = {
        "n": int(len(df)),
        "sbp_mae": float(df["sbp_err"].abs().mean()),
        "sbp_me": float(df["sbp_err"].mean()),
        "sbp_sd": float(df["sbp_err"].std(ddof=1)),
        "sbp_rmse": float(np.sqrt(np.mean(df["sbp_err"] ** 2))),
        "dbp_mae": float(df["dbp_err"].abs().mean()),
        "dbp_me": float(df["dbp_err"].mean()),
        "dbp_sd": float(df["dbp_err"].std(ddof=1)),
        "dbp_rmse": float(np.sqrt(np.mean(df["dbp_err"] ** 2))),
    }
    return df, metrics


def paired_bootstrap_delta(df_base: pd.DataFrame, df_var: pd.DataFrame,
                            col: str, n_boot: int = 10000, seed: int = 42) -> Dict[str, float]:
    """Paired bootstrap on |err| differences (variant - baseline).

    Negative delta MAE means variant is BETTER (lower MAE).
    """
    assert df_base["person"].equals(df_var["person"]), "person order must match"
    rng = np.random.default_rng(seed)
    abs_base = df_base[col].abs().to_numpy()
    abs_var = df_var[col].abs().to_numpy()
    paired_diff = abs_var - abs_base
    n = len(paired_diff)

    boot_deltas = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_deltas[b] = paired_diff[idx].mean()

    mean_delta = float(paired_diff.mean())
    ci_lo, ci_hi = np.percentile(boot_deltas, [2.5, 97.5])
    p_two_sided = 2.0 * min((boot_deltas >= 0).mean(), (boot_deltas <= 0).mean())
    p_two_sided = float(min(p_two_sided, 1.0))

    return {
        "n": int(n),
        "mean_delta": mean_delta,
        "ci95_lo": float(ci_lo),
        "ci95_hi": float(ci_hi),
        "p_two_sided": p_two_sided,
        "n_boot": int(n_boot),
        "interpretation": (
            "negative delta = variant improves over baseline (lower abs error)"
        ),
    }


def main() -> int:
    print("=" * 78)
    print("Bandpass cutoff ablation: 4.0 Hz (baseline) vs 15.0 Hz (variant) — LOSO BP")
    print("=" * 78)

    out_dir = HERE / "bandpass_15hz_2026-05-14"
    out_dir.mkdir(parents=True, exist_ok=True)

    parent_baseline_csv = HERE / "per_subject_baseline_4hz.csv"
    parent_baseline_json = HERE / "results_baseline_4hz.json"

    # ----- Baseline 4Hz: reuse if available, else run -----
    if parent_baseline_csv.exists() and parent_baseline_json.exists():
        print("\n[baseline] Reusing existing per_subject_baseline_4hz.csv (4Hz unchanged)")
        df_base = pd.read_csv(parent_baseline_csv)
        shutil.copy2(parent_baseline_csv, out_dir / "per_subject_baseline_4hz.csv")
        shutil.copy2(parent_baseline_json, out_dir / "results_baseline_4hz.json")
        with open(parent_baseline_json, "r", encoding="utf-8") as f:
            base_payload = json.load(f)
        metrics_base = base_payload["loso_metrics"]
        print(f"[baseline] SBP MAE = {metrics_base['sbp_mae']:.3f} | DBP MAE = {metrics_base['dbp_mae']:.3f}")
    else:
        data_base = extract_features_for_highcut(4.0)
        df_base, metrics_base = run_loso(data_base, "baseline_4hz")
        df_base.to_csv(out_dir / "per_subject_baseline_4hz.csv", index=False)
        with open(out_dir / "results_baseline_4hz.json", "w", encoding="utf-8") as f:
            json.dump({
                "config_label": "baseline_4hz",
                "highcut_hz": 4.0,
                "lowcut_hz": 0.5,
                "filter_order": 4,
                "loso_metrics": metrics_base,
            }, f, indent=2)

    # ----- Variant 15Hz -----
    data_var = extract_features_for_highcut(15.0)
    df_var, metrics_var = run_loso(data_var, "widebp_15hz")
    csv_path = out_dir / "per_subject_widebp_15hz.csv"
    df_var.to_csv(csv_path, index=False)
    print(f"[save] {csv_path}")

    json_path = out_dir / "results_widebp_15hz.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "config_label": "widebp_15hz",
            "highcut_hz": 15.0,
            "lowcut_hz": 0.5,
            "filter_order": 4,
            "loso_metrics": metrics_var,
        }, f, indent=2)
    print(f"[save] {json_path}")

    # ----- Bootstrap delta -----
    print("\n" + "=" * 78)
    print("Paired bootstrap on |err| difference (widebp_15hz - baseline_4hz)")
    print("=" * 78)
    sbp_boot = paired_bootstrap_delta(df_base, df_var, "sbp_err")
    dbp_boot = paired_bootstrap_delta(df_base, df_var, "dbp_err")

    print(f"\nSBP delta MAE (widebp_15hz - baseline_4hz):")
    print(f"  mean = {sbp_boot['mean_delta']:+.3f} mmHg")
    print(f"  95% CI = [{sbp_boot['ci95_lo']:+.3f}, {sbp_boot['ci95_hi']:+.3f}]")
    print(f"  p two-sided = {sbp_boot['p_two_sided']:.4f}")

    print(f"\nDBP delta MAE (widebp_15hz - baseline_4hz):")
    print(f"  mean = {dbp_boot['mean_delta']:+.3f} mmHg")
    print(f"  95% CI = [{dbp_boot['ci95_lo']:+.3f}, {dbp_boot['ci95_hi']:+.3f}]")
    print(f"  p two-sided = {dbp_boot['p_two_sided']:.4f}")

    with open(out_dir / "bootstrap_delta.json", "w", encoding="utf-8") as f:
        json.dump({
            "sbp": sbp_boot,
            "dbp": dbp_boot,
            "convention": "delta = widebp_15hz - baseline_4hz; negative = improvement",
        }, f, indent=2)
    print(f"[save] {out_dir / 'bootstrap_delta.json'}")

    print("\n" + "=" * 78)
    print("HEADLINE")
    print("=" * 78)
    print(f"  Baseline (4 Hz):   SBP MAE = {metrics_base['sbp_mae']:.3f} | DBP MAE = {metrics_base['dbp_mae']:.3f}")
    print(f"  Variant  (15 Hz):  SBP MAE = {metrics_var['sbp_mae']:.3f} | DBP MAE = {metrics_var['dbp_mae']:.3f}")
    print(f"  Delta (var - base): SBP {metrics_var['sbp_mae']-metrics_base['sbp_mae']:+.3f} | "
          f"DBP {metrics_var['dbp_mae']-metrics_base['dbp_mae']:+.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
