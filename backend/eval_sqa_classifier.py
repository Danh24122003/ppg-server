"""Train multinomial Logistic Regression on 7 SQI features → 3-class pseudo-label.

 of Deep SQA plan. Compare LR vs current 3-criterion heuristic baseline
(`assess_signal_quality()` from main.py).

Pipeline:
  1. Load `pseudo_labels.csv` (11,850 windows × 4 datasets, 43 unique subjects).
  2. Recompute 7 SQI features per window from raw signal:
       6 batch features (compute_sqi_features from main.py):
         spectral_purity, hr_bpm, amplitude_ratio, ssqi, ksqi, entropy_amp_dist
       1 per-beat aggregate (compute_tsqi_per_beat from sqa_per_beat.py):
         tsqi_median
  3. Cache features → parquet (skip recompute on re-run).
  4. Train multinomial LR + StandardScaler in Pipeline.
  5. 5-fold subject-level GroupKFold (group by subject_id).
  6. cross_val_predict(method='predict_proba') → out-of-fold probabilities.
  7. Metrics: macro/weighted ROC AUC (one-vs-rest), per-class P/R/F1, 3x3 CM.
  8. 3-criterion heuristic baseline: assess_signal_quality() string -> mapping.
  9. Refit LR on all data → joblib.dump.
 10. Save figures: ROC curves, feature importance, confusion matrices.

Usage:
    python "backend/eval_sqa_classifier.py"
    python "backend/eval_sqa_classifier.py" --no-cache       # force recompute features
    python "backend/eval_sqa_classifier.py" --max-files 5    # debug subset
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import pickle
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks

ROOT = Path(__file__).resolve().parent  # backend/
sys.path.insert(0, str(ROOT))
from main import compute_sqi_features, assess_signal_quality  # noqa: E402
from sqa_per_beat import compute_tsqi_per_beat  # noqa: E402


PROJECT_ROOT = ROOT.parent
PSEUDO_LABELS_CSV = ROOT / "pseudo_labels.csv"
# Use pickle for cache (pyarrow / fastparquet not installed in this env).
# Same intent: fast binary cache preserving dtypes for re-runs.
CACHE_PARQUET = ROOT / "sqa_features_cache.pkl"
CACHE_META_JSON = ROOT / "sqa_features_cache_meta.json"
MAIN_PY_PATH = ROOT / "main.py"
SQA_PER_BEAT_PY_PATH = ROOT / "sqa_per_beat.py"
RESULTS_CSV = ROOT / "eval_sqa_classifier_results.csv"
MODEL_PKL = ROOT / "sqa_lr_model.pkl"
FIGURES_DIR = ROOT / "figures"

BP_CLEANED_DIR = PROJECT_ROOT / "collected_data" / "cleaned"
SQA_DIR = PROJECT_ROOT / "collected_data" / "sqa_experiment"
BIDMC_DIR = PROJECT_ROOT / "external_data" / "bidmc"
DALIA_DIR = PROJECT_ROOT / "external_data" / "ppg_dalia" / "PPG_FieldStudy"

FEATURE_COLS = [
    "spectral_purity",
    "hr_bpm",
    "amplitude_ratio",
    "ssqi",
    "ksqi",
    "entropy_amp_dist",
    "tsqi_median",
]
LABEL_ORDER = ["excellent", "acceptable", "unfit"]


# ============================================================
# Raw signal loaders (one per dataset, mirrors generate_pseudo_labels.py)
# ============================================================
def _load_bp_or_quoc(path: Path) -> np.ndarray:
    df = pd.read_csv(
        path,
        comment="#",
        usecols=lambda c: c in {"timestamp_ms", "ir", "red"},
    )
    return df["ir"].to_numpy(dtype=float)


def _load_bidmc(path: Path) -> np.ndarray:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df["PLETH"].to_numpy(dtype=float)


def _load_dalia(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    bvp = np.asarray(data["signal"]["wrist"]["BVP"], dtype=float).reshape(-1).copy()
    del data
    gc.collect()
    return bvp


def resolve_signal_path(dataset: str, file_name: str) -> Path | None:
    """Map (dataset, file_name) -> absolute path on disk."""
    if dataset == "BP-protocol":
        return BP_CLEANED_DIR / file_name
    if dataset in {"S09-SQA", "S01-SQA", "Duy-SQA"}:
        return SQA_DIR / file_name
    if dataset == "BIDMC":
        return BIDMC_DIR / file_name
    if dataset == "PPG-DaLiA":
        # file_name = "S1.pkl" -> S1/S1.pkl
        subj = file_name.replace(".pkl", "")
        return DALIA_DIR / subj / file_name
    return None


def load_full_signal(dataset: str, file_name: str) -> np.ndarray:
    path = resolve_signal_path(dataset, file_name)
    if path is None or not path.exists():
        raise FileNotFoundError(f"{dataset}/{file_name} -> {path}")
    if dataset in {"BP-protocol", "S09-SQA", "S01-SQA", "Duy-SQA"}:
        return _load_bp_or_quoc(path)
    if dataset == "BIDMC":
        return _load_bidmc(path)
    if dataset == "PPG-DaLiA":
        return _load_dalia(path)
    raise ValueError(f"unknown dataset: {dataset}")


# ============================================================
# Feature extraction per window
# ============================================================
def compute_window_features(raw_window: np.ndarray, fs: int) -> dict:
    """Return dict with 7 features + heuristic_label."""
    out = {k: 0.0 for k in FEATURE_COLS}
    out["heuristic_label"] = "poor"

    if raw_window.size < 16:
        return out

    b, a = butter(4, [0.5, 4.0], btype="band", fs=fs)
    filtered = filtfilt(b, a, raw_window)
    peaks, _ = find_peaks(filtered, distance=int(fs * 0.3))

    feats = compute_sqi_features(raw_window, filtered, list(peaks), fs)
    for k in ("spectral_purity", "hr_bpm", "amplitude_ratio",
              "ssqi", "ksqi", "entropy_amp_dist"):
        out[k] = float(feats.get(k, 0.0))

    tsqi_list, _ = compute_tsqi_per_beat(filtered, peaks, fs, 600, 5)
    if tsqi_list:
        out["tsqi_median"] = float(np.median(tsqi_list))

    out["heuristic_label"] = assess_signal_quality(
        raw_window, filtered, list(peaks), fs
    )
    return out


# ============================================================
# Cache metadata (Fix #1 — hash-based invalidation)
# ============================================================
def _md5_first_64kb(path: Path) -> str:
    """Return md5 hex of first 64KB of file (cheap fingerprint)."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read(64 * 1024))
    return h.hexdigest()


def _compute_cache_meta(df_labels: pd.DataFrame) -> dict:
    return {
        "n_rows": int(len(df_labels)),
        "labels_csv_md5": _md5_first_64kb(PSEUDO_LABELS_CSV),
        "main_py_mtime": os.path.getmtime(MAIN_PY_PATH),
        "sqa_per_beat_py_mtime": os.path.getmtime(SQA_PER_BEAT_PY_PATH),
    }


def _check_cache_meta(current: dict) -> tuple[bool, str]:
    """Return (valid, reason). reason is empty on hit."""
    if not CACHE_META_JSON.exists():
        return False, "metadata sidecar missing"
    try:
        with open(CACHE_META_JSON, "r", encoding="utf-8") as f:
            saved = json.load(f)
    except Exception as e:
        return False, f"metadata read failed: {type(e).__name__}: {e}"
    for key in ("n_rows", "labels_csv_md5", "main_py_mtime", "sqa_per_beat_py_mtime"):
        if key not in saved:
            return False, f"metadata key missing: {key}"
        if saved[key] != current[key]:
            return False, f"{key} changed (cache={saved[key]} vs current={current[key]})"
    return True, ""


# ============================================================
# Build feature matrix from pseudo_labels.csv
# ============================================================
def build_feature_matrix(
    df_labels: pd.DataFrame,
    use_cache: bool = True,
    max_files: int | None = None,
) -> pd.DataFrame:
    """Return DataFrame with all label columns + 7 features + heuristic_label.

    Caching strategy: pickle + JSON sidecar (Fix #1) keyed on
    (n_rows, labels_csv_md5, main_py_mtime, sqa_per_beat_py_mtime).
    """
    current_meta = _compute_cache_meta(df_labels)
    if use_cache and CACHE_PARQUET.exists():
        valid, reason = _check_cache_meta(current_meta)
        if valid:
            cached = pd.read_pickle(CACHE_PARQUET)
            # Defensive: row count must still align (already checked via meta)
            if len(cached) == len(df_labels):
                print(f"[cache] hit: {len(cached)} rows from {CACHE_PARQUET}")
                return cached
            print(f"[cache] stale (cached {len(cached)} vs labels {len(df_labels)}); rebuild")
        else:
            print(f"[cache] invalidated: {reason}")

    print(f"[features] computing per-window features for {len(df_labels)} windows ...")

    file_keys = df_labels[["dataset", "file"]].drop_duplicates().reset_index(drop=True)
    if max_files is not None:
        file_keys = file_keys.head(max_files)
        keep_files = set(zip(file_keys["dataset"], file_keys["file"]))
        df_labels = df_labels[
            df_labels.apply(lambda r: (r["dataset"], r["file"]) in keep_files, axis=1)
        ].reset_index(drop=True)
        print(f"[debug] max-files={max_files}: limit to {len(df_labels)} windows")

    # Pre-allocate output columns
    all_rows: list[dict] = []
    t0 = time.time()
    n_files = len(file_keys)

    for fi, (_, key) in enumerate(file_keys.iterrows(), start=1):
        dataset = key["dataset"]
        file_name = key["file"]
        try:
            sig_full = load_full_signal(dataset, file_name)
        except Exception as e:
            print(f"  [skip] {dataset}/{file_name}: {type(e).__name__}: {e}")
            continue

        sub = df_labels[
            (df_labels["dataset"] == dataset) & (df_labels["file"] == file_name)
        ]
        elapsed = time.time() - t0
        print(
            f"  [{fi:3d}/{n_files}] {dataset:<12} {file_name:<40} "
            f"len={len(sig_full):>7} windows={len(sub):>4} elapsed={elapsed:6.1f}s"
        )

        for _, row in sub.iterrows():
            fs = int(row["fs"])
            t_start = float(row["t_start_s"])
            t_end = float(row["t_end_s"])
            i0 = int(t_start * fs)
            i1 = int(t_end * fs)
            i1 = min(i1, len(sig_full))
            if i1 <= i0:
                continue
            window = sig_full[i0:i1]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                feats = compute_window_features(window, fs)
            rec = {
                "dataset": dataset,
                "file": file_name,
                "subject_id": row["subject_id"],
                "window_idx": int(row["window_idx"]),
                "t_start_s": t_start,
                "t_end_s": t_end,
                "fs": fs,
                "label": row["label"],
            }
            rec.update(feats)
            all_rows.append(rec)

        del sig_full
        gc.collect()

    feat_df = pd.DataFrame(all_rows)
    feat_df.to_pickle(CACHE_PARQUET)
    # Skip metadata write when --max-files truncates the dataset
    # (current_meta describes the FULL labels CSV, not the subset).
    if max_files is None:
        with open(CACHE_META_JSON, "w", encoding="utf-8") as f:
            json.dump(current_meta, f, indent=2)
        print(f"[cache] wrote {len(feat_df)} rows -> {CACHE_PARQUET} (+ meta sidecar)")
    else:
        print(f"[cache] wrote {len(feat_df)} rows -> {CACHE_PARQUET} (no meta: --max-files)")
    return feat_df


# ============================================================
# LR training + evaluation
# ============================================================
def map_heuristic_to_3class(s: str) -> str:
    """Map assess_signal_quality() string to 3-class pseudo-label space."""
    if s == "good":
        return "excellent"
    if s == "fair":
        return "acceptable"
    return "unfit"  # "poor" or anything else


def train_and_evaluate(feat_df: pd.DataFrame) -> dict:
    """Fit LR via cross_val_predict + heuristic baseline.

    Returns dict with:
        y_true, y_pred_lr, proba_lr (n × 3, columns ordered LABEL_ORDER),
        y_pred_heuristic, fold_coefs (list of (n_classes × n_features)).
    """
    # Lazy-import sklearn so file imports cheap if user runs --no-cache check etc.
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import GroupKFold, cross_val_predict

    X = feat_df[FEATURE_COLS].to_numpy(dtype=float)
    y = feat_df["label"].astype(str).to_numpy()
    groups = feat_df["subject_id"].astype(str).to_numpy()

    n_subjects = len(np.unique(groups))
    n_splits = min(5, n_subjects)
    print(f"[cv] GroupKFold n_splits={n_splits} (n_subjects={n_subjects})")

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            solver="lbfgs",  # multinomial by default in sklearn >= 1.5
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        )),
    ])

    cv = GroupKFold(n_splits=n_splits)

    print("[cv] computing out-of-fold predict_proba ...")
    t0 = time.time()
    proba_oof = cross_val_predict(
        pipe, X, y, cv=cv, groups=groups,
        method="predict_proba", n_jobs=-1,
    )
    print(f"[cv] done in {time.time()-t0:.1f}s")

    # Fit each fold separately to extract per-feature coefficients
    print("[cv] refitting per-fold to extract coefficients ...")
    fold_coefs: list[np.ndarray] = []
    fold_classes: list[np.ndarray] = []
    for tr_idx, _te_idx in cv.split(X, y, groups):
        p = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(
                solver="lbfgs",  # multinomial by default in sklearn >= 1.5
                max_iter=2000,
                class_weight="balanced",
                random_state=42,
            )),
        ])
        p.fit(X[tr_idx], y[tr_idx])
        fold_coefs.append(p.named_steps["lr"].coef_.copy())
        fold_classes.append(p.named_steps["lr"].classes_.copy())

    # cross_val_predict returns proba columns in alphabetical class order.
    # Map columns to LABEL_ORDER (excellent, acceptable, unfit). Classes
    # absent from y get a zero column (defensive for --max-files debug runs;
    # full dataset has all 3 classes).
    cv_classes_ = np.array(sorted(np.unique(y)))
    proba_ordered = np.zeros((proba_oof.shape[0], len(LABEL_ORDER)))
    for i, lab in enumerate(LABEL_ORDER):
        hits = np.where(cv_classes_ == lab)[0]
        if hits.size:
            proba_ordered[:, i] = proba_oof[:, int(hits[0])]

    # Predicted class = argmax over LABEL_ORDER columns
    y_pred_lr = np.array(LABEL_ORDER)[proba_ordered.argmax(axis=1)]

    # Heuristic baseline
    y_pred_heuristic = feat_df["heuristic_label"].map(map_heuristic_to_3class).to_numpy()

    return {
        "y_true": y,
        "y_pred_lr": y_pred_lr,
        "proba_lr": proba_ordered,
        "y_pred_heuristic": y_pred_heuristic,
        "fold_coefs": fold_coefs,
        "fold_classes": fold_classes,
    }


# ============================================================
# Fix #2 — LODO (leave-dataset-out) evaluation
# ============================================================
def evaluate_lodo(feat_df: pd.DataFrame) -> dict:
    """Train on N-1 datasets, predict on held-out dataset, aggregate.

    Returns dict with:
        per_dataset_auc: {dataset: macro AUC or None}
        per_dataset_f1:  {dataset: macro F1}
        per_dataset_n:   {dataset: n_windows}
        agg_macro_auc:   macro AUC across all aggregated predictions
        agg_macro_f1:    macro F1 across all aggregated predictions
        lodo_pred_label: array length len(feat_df), aligned to feat_df index
        lodo_proba:      (n × 3) array, columns ordered LABEL_ORDER
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import roc_auc_score, f1_score

    datasets = sorted(feat_df["dataset"].astype(str).unique())
    print()
    print("=" * 80)
    print(f"LODO EVALUATION (leave-dataset-out, {len(datasets)} datasets)")
    print("=" * 80)

    X = feat_df[FEATURE_COLS].to_numpy(dtype=float)
    y = feat_df["label"].astype(str).to_numpy()
    ds = feat_df["dataset"].astype(str).to_numpy()

    n = len(feat_df)
    lodo_proba = np.zeros((n, len(LABEL_ORDER)))
    lodo_pred_label = np.array(["unfit"] * n, dtype=object)
    pred_assigned = np.zeros(n, dtype=bool)

    per_dataset_auc: dict = {}
    per_dataset_f1: dict = {}
    per_dataset_n: dict = {}

    for held in datasets:
        train_mask = ds != held
        test_mask = ds == held
        n_test = int(test_mask.sum())
        per_dataset_n[held] = n_test
        if n_test == 0:
            print(f"  [{held:<12}] empty test set — skip")
            per_dataset_auc[held] = None
            per_dataset_f1[held] = None
            continue
        if train_mask.sum() == 0:
            print(f"  [{held:<12}] empty train set — skip")
            per_dataset_auc[held] = None
            per_dataset_f1[held] = None
            continue

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(
                solver="lbfgs",
                max_iter=2000,
                class_weight="balanced",
                random_state=42,
            )),
        ])
        pipe.fit(X[train_mask], y[train_mask])
        proba_test = pipe.predict_proba(X[test_mask])
        cls = pipe.named_steps["lr"].classes_

        # Map predict_proba columns to LABEL_ORDER
        proba_ordered = np.zeros((n_test, len(LABEL_ORDER)))
        for i, lab in enumerate(LABEL_ORDER):
            hits = np.where(cls == lab)[0]
            if hits.size:
                proba_ordered[:, i] = proba_test[:, int(hits[0])]
        pred_test = np.array(LABEL_ORDER)[proba_ordered.argmax(axis=1)]

        idx_test = np.where(test_mask)[0]
        lodo_proba[idx_test] = proba_ordered
        lodo_pred_label[idx_test] = pred_test
        pred_assigned[idx_test] = True

        # Per-dataset metrics (defensive: skip class with no positives)
        y_test = y[test_mask]
        y_bin = np.column_stack([(y_test == lab).astype(int) for lab in LABEL_ORDER])
        valid_cols = [i for i in range(len(LABEL_ORDER)) if y_bin[:, i].sum() > 0]
        if len(valid_cols) >= 2:
            try:
                auc = roc_auc_score(
                    y_bin[:, valid_cols], proba_ordered[:, valid_cols],
                    multi_class="ovr", average="macro",
                )
            except ValueError as e:
                print(f"  [{held:<12}] AUC undefined: {e}")
                auc = None
        else:
            print(f"  [{held:<12}] only {len(valid_cols)} class(es) present — AUC undefined")
            auc = None
        f1m = f1_score(y_test, pred_test, labels=LABEL_ORDER, average="macro", zero_division=0)
        per_dataset_auc[held] = auc
        per_dataset_f1[held] = float(f1m)
        auc_str = f"{auc:.4f}" if auc is not None else "  N/A "
        print(
            f"  [{held:<12}] n_test={n_test:>5}  "
            f"macro_AUC={auc_str}  macro_F1={f1m:.4f}"
        )

    # Aggregate over rows where prediction was assigned
    if pred_assigned.any():
        y_all = y[pred_assigned]
        proba_all = lodo_proba[pred_assigned]
        pred_all = lodo_pred_label[pred_assigned]
        y_bin_all = np.column_stack([(y_all == lab).astype(int) for lab in LABEL_ORDER])
        valid_cols = [i for i in range(len(LABEL_ORDER)) if y_bin_all[:, i].sum() > 0]
        try:
            agg_auc = float(roc_auc_score(
                y_bin_all[:, valid_cols], proba_all[:, valid_cols],
                multi_class="ovr", average="macro",
            ))
        except ValueError:
            agg_auc = float("nan")
        agg_f1 = float(f1_score(y_all, pred_all, labels=LABEL_ORDER, average="macro", zero_division=0))
    else:
        agg_auc = float("nan")
        agg_f1 = float("nan")

    print(f"  AGGREGATE     macro_AUC={agg_auc:.4f}  macro_F1={agg_f1:.4f}")

    return {
        "per_dataset_auc": per_dataset_auc,
        "per_dataset_f1": per_dataset_f1,
        "per_dataset_n": per_dataset_n,
        "agg_macro_auc": agg_auc,
        "agg_macro_f1": agg_f1,
        "lodo_pred_label": lodo_pred_label,
        "lodo_proba": lodo_proba,
        "pred_assigned": pred_assigned,
    }


# ============================================================
# Fix #3 — Unweighted LR sanity check
# ============================================================
def evaluate_unweighted_lr(feat_df: pd.DataFrame) -> dict:
    """Same Pipeline as main path but class_weight=None. Console-only output."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import GroupKFold, cross_val_predict
    from sklearn.metrics import roc_auc_score

    X = feat_df[FEATURE_COLS].to_numpy(dtype=float)
    y = feat_df["label"].astype(str).to_numpy()
    groups = feat_df["subject_id"].astype(str).to_numpy()

    n_subjects = len(np.unique(groups))
    n_splits = min(5, n_subjects)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            solver="lbfgs",
            max_iter=2000,
            class_weight=None,
            random_state=42,
        )),
    ])
    cv = GroupKFold(n_splits=n_splits)
    proba_oof = cross_val_predict(
        pipe, X, y, cv=cv, groups=groups,
        method="predict_proba", n_jobs=-1,
    )
    cv_classes_ = np.array(sorted(np.unique(y)))
    proba_ordered = np.zeros((proba_oof.shape[0], len(LABEL_ORDER)))
    for i, lab in enumerate(LABEL_ORDER):
        hits = np.where(cv_classes_ == lab)[0]
        if hits.size:
            proba_ordered[:, i] = proba_oof[:, int(hits[0])]

    y_bin = np.column_stack([(y == lab).astype(int) for lab in LABEL_ORDER])
    valid_cols = [i for i in range(len(LABEL_ORDER)) if y_bin[:, i].sum() > 0]
    auc_macro = float(roc_auc_score(
        y_bin[:, valid_cols], proba_ordered[:, valid_cols],
        multi_class="ovr", average="macro",
    ))
    auc_weighted = float(roc_auc_score(
        y_bin[:, valid_cols], proba_ordered[:, valid_cols],
        multi_class="ovr", average="weighted",
    ))
    return {"auc_macro": auc_macro, "auc_weighted": auc_weighted}


# ============================================================
# Metric reporting
# ============================================================
def report_metrics(
    res: dict,
    feat_df: pd.DataFrame | None = None,
    unweighted_metrics: dict | None = None,
    lodo_metrics: dict | None = None,
) -> dict:
    from sklearn.metrics import (
        roc_auc_score,
        classification_report,
        confusion_matrix,
        accuracy_score,
        f1_score,
    )

    y_true = res["y_true"]
    y_pred_lr = res["y_pred_lr"]
    proba_lr = res["proba_lr"]
    y_pred_h = res["y_pred_heuristic"]

    # Binarize y_true for ROC AUC (column order = LABEL_ORDER)
    y_bin = np.column_stack([
        (y_true == lab).astype(int) for lab in LABEL_ORDER
    ])
    # Drop classes with zero positive (defensive — shouldn't happen here)
    valid_cols = [i for i in range(len(LABEL_ORDER)) if y_bin[:, i].sum() > 0]
    if len(valid_cols) < len(LABEL_ORDER):
        print(f"[warn] missing classes in y_true: {set(LABEL_ORDER)-set(LABEL_ORDER[i] for i in valid_cols)}")
    auc_macro = roc_auc_score(
        y_bin[:, valid_cols], proba_lr[:, valid_cols],
        multi_class="ovr", average="macro",
    )
    auc_weighted = roc_auc_score(
        y_bin[:, valid_cols], proba_lr[:, valid_cols],
        multi_class="ovr", average="weighted",
    )

    # Heuristic AUC requires probabilities; we approximate via one-hot
    # (degenerate "probability" — same as accuracy-style ROC). Report only
    # accuracy + F1 + confusion matrix for heuristic.
    h_acc = accuracy_score(y_true, y_pred_h)
    h_f1_macro = f1_score(y_true, y_pred_h, labels=LABEL_ORDER, average="macro", zero_division=0)
    h_f1_weighted = f1_score(y_true, y_pred_h, labels=LABEL_ORDER, average="weighted", zero_division=0)

    lr_acc = accuracy_score(y_true, y_pred_lr)
    lr_f1_macro = f1_score(y_true, y_pred_lr, labels=LABEL_ORDER, average="macro", zero_division=0)
    lr_f1_weighted = f1_score(y_true, y_pred_lr, labels=LABEL_ORDER, average="weighted", zero_division=0)

    print()
    print("=" * 80)
    print("LR CLASSIFIER (multinomial, 5-fold GroupKFold, class_weight='balanced')")
    print("=" * 80)
    print(f"Macro ROC AUC (one-vs-rest)    : {auc_macro:.4f}")
    print(f"Weighted ROC AUC (one-vs-rest) : {auc_weighted:.4f}")
    print(f"Accuracy                       : {lr_acc:.4f}")
    print(f"Macro F1                       : {lr_f1_macro:.4f}")
    print(f"Weighted F1                    : {lr_f1_weighted:.4f}")
    print()
    print("Per-class report (LR):")
    print(classification_report(
        y_true, y_pred_lr, labels=LABEL_ORDER, zero_division=0
    ))
    cm_lr = confusion_matrix(y_true, y_pred_lr, labels=LABEL_ORDER)
    print(f"Confusion matrix (rows=true, cols=pred, order={LABEL_ORDER}):")
    print(cm_lr)

    print()
    print("=" * 80)
    print("3-CRITERION HEURISTIC BASELINE (assess_signal_quality)")
    print("=" * 80)
    print(f"Accuracy                       : {h_acc:.4f}")
    print(f"Macro F1                       : {h_f1_macro:.4f}")
    print(f"Weighted F1                    : {h_f1_weighted:.4f}")
    print()
    print("Per-class report (heuristic):")
    print(classification_report(
        y_true, y_pred_h, labels=LABEL_ORDER, zero_division=0
    ))
    cm_h = confusion_matrix(y_true, y_pred_h, labels=LABEL_ORDER)
    print(f"Confusion matrix (rows=true, cols=pred, order={LABEL_ORDER}):")
    print(cm_h)

    # Per-feature importance (mean |coef| across folds, per class).
    # Align to LABEL_ORDER row layout; classes missing in a fold get zero rows.
    # Binary folds (sklearn returns shape (1, n_features)) handled defensively.
    fold_coefs = res["fold_coefs"]
    fold_classes = res["fold_classes"]
    n_features = len(FEATURE_COLS)
    aligned = np.zeros((len(fold_coefs), len(LABEL_ORDER), n_features))
    for fi, (coef, cls_arr) in enumerate(zip(fold_coefs, fold_classes)):
        for li, lab in enumerate(LABEL_ORDER):
            hits = np.where(cls_arr == lab)[0]
            if hits.size == 0:
                continue
            row = int(hits[0])
            if row < coef.shape[0]:
                aligned[fi, li] = coef[row]
            elif coef.shape[0] == 1 and len(cls_arr) == 2:
                # Binary case: sklearn stores ONE coef vector (for the
                # "positive" class = classes_[1]). The other class's coef
                # is the negation.
                aligned[fi, li] = coef[0] if lab == cls_arr[1] else -coef[0]
    coefs_arr = aligned
    abs_mean = np.abs(coefs_arr).mean(axis=0)  # (n_classes, n_features)
    abs_std = np.abs(coefs_arr).std(axis=0)

    print()
    print("=" * 80)
    print("PER-FEATURE IMPORTANCE (mean |coefficient| across folds)")
    print("=" * 80)
    print(f"{'feature':<22}" + "".join(f"{lab:>14}" for lab in LABEL_ORDER))
    for fi, fname in enumerate(FEATURE_COLS):
        line = f"{fname:<22}"
        for ci in range(len(LABEL_ORDER)):
            line += f"{abs_mean[ci, fi]:>8.3f}±{abs_std[ci, fi]:.2f}"
        print(line)

    # Rank by sum of |coef| across all classes
    importance_sum = abs_mean.sum(axis=0)
    rank_idx = np.argsort(-importance_sum)
    print()
    print("Ranking (sum |coef| across 3 classes):")
    for rk, idx in enumerate(rank_idx, start=1):
        print(f"  {rk}. {FEATURE_COLS[idx]:<22} {importance_sum[idx]:.3f}")

    # Fix #3 — Unweighted LR sanity check (console only)
    if unweighted_metrics is not None:
        print()
        print("=" * 80)
        print("UNWEIGHTED LR SANITY CHECK (class_weight=None, same Pipeline + CV)")
        print("=" * 80)
        print(f"Unweighted LR macro AUC    : {unweighted_metrics['auc_macro']:.4f}")
        print(f"Unweighted LR weighted AUC : {unweighted_metrics['auc_weighted']:.4f}")
        print(
            f"Delta vs balanced training : "
            f"macro {auc_macro - unweighted_metrics['auc_macro']:+.4f}  "
            f"weighted {auc_weighted - unweighted_metrics['auc_weighted']:+.4f}"
        )

    # Fix #4 — Defense disclaimer block
    # Compute dataset imbalance % for disclaimer note 2
    dalia_pct = float("nan")
    if feat_df is not None and "dataset" in feat_df.columns:
        n_total = len(feat_df)
        if n_total > 0:
            dalia_pct = 100.0 * (feat_df["dataset"] == "PPG-DaLiA").sum() / n_total

    # Compute "acceptable" rarity for disclaimer note 3
    acc_pct = float("nan")
    if len(y_true) > 0:
        acc_pct = 100.0 * (y_true == "acceptable").sum() / len(y_true)

    if unweighted_metrics is not None:
        unw_macro_str = f"{unweighted_metrics['auc_macro']:.4f}"
    else:
        unw_macro_str = "N/A"

    if lodo_metrics is not None:
        agg_auc = lodo_metrics.get("agg_macro_auc", float("nan"))
        agg_auc_str = f"{agg_auc:.4f}"
    else:
        agg_auc_str = "N/A"

    print()
    print("=" * 80)
    print("THESIS DEFENSE NOTES (auto-generated for honest framing):")
    print()
    print("1. HEURISTIC BASELINE: assess_signal_quality() is a real-time binary")
    print("   pass/reject gate. It predicts \"good\" for ~99.9% of windows because")
    print("   spectral_purity + amplitude_check easily score >= 4 on real PPG.")
    print(f"   The gap LR macro-F1 {lr_f1_macro:.2f} vs heuristic {h_f1_macro:.2f} demonstrates that")
    print("   gate-based SQA is fundamentally not 3-class capable — motivating")
    print("   the dedicated classifier approach.")
    print()
    print(f"2. DATASET IMBALANCE: PPG-DaLiA dominates with ~{dalia_pct:.1f}% of windows from")
    print("   5 subjects. Standard GroupKFold may not reflect cross-dataset")
    print(f"   generalization. LODO aggregate macro AUC = {agg_auc_str} (vs {auc_macro:.4f}")
    print("   under GroupKFold) quantifies this gap.")
    print()
    print(f"3. CLASS WEIGHTING: class_weight='balanced' upweights rare class")
    print(f"   \"acceptable\" ({acc_pct:.1f}%) ~{(100.0/max(acc_pct,1e-9))/3:.1f}x. Macro AUC {auc_macro:.4f} is computed on these")
    print(f"   balanced-trained probabilities. Unweighted sanity macro AUC = {unw_macro_str}")
    print(f"   demonstrates the magnitude of this effect.")
    print()
    print("4. PEAK DETECTION: this script uses scipy.find_peaks for feature")
    print("   extraction; production pipeline uses HeartPy adaptive threshold.")
    print("   Cross-validate hr_bpm feature consistency before deployment.")
    print("=" * 80)

    return {
        "auc_macro": auc_macro,
        "auc_weighted": auc_weighted,
        "lr_acc": lr_acc,
        "lr_f1_macro": lr_f1_macro,
        "lr_f1_weighted": lr_f1_weighted,
        "h_acc": h_acc,
        "h_f1_macro": h_f1_macro,
        "h_f1_weighted": h_f1_weighted,
        "cm_lr": cm_lr,
        "cm_heuristic": cm_h,
        "abs_mean_coef": abs_mean,
        "abs_std_coef": abs_std,
        "importance_sum": importance_sum,
        "rank_idx": rank_idx,
    }


# ============================================================
# Figures
# ============================================================
def make_figures(res: dict, metrics: dict, lodo_metrics: dict | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc as sk_auc

    FIGURES_DIR.mkdir(exist_ok=True)

    y_true = res["y_true"]
    proba_lr = res["proba_lr"]
    cm_lr = metrics["cm_lr"]
    cm_h = metrics["cm_heuristic"]

    # --- ROC curves ---
    fig, ax = plt.subplots(figsize=(6, 5.5))
    colors = {"excellent": "#2ca02c", "acceptable": "#ff7f0e", "unfit": "#d62728"}
    for i, lab in enumerate(LABEL_ORDER):
        y_bin = (y_true == lab).astype(int)
        if y_bin.sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin, proba_lr[:, i])
        a = sk_auc(fpr, tpr)
        ax.plot(fpr, tpr, color=colors[lab], lw=2, label=f"{lab} (AUC={a:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(
        f"LR one-vs-rest ROC curves\nMacro AUC={metrics['auc_macro']:.3f}  "
        f"Weighted AUC={metrics['auc_weighted']:.3f}"
    )
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_lr_roc_curves.pdf")
    plt.close(fig)
    print(f"[fig] wrote {FIGURES_DIR/'fig_lr_roc_curves.pdf'}")

    # --- Feature importance ---
    abs_mean = metrics["abs_mean_coef"]  # (n_classes, n_features)
    fig, ax = plt.subplots(figsize=(8, 5))
    n_feat = len(FEATURE_COLS)
    x = np.arange(n_feat)
    width = 0.27
    for i, lab in enumerate(LABEL_ORDER):
        ax.bar(x + (i - 1) * width, abs_mean[i], width,
               label=lab, color=colors[lab])
    ax.set_xticks(x)
    ax.set_xticklabels(FEATURE_COLS, rotation=30, ha="right")
    ax.set_ylabel("Mean |coefficient| across folds")
    ax.set_title("LR feature importance per class (StandardScaled inputs)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_lr_feature_importance.pdf")
    plt.close(fig)
    print(f"[fig] wrote {FIGURES_DIR/'fig_lr_feature_importance.pdf'}")

    # --- Confusion matrices (LR vs heuristic, row-normalized) ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, cm, title in [(axes[0], cm_lr, "LR"), (axes[1], cm_h, "Heuristic")]:
        with np.errstate(invalid="ignore"):
            cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm_norm = np.nan_to_num(cm_norm)
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels(LABEL_ORDER, rotation=30, ha="right")
        ax.set_yticklabels(LABEL_ORDER)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(title)
        for i in range(3):
            for j in range(3):
                txt = f"{cm[i,j]}\n({cm_norm[i,j]*100:.1f}%)"
                ax.text(j, i, txt, ha="center", va="center",
                        color="white" if cm_norm[i, j] > 0.5 else "black",
                        fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Confusion matrix (row-normalized): LR vs 3-criterion heuristic")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_lr_confusion_matrix.pdf")
    plt.close(fig)
    print(f"[fig] wrote {FIGURES_DIR/'fig_lr_confusion_matrix.pdf'}")

    # --- Fix #2: LODO bar chart ---
    if lodo_metrics is not None:
        per_ds_auc = lodo_metrics.get("per_dataset_auc", {})
        per_ds_n = lodo_metrics.get("per_dataset_n", {})
        agg_auc = lodo_metrics.get("agg_macro_auc", float("nan"))
        if per_ds_auc:
            datasets = sorted(per_ds_auc.keys())
            heights = [
                per_ds_auc[d] if per_ds_auc[d] is not None else 0.0
                for d in datasets
            ]
            colors_ds = ["#1f77b4" if per_ds_auc[d] is not None else "#bbbbbb"
                         for d in datasets]
            fig, ax = plt.subplots(figsize=(7.5, 4.5))
            xpos = np.arange(len(datasets))
            bars = ax.bar(xpos, heights, color=colors_ds, edgecolor="black")
            ax.axhline(agg_auc, color="#d62728", linestyle="--", lw=1.5,
                       label=f"Aggregate macro AUC = {agg_auc:.3f}")
            for bi, (bar, d) in enumerate(zip(bars, datasets)):
                auc = per_ds_auc[d]
                n = per_ds_n.get(d, 0)
                if auc is None:
                    label = f"N/A\n(n={n})"
                else:
                    label = f"{auc:.3f}\n(n={n})"
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.015,
                        label, ha="center", va="bottom", fontsize=9)
            ax.set_xticks(xpos)
            ax.set_xticklabels(datasets, rotation=20, ha="right")
            ax.set_ylim(0, 1.05)
            ax.set_ylabel("Macro ROC AUC (one-vs-rest)")
            ax.set_title(
                "LODO evaluation: train on 3 datasets, test on held-out dataset"
            )
            ax.legend(loc="lower right")
            ax.grid(True, axis="y", alpha=0.3)
            fig.tight_layout()
            fig.savefig(FIGURES_DIR / "fig_lr_lodo_auc.pdf")
            plt.close(fig)
            print(f"[fig] wrote {FIGURES_DIR/'fig_lr_lodo_auc.pdf'}")


# ============================================================
# Main
# ============================================================
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-cache", action="store_true",
                    help="Force feature recompute (ignore cache parquet)")
    ap.add_argument("--max-files", type=int, default=None,
                    help="Limit number of source files (debug)")
    args = ap.parse_args()

    if not PSEUDO_LABELS_CSV.exists():
        print(f"[error] {PSEUDO_LABELS_CSV} not found. Run generate_pseudo_labels.py first.")
        return 1

    df_labels = pd.read_csv(PSEUDO_LABELS_CSV)
    print(f"[load] pseudo_labels.csv: {len(df_labels)} windows, "
          f"{df_labels['subject_id'].nunique()} subjects, "
          f"{df_labels['dataset'].nunique()} datasets")
    print(f"       label distribution: {df_labels['label'].value_counts().to_dict()}")

    feat_df = build_feature_matrix(
        df_labels,
        use_cache=not args.no_cache,
        max_files=args.max_files,
    )
    if feat_df.empty:
        print("[error] no features computed")
        return 2

    # Drop rows with any non-finite feature (defensive — should be rare)
    n_pre = len(feat_df)
    mask = np.isfinite(feat_df[FEATURE_COLS].to_numpy()).all(axis=1)
    feat_df = feat_df[mask].reset_index(drop=True)
    if len(feat_df) < n_pre:
        print(f"[clean] dropped {n_pre - len(feat_df)} rows with non-finite features")

    res = train_and_evaluate(feat_df)

    # Fix #3 — Unweighted LR sanity check
    print()
    print("[unweighted] computing class_weight=None CV for sanity check ...")
    t0 = time.time()
    unweighted_metrics = evaluate_unweighted_lr(feat_df)
    print(f"[unweighted] done in {time.time()-t0:.1f}s")

    # Fix #2 — LODO evaluation
    t0 = time.time()
    lodo_metrics = evaluate_lodo(feat_df)
    print(f"[lodo] done in {time.time()-t0:.1f}s")

    metrics = report_metrics(
        res,
        feat_df=feat_df,
        unweighted_metrics=unweighted_metrics,
        lodo_metrics=lodo_metrics,
    )

    # Save per-window predictions
    out_df = feat_df.copy()
    out_df["label_pred_lr"] = res["y_pred_lr"]
    out_df["prob_excellent"] = res["proba_lr"][:, 0]
    out_df["prob_acceptable"] = res["proba_lr"][:, 1]
    out_df["prob_unfit"] = res["proba_lr"][:, 2]
    out_df["label_pred_heuristic"] = res["y_pred_heuristic"]
    # Fix #2 — LODO columns
    out_df["lodo_pred_label"] = lodo_metrics["lodo_pred_label"]
    out_df["lodo_proba_excellent"] = lodo_metrics["lodo_proba"][:, 0]
    out_df["lodo_proba_acceptable"] = lodo_metrics["lodo_proba"][:, 1]
    out_df["lodo_proba_unfit"] = lodo_metrics["lodo_proba"][:, 2]
    keep_cols = (
        ["dataset", "file", "subject_id", "window_idx", "t_start_s", "t_end_s",
         "fs", "label"]
        + FEATURE_COLS
        + ["label_pred_lr", "prob_excellent", "prob_acceptable", "prob_unfit",
           "heuristic_label", "label_pred_heuristic",
           "lodo_pred_label", "lodo_proba_excellent",
           "lodo_proba_acceptable", "lodo_proba_unfit"]
    )
    out_df[keep_cols].to_csv(RESULTS_CSV, index=False)
    print(f"[ok] wrote {len(out_df)} rows -> {RESULTS_CSV}")

    # Refit on ALL data → save model
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    final_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            solver="lbfgs",  # multinomial by default in sklearn >= 1.5
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        )),
    ])
    X = feat_df[FEATURE_COLS].to_numpy(dtype=float)
    y = feat_df["label"].astype(str).to_numpy()
    final_pipe.fit(X, y)
    joblib.dump({
        "pipeline": final_pipe,
        "feature_cols": FEATURE_COLS,
        "label_order": LABEL_ORDER,
        "training_n": int(len(X)),
    }, MODEL_PKL)
    print(f"[ok] wrote final LR model -> {MODEL_PKL}")

    # Figures
    make_figures(res, metrics, lodo_metrics=lodo_metrics)

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"LR macro AUC      : {metrics['auc_macro']:.4f}")
    print(f"LR weighted AUC   : {metrics['auc_weighted']:.4f}")
    print(f"LR accuracy       : {metrics['lr_acc']:.4f}")
    print(f"LR macro F1       : {metrics['lr_f1_macro']:.4f}")
    print(f"Heuristic acc     : {metrics['h_acc']:.4f}")
    print(f"Heuristic macro F1: {metrics['h_f1_macro']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
