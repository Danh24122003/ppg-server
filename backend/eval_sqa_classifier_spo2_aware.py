"""SpO2-aware LR classifier on BP-protocol + S09-SQA — apples-to-apples.

+ extension. Train TWO BINARY Logistic Regressions on the SAME 1033
BP+S09 windows (18 subjects, GroupKFold k=5, class_weight=balanced):

  LR-7  (baseline)  : 7 Day-5 features
                      (spectral_purity, hr_bpm, amplitude_ratio,
                       ssqi, ksqi, entropy_amp_dist, tsqi_median).
  LR-9  (treatment) : LR-7 features + 2 SpO2-specific features
                      (r_ratio_std, red_ir_correlation).

Both LRs are trained from scratch on the same 1033 windows; the only
difference is the feature set. The previous attempt compared a 9-feature
binary LR (BP+S09, 18 subjects) against the  3-class LR (43 subjects,
11,850 windows, BIDMC + PPG-DaLiA included) — that was a scope mismatch.
The  LR predictions on these 1033 windows are still emitted under
``label_source='day5_lr_3class'`` for reference only, NOT as headline.

Pipeline:
  1. Filter pseudo_labels.csv to dataset in {BP-protocol, S09-SQA}.
     Drop any 'acceptable' rows (BP+S09 has 0 by  finding).
  2. Recompute 7 base features + 2 SpO2-aware features per window.
       r_ratio_std         : std of per-beat R = (Red_AC/Red_DC)/(IR_AC/IR_DC).
                             Indirect industry support: Berwal et al. 2022
                             IEEE Sensors 22(12):11653.
       red_ir_correlation  : np.corrcoef(ir_filtered, red_filtered)[0, 1].
                             Cross-channel template-matching extension of
                             Orphanidou 2015. Engineering motivated.
  3. Cache features -> sqa_features_spo2_aware_cache.pkl (+ md5 sidecar).
  4. Train two LRs (LR-7 cols, LR-9 cols) with the same GroupKFold splitter.
  5. cross_val_predict(method='predict_proba') -> OOF probabilities each.
  6. Re-run  SpO2 ablation per LR-tier and per  LR-tier
     (reference only).
  7. Headline comparison: macro AUC, F1, accuracy, per-tier pct_in_94_100
     with Wilson 95% CI.
  8. Save final LR-9 model + 3 PDFs.

Constraints:
  - Eval-only NEW script. Does not modify main.py, sqa_per_beat.py,
    eval_sqa_classifier.py, eval_spo2_ablation.py, ml/main.py.
  - 18 subjects is small N for ML; GroupKFold k=5 may have unstable folds.
  - 2 new features carry redundant information (Pearson r ~ -0.85 on
    real data). Both are retained for transparency and per-feature
    importance attribution.

Usage:
    python "backend/eval_sqa_classifier_spo2_aware.py"
    python "backend/eval_sqa_classifier_spo2_aware.py" --no-cache
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
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
from main import compute_sqi_features, bandpass_filter, calculate_spo2_v23  # noqa: E402
from sqa_per_beat import compute_tsqi_per_beat  # noqa: E402

PROJECT_ROOT = ROOT.parent
PSEUDO_LABELS_CSV = ROOT / "pseudo_labels.csv"
DAY5_RESULTS_CSV = ROOT / "eval_sqa_classifier_results.csv"
CACHE_PKL = ROOT / "sqa_features_spo2_aware_cache.pkl"
CACHE_META_JSON = ROOT / "sqa_features_spo2_aware_cache_meta.json"
MAIN_PY_PATH = ROOT / "main.py"
SQA_PER_BEAT_PY_PATH = ROOT / "sqa_per_beat.py"
RESULTS_CSV = ROOT / "eval_sqa_spo2_aware_results.csv"
SUMMARY_CSV = ROOT / "eval_sqa_spo2_aware_summary.csv"
MODEL_PKL = ROOT / "sqa_lr_spo2_aware_model.pkl"
FIGURES_DIR = ROOT / "figures"

BP_CLEANED_DIR = PROJECT_ROOT / "collected_data" / "cleaned"
SQA_DIR = PROJECT_ROOT / "collected_data" / "sqa_experiment"

ELIGIBLE_DATASETS = {"BP-protocol", "S09-SQA"}

FEATURE_COLS_OLD = [
    "spectral_purity",
    "hr_bpm",
    "amplitude_ratio",
    "ssqi",
    "ksqi",
    "entropy_amp_dist",
    "tsqi_median",
]
FEATURE_COLS_NEW = ["r_ratio_std", "red_ir_correlation"]
FEATURE_COLS = FEATURE_COLS_OLD + FEATURE_COLS_NEW  # 9 features cache schema

# Apples-to-apples binary LR setup: 'acceptable' is dropped (0 windows in
# BP+S09 anyway); the headline classifier is binary.
BINARY_LABEL_ORDER = ["excellent", "unfit"]
LABEL_ORDER = ["excellent", "acceptable", "unfit"]  # for  reference rows
NORMOXIC_LO = 94.0
NORMOXIC_HI = 100.0


# ============================================================
# Loaders
# ============================================================
def _resolve_path(dataset: str, file_name: str) -> Path | None:
    if dataset == "BP-protocol":
        return BP_CLEANED_DIR / file_name
    if dataset == "S09-SQA":
        return SQA_DIR / file_name
    return None


def load_ir_red(dataset: str, file_name: str) -> tuple[np.ndarray, np.ndarray] | None:
    path = _resolve_path(dataset, file_name)
    if path is None or not path.exists():
        return None
    df = pd.read_csv(
        path, comment="#",
        usecols=lambda c: c in {"timestamp_ms", "ir", "red"},
    )
    if "ir" not in df.columns or "red" not in df.columns:
        return None
    return (
        df["ir"].to_numpy(dtype=float),
        df["red"].to_numpy(dtype=float),
    )


# ============================================================
# Feature extraction
# ============================================================
def compute_spo2_aware_features(
    ir: np.ndarray, red: np.ndarray, fs: int
) -> dict:
    """Compute 2 SpO2-specific features for a window.

    r_ratio_std        : std of per-beat R-ratio.
    red_ir_correlation : Pearson r between bandpassed IR and Red.

    Bug #2 fix (2026-05-08): the guard ``np.std(red_f) < 1e-9`` was
    bypassed by filtfilt numerical noise (~5e-7) on truly flat signals,
    producing spurious ~0.996 correlations and finite r_ratio_std values
    sourced from numerical noise. We now check the RAW signal std before
    filtering: a flat raw signal has no physiological oscillation and must
    return red_ir_correlation=0.0 and r_ratio_std=NaN.
    """
    out = {"r_ratio_std": float("nan"), "red_ir_correlation": 0.0}

    if len(ir) < 16 or len(red) < 16 or len(ir) != len(red):
        return out

    # Bug #2 fix: guard on RAW std (filtfilt noise on flat signal can pass
    # any threshold on filtered std). 1.0 ADC unit is physiologically tiny
    # for MAX30102 (typical AC 100-3000); below that the signal is flat.
    if np.std(red) < 1.0 or np.std(ir) < 1.0:
        return out

    # Bandpass both channels [0.5, 4.0] Hz (production filter)
    red_f = bandpass_filter(red, fs, 0.5, 4.0)
    ir_f = bandpass_filter(ir, fs, 0.5, 4.0)

    # red_ir_correlation
    if np.std(red_f) < 1e-9 or np.std(ir_f) < 1e-9:
        out["red_ir_correlation"] = 0.0
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r_corr = float(np.corrcoef(ir_f, red_f)[0, 1])
        if not np.isfinite(r_corr):
            r_corr = 0.0
        out["red_ir_correlation"] = r_corr

    # r_ratio_std: per-beat R computation reusing find_peaks
    peaks, _ = find_peaks(ir_f, distance=int(fs * 0.3))
    if len(peaks) < 4:
        return out

    half_w = int(fs * 0.3)
    min_len = int(fs * 0.4)
    r_per_beat: list[float] = []
    for p in peaks:
        s, e = max(0, p - half_w), min(len(ir), p + half_w + 1)
        if e - s < min_len:
            continue
        ir_seg = ir[s:e]
        red_seg = red[s:e]
        ir_seg_f = ir_f[s:e]
        red_seg_f = red_f[s:e]
        ac_ir = float(np.sqrt(np.mean(ir_seg_f ** 2)))
        ac_red = float(np.sqrt(np.mean(red_seg_f ** 2)))
        dc_ir = float(np.median(ir_seg))
        dc_red = float(np.median(red_seg))
        if dc_ir < 1e-3 or dc_red < 1e-3 or ac_ir < 1e-9:
            continue
        r = (ac_red / dc_red) / (ac_ir / dc_ir)
        if np.isfinite(r):
            r_per_beat.append(r)

    if len(r_per_beat) >= 3:
        out["r_ratio_std"] = float(np.std(r_per_beat, ddof=0))
    return out


def compute_window_features_9(
    ir: np.ndarray, red: np.ndarray, fs: int
) -> dict:
    """Return dict with 9 features (7 from  + 2 SpO2-specific)."""
    out: dict = {k: 0.0 for k in FEATURE_COLS_OLD}
    out["r_ratio_std"] = float("nan")
    out["red_ir_correlation"] = 0.0

    if len(ir) < 16:
        return out

    # 7 base features (use IR channel like )
    b, a = butter(4, [0.5, 4.0], btype="band", fs=fs)
    filtered_ir = filtfilt(b, a, ir)
    peaks, _ = find_peaks(filtered_ir, distance=int(fs * 0.3))

    feats = compute_sqi_features(ir, filtered_ir, list(peaks), fs)
    for k in FEATURE_COLS_OLD[:6]:
        out[k] = float(feats.get(k, 0.0))

    tsqi_list, _ = compute_tsqi_per_beat(filtered_ir, peaks, fs, 600, 5)
    if tsqi_list:
        out["tsqi_median"] = float(np.median(tsqi_list))

    # 2 new SpO2-aware features
    new_feats = compute_spo2_aware_features(ir, red, fs)
    out["r_ratio_std"] = new_feats["r_ratio_std"]
    out["red_ir_correlation"] = new_feats["red_ir_correlation"]
    return out


# ============================================================
# Cache metadata (mirror  strategy)
# ============================================================
def _md5_first_64kb(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read(64 * 1024))
    return h.hexdigest()


CACHE_VERSION = 2  # bump when feature semantics change (Bug #2 raw-std guard)


def _compute_cache_meta(df_labels: pd.DataFrame) -> dict:
    return {
        "cache_version": CACHE_VERSION,
        "n_rows": int(len(df_labels)),
        "labels_csv_md5": _md5_first_64kb(PSEUDO_LABELS_CSV),
        "main_py_mtime": os.path.getmtime(MAIN_PY_PATH),
        "sqa_per_beat_py_mtime": os.path.getmtime(SQA_PER_BEAT_PY_PATH),
        "feature_cols": FEATURE_COLS,
    }


def _check_cache_meta(current: dict) -> tuple[bool, str]:
    if not CACHE_META_JSON.exists():
        return False, "metadata sidecar missing"
    try:
        with open(CACHE_META_JSON, "r", encoding="utf-8") as f:
            saved = json.load(f)
    except Exception as e:
        return False, f"metadata read failed: {type(e).__name__}: {e}"
    for key in ("cache_version", "n_rows", "labels_csv_md5", "main_py_mtime",
                "sqa_per_beat_py_mtime", "feature_cols"):
        if key not in saved:
            return False, f"metadata key missing: {key}"
        if saved[key] != current[key]:
            return False, f"{key} changed (cache vs current)"
    return True, ""


# ============================================================
# Build feature matrix (BP + S09 only)
# ============================================================
def build_feature_matrix(
    df_labels: pd.DataFrame,
    use_cache: bool = True,
) -> pd.DataFrame:
    current_meta = _compute_cache_meta(df_labels)
    if use_cache and CACHE_PKL.exists():
        valid, reason = _check_cache_meta(current_meta)
        if valid:
            cached = pd.read_pickle(CACHE_PKL)
            if len(cached) == len(df_labels):
                print(f"[cache] hit: {len(cached)} rows from {CACHE_PKL.name}")
                return cached
            print(f"[cache] stale (cached {len(cached)} vs labels {len(df_labels)}); rebuild")
        else:
            print(f"[cache] invalidated: {reason}")

    print(f"[features] computing 9 features per window for {len(df_labels)} windows ...")
    file_keys = df_labels[["dataset", "file"]].drop_duplicates().reset_index(drop=True)
    n_files = len(file_keys)

    all_rows: list[dict] = []
    t0 = time.time()
    for fi, (_, key) in enumerate(file_keys.iterrows(), start=1):
        dataset = key["dataset"]
        file_name = key["file"]
        loaded = load_ir_red(dataset, file_name)
        if loaded is None:
            print(f"  [skip] {dataset}/{file_name}: missing or no ir/red columns")
            continue
        ir_full, red_full = loaded
        sub = df_labels[
            (df_labels["dataset"] == dataset) & (df_labels["file"] == file_name)
        ]
        elapsed = time.time() - t0
        print(
            f"  [{fi:3d}/{n_files}] {dataset:<12} {file_name:<46} "
            f"len={len(ir_full):>7} windows={len(sub):>3} elapsed={elapsed:6.1f}s"
        )

        for _, row in sub.iterrows():
            fs = int(row["fs"])
            t_start = float(row["t_start_s"])
            t_end = float(row["t_end_s"])
            i0 = int(t_start * fs)
            i1 = min(int(t_end * fs), len(ir_full), len(red_full))
            if i1 <= i0:
                continue
            ir_win = ir_full[i0:i1]
            red_win = red_full[i0:i1]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                feats = compute_window_features_9(ir_win, red_win, fs)
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

        del ir_full, red_full
        gc.collect()

    feat_df = pd.DataFrame(all_rows)
    feat_df.to_pickle(CACHE_PKL)
    with open(CACHE_META_JSON, "w", encoding="utf-8") as f:
        json.dump(current_meta, f, indent=2)
    print(f"[cache] wrote {len(feat_df)} rows -> {CACHE_PKL.name} (+ meta sidecar)")
    return feat_df


# ============================================================
# Train one binary LR (LR-7 or LR-9)
# ============================================================
def train_binary_lr(
    feat_df: pd.DataFrame,
    feature_cols: list[str],
    tag: str,
) -> dict:
    """Train a binary LR (excellent vs unfit) with subject-level GroupKFold.

    Returns dict with y_true, y_pred, proba_excellent, fold_coefs, fold_classes.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import GroupKFold, cross_val_predict

    X = feat_df[feature_cols].to_numpy(dtype=float)
    y = feat_df["label"].astype(str).to_numpy()
    groups = feat_df["subject_id"].astype(str).to_numpy()

    n_subjects = len(np.unique(groups))
    n_splits = min(5, n_subjects)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            solver="lbfgs",
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        )),
    ])
    cv = GroupKFold(n_splits=n_splits)

    fold_subject_counts = [
        len(np.unique(groups[te_idx])) for _, te_idx in cv.split(X, y, groups)
    ]
    print(f"[{tag}] GroupKFold n_splits={n_splits} (n_subjects={n_subjects}); "
          f"subjects per test fold: {fold_subject_counts}")

    print(f"[{tag}] computing OOF predict_proba ...")
    t0 = time.time()
    proba_oof = cross_val_predict(
        pipe, X, y, cv=cv, groups=groups,
        method="predict_proba", n_jobs=-1,
    )
    print(f"[{tag}] done in {time.time()-t0:.1f}s")

    # Refit per fold to extract coefficients
    fold_coefs: list[np.ndarray] = []
    fold_classes: list[np.ndarray] = []
    for tr_idx, _te_idx in cv.split(X, y, groups):
        p = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(
                solver="lbfgs",
                max_iter=2000,
                class_weight="balanced",
                random_state=42,
            )),
        ])
        p.fit(X[tr_idx], y[tr_idx])
        fold_coefs.append(p.named_steps["lr"].coef_.copy())
        fold_classes.append(p.named_steps["lr"].classes_.copy())

    cv_classes_ = np.array(sorted(np.unique(y)))
    proba_ordered = np.zeros((proba_oof.shape[0], len(BINARY_LABEL_ORDER)))
    for i, lab in enumerate(BINARY_LABEL_ORDER):
        hits = np.where(cv_classes_ == lab)[0]
        if hits.size:
            proba_ordered[:, i] = proba_oof[:, int(hits[0])]
    y_pred = np.array(BINARY_LABEL_ORDER)[proba_ordered.argmax(axis=1)]

    return {
        "tag": tag,
        "feature_cols": feature_cols,
        "y_true": y,
        "y_pred": y_pred,
        "proba": proba_ordered,
        "proba_excellent": proba_ordered[:, 0],
        "fold_coefs": fold_coefs,
        "fold_classes": fold_classes,
    }


def report_binary_metrics(res: dict) -> dict:
    """Compute macro AUC, F1, accuracy + per-feature importance for one LR."""
    from sklearn.metrics import (
        roc_auc_score,
        accuracy_score,
        f1_score,
        confusion_matrix,
    )

    y_true = res["y_true"]
    y_pred = res["y_pred"]
    proba_exc = res["proba_excellent"]
    feature_cols = res["feature_cols"]
    tag = res["tag"]

    # Binary classification metrics: positive class = 'excellent'
    y_bin = (y_true == "excellent").astype(int)
    auc = float(roc_auc_score(y_bin, proba_exc))
    acc = float(accuracy_score(y_true, y_pred))
    f1_macro = float(f1_score(y_true, y_pred, labels=BINARY_LABEL_ORDER,
                              average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, labels=BINARY_LABEL_ORDER,
                                 average="weighted", zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=BINARY_LABEL_ORDER)

    # Feature importance: binary LR has coef_ of shape (1, n_features).
    # Sum |coef| across folds, normalised by fold count.
    n_features = len(feature_cols)
    abs_sum = np.zeros(n_features)
    n_folds = 0
    for coef, _cls_arr in zip(res["fold_coefs"], res["fold_classes"]):
        abs_sum += np.abs(coef[0])
        n_folds += 1
    importance = abs_sum / max(n_folds, 1)
    rank_idx = np.argsort(-importance)

    print()
    print("=" * 80)
    print(f"{tag} — binary LR (excellent vs unfit) — GroupKFold class_weight=balanced")
    print("=" * 80)
    print(f"Macro ROC AUC (binary): {auc:.4f}")
    print(f"Accuracy              : {acc:.4f}")
    print(f"Macro F1              : {f1_macro:.4f}")
    print(f"Weighted F1           : {f1_weighted:.4f}")
    print(f"Confusion matrix (rows=true, cols=pred, order={BINARY_LABEL_ORDER}):")
    print(cm)
    print()
    print("Per-feature importance (mean |coef| across folds):")
    for rk, idx in enumerate(rank_idx, start=1):
        print(f"  {rk}. {feature_cols[idx]:<22} {importance[idx]:.3f}")

    return {
        "tag": tag,
        "auc": auc,
        "acc": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "cm": cm,
        "importance": importance,
        "rank_idx": rank_idx,
        "feature_cols": feature_cols,
    }


# ============================================================
#  SpO2 ablation re-run
# ============================================================
def wilson_ci_95(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    z = 1.96
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half_width = z * (p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5 / denom
    lo = max(0.0, (centre - half_width) * 100.0)
    hi = min(100.0, (centre + half_width) * 100.0)
    return (lo, hi)


def compute_spo2_per_window(feat_df: pd.DataFrame) -> pd.DataFrame:
    """Call calculate_spo2_v23 per window. Cache IR/Red per file."""
    print()
    print(f"[spo2] computing SpO2 for {len(feat_df)} windows ...")
    cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray] | None] = {}
    rows: list[dict] = []
    t0 = time.time()
    n = len(feat_df)
    for i, (_, r) in enumerate(feat_df.iterrows()):
        if i % 200 == 0 and i > 0:
            print(f"  {i}/{n} elapsed={time.time()-t0:.0f}s")
        ds = r["dataset"]
        fn = r["file"]
        key = (ds, fn)
        if key not in cache:
            loaded = load_ir_red(ds, fn)
            cache[key] = loaded
        loaded = cache[key]
        rec = {
            "dataset": ds,
            "file": fn,
            "subject_id": r["subject_id"],
            "window_idx": int(r["window_idx"]),
            "t_start_s": float(r["t_start_s"]),
            "t_end_s": float(r["t_end_s"]),
            "fs": int(r["fs"]),
            "spo2_pred": float("nan"),
            "valid_flag": False,
            "ratio_r": float("nan"),
            "rejection_reason": "",
        }
        if loaded is None:
            rec["rejection_reason"] = "file_missing"
            rows.append(rec)
            continue
        ir, red = loaded
        fs = int(r["fs"])
        s = int(float(r["t_start_s"]) * fs)
        e = min(int(float(r["t_end_s"]) * fs), len(ir), len(red))
        if e <= s:
            rec["rejection_reason"] = "window_out_of_bounds"
            rows.append(rec)
            continue
        spo2, R, _, _, status = calculate_spo2_v23(ir[s:e], red[s:e], fs)
        rec["ratio_r"] = float(R) if R == R else float("nan")
        if status == "ok" and spo2 is not None:
            rec["spo2_pred"] = float(spo2)
            rec["valid_flag"] = True
        else:
            rec["rejection_reason"] = status
        rows.append(rec)
    print(f"[spo2] done {n}/{n} in {time.time()-t0:.1f}s")
    return pd.DataFrame(rows)


def build_summary(
    spo2_df: pd.DataFrame,
    pred_columns: dict[str, np.ndarray],
    tier_orders: dict[str, list[str]],
) -> pd.DataFrame:
    """Build per-(label_source, dataset, tier) summary with Wilson CIs.

    pred_columns : {label_source: prediction array aligned with spo2_df}
    tier_orders  : {label_source: list of tier strings to iterate over}
    """
    spo2_df = spo2_df.reset_index(drop=True).copy()
    rows: list[dict] = []
    for label_source, pred in pred_columns.items():
        col = f"_pred_{label_source}"
        spo2_df[col] = pred
        tier_order = tier_orders[label_source]
        for ds, ds_grp in spo2_df.groupby("dataset"):
            for tier in tier_order:
                sub = ds_grp[ds_grp[col] == tier]
                n_total = int(len(sub))
                valid = sub[sub["valid_flag"]]
                n_valid = int(len(valid))
                rec: dict = {
                    "label_source": label_source,
                    "dataset": ds,
                    "tier": tier,
                    "n_total": n_total,
                    "n_valid": n_valid,
                    "pct_valid": float("nan"),
                    "pct_in_94_100": float("nan"),
                    "pct_in_94_100_ci_lo": float("nan"),
                    "pct_in_94_100_ci_hi": float("nan"),
                    "pct_in_94_100_of_total": float("nan"),
                    "ci_lo_of_total": float("nan"),
                    "ci_hi_of_total": float("nan"),
                    "median_spo2": float("nan"),
                    "top_rejection_reason": "",
                }
                if n_total == 0:
                    rows.append(rec)
                    continue
                rec["pct_valid"] = n_valid / n_total * 100.0
                spo2_arr = valid["spo2_pred"].to_numpy(dtype=float)
                if spo2_arr.size > 0:
                    n_in = int(((spo2_arr >= NORMOXIC_LO) & (spo2_arr <= NORMOXIC_HI)).sum())
                    rec["pct_in_94_100"] = n_in / spo2_arr.size * 100.0
                    ci_lo, ci_hi = wilson_ci_95(n_in, int(spo2_arr.size))
                    rec["pct_in_94_100_ci_lo"] = ci_lo
                    rec["pct_in_94_100_ci_hi"] = ci_hi
                    rec["median_spo2"] = float(np.median(spo2_arr))
                    rec["pct_in_94_100_of_total"] = n_in / n_total * 100.0
                    ci_lo_t, ci_hi_t = wilson_ci_95(n_in, n_total)
                    rec["ci_lo_of_total"] = ci_lo_t
                    rec["ci_hi_of_total"] = ci_hi_t
                # top rejection reason among the n_total - n_valid invalid rows
                inv = sub[~sub["valid_flag"]]
                if len(inv) > 0:
                    reasons = inv["rejection_reason"].astype(str)
                    reasons = reasons[reasons != ""]
                    if len(reasons) > 0:
                        rec["top_rejection_reason"] = reasons.value_counts().index[0]
                rows.append(rec)
    return pd.DataFrame(rows)


def print_headline_table(
    summary: pd.DataFrame,
    metrics_lr7: dict,
    metrics_lr9: dict,
) -> str:
    """Print apples-to-apples LR-7 vs LR-9 comparison table.

    Returns verdict: BETTER / SAME / WORSE.
    """
    print()
    print("=" * 100)
    print("HEADLINE: apples-to-apples LR-7 vs LR-9 (binary, same 1033 BP+S09 windows)")
    print("=" * 100)

    # Aggregate per-tier metrics across both datasets (BP + S09 combined)
    def _row(label_source: str, tier: str) -> dict:
        sub = summary[
            (summary["label_source"] == label_source) & (summary["tier"] == tier)
        ]
        if sub.empty:
            return {"n_total": 0, "n_valid": 0, "n_in_94_100": float("nan"),
                    "pct_in_94_100": float("nan"), "ci_lo": float("nan"),
                    "ci_hi": float("nan")}
        n_total = int(sub["n_total"].sum())
        n_valid = int(sub["n_valid"].sum())
        n_in = int(((sub["pct_in_94_100"].fillna(0) / 100.0) * sub["n_valid"]).round().sum())
        pct = n_in / n_valid * 100.0 if n_valid > 0 else float("nan")
        ci_lo, ci_hi = wilson_ci_95(n_in, n_valid) if n_valid > 0 else (float("nan"), float("nan"))
        return {
            "n_total": n_total,
            "n_valid": n_valid,
            "n_in_94_100": n_in,
            "pct_in_94_100": pct,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
        }

    lr7_exc = _row("lr7_binary", "excellent")
    lr7_unf = _row("lr7_binary", "unfit")
    lr9_exc = _row("lr9_binary", "excellent")
    lr9_unf = _row("lr9_binary", "unfit")

    # Discrimination Δ pp = excellent% in [94,100] - unfit% in [94,100]
    disc_lr7 = lr7_exc["pct_in_94_100"] - lr7_unf["pct_in_94_100"]
    disc_lr9 = lr9_exc["pct_in_94_100"] - lr9_unf["pct_in_94_100"]

    # CI overlap check: excellent CI low > unfit CI high?
    overlap_lr7 = not (
        not np.isnan(lr7_exc["ci_lo"]) and not np.isnan(lr7_unf["ci_hi"])
        and lr7_exc["ci_lo"] > lr7_unf["ci_hi"]
    )
    overlap_lr9 = not (
        not np.isnan(lr9_exc["ci_lo"]) and not np.isnan(lr9_unf["ci_hi"])
        and lr9_exc["ci_lo"] > lr9_unf["ci_hi"]
    )

    print(f"{'Metric':<30}{'LR-7 (baseline)':>22}{'LR-9 (with 2 SpO2)':>22}{'Delta':>20}")
    print("-" * 100)
    print(f"{'N_excellent (predicted)':<30}{lr7_exc['n_total']:>22}{lr9_exc['n_total']:>22}"
          f"{lr9_exc['n_total']-lr7_exc['n_total']:>+20}")
    print(f"{'N_unfit (predicted)':<30}{lr7_unf['n_total']:>22}{lr9_unf['n_total']:>22}"
          f"{lr9_unf['n_total']-lr7_unf['n_total']:>+20}")
    print(f"{'pct_in_94_100 excellent':<30}{lr7_exc['pct_in_94_100']:>21.2f}%"
          f"{lr9_exc['pct_in_94_100']:>21.2f}%"
          f"{lr9_exc['pct_in_94_100']-lr7_exc['pct_in_94_100']:>+19.2f}pp")
    print(f"{'pct_in_94_100 unfit':<30}{lr7_unf['pct_in_94_100']:>21.2f}%"
          f"{lr9_unf['pct_in_94_100']:>21.2f}%"
          f"{lr9_unf['pct_in_94_100']-lr7_unf['pct_in_94_100']:>+19.2f}pp")
    print(f"{'Discrimination Delta (pp)':<30}{disc_lr7:>21.2f}pp{disc_lr9:>21.2f}pp"
          f"{disc_lr9-disc_lr7:>+19.2f}pp")
    print(f"{'Wilson CI overlap?':<30}{('YES' if overlap_lr7 else 'NO'):>22}"
          f"{('YES' if overlap_lr9 else 'NO'):>22}"
          f"{'':>20}")
    print()
    print(f"{'GroupKFold k=5 (18 subj)':<30}{'':>22}{'':>22}")
    print(f"  Macro AUC (binary)        {metrics_lr7['auc']:>21.4f} {metrics_lr9['auc']:>21.4f}"
          f"{metrics_lr9['auc']-metrics_lr7['auc']:>+19.4f}")
    print(f"  Macro F1                  {metrics_lr7['f1_macro']:>21.4f} {metrics_lr9['f1_macro']:>21.4f}"
          f"{metrics_lr9['f1_macro']-metrics_lr7['f1_macro']:>+19.4f}")
    print(f"  Accuracy                  {metrics_lr7['acc']:>21.4f} {metrics_lr9['acc']:>21.4f}"
          f"{metrics_lr9['acc']-metrics_lr7['acc']:>+19.4f}")
    print()
    print("Top 3 features (LR-9):")
    for rk, idx in enumerate(metrics_lr9["rank_idx"][:3], start=1):
        print(f"  {rk}. {metrics_lr9['feature_cols'][idx]:<22} {metrics_lr9['importance'][idx]:.3f}")

    # Verdict logic
    auc_delta = metrics_lr9["auc"] - metrics_lr7["auc"]
    disc_delta = disc_lr9 - disc_lr7
    if auc_delta > 0.005 and disc_delta > 0.5:
        verdict = "BETTER"
    elif auc_delta < -0.005 or disc_delta < -0.5:
        verdict = "WORSE"
    else:
        verdict = "SAME"
    print()
    print(f"VERDICT: LR-9 is {verdict} than LR-7 "
          f"(AUC delta {auc_delta:+.4f}, discrimination delta {disc_delta:+.2f}pp)")
    return verdict


# ============================================================
# Figures
# ============================================================
def make_figures(
    res_lr7: dict,
    res_lr9: dict,
    metrics_lr7: dict,
    metrics_lr9: dict,
    summary: pd.DataFrame,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc as sk_auc

    FIGURES_DIR.mkdir(exist_ok=True)

    # --- ROC curves: LR-7 vs LR-9 binary, positive=excellent ---
    fig, ax = plt.subplots(figsize=(6, 5.5))
    for res, metrics, color, tag in [
        (res_lr7, metrics_lr7, "#1f77b4", "LR-7"),
        (res_lr9, metrics_lr9, "#d62728", "LR-9"),
    ]:
        y_bin = (res["y_true"] == "excellent").astype(int)
        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            continue
        fpr, tpr, _ = roc_curve(y_bin, res["proba_excellent"])
        a = sk_auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{tag} (AUC={a:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(
        "LR-7 vs LR-9 ROC (binary, BP+S09 1033 windows, GroupKFold)\n"
        f"LR-7 AUC={metrics_lr7['auc']:.3f}  LR-9 AUC={metrics_lr9['auc']:.3f}"
    )
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_spo2_aware_lr_roc.pdf")
    plt.close(fig)
    print(f"[fig] wrote {FIGURES_DIR/'fig_spo2_aware_lr_roc.pdf'}")

    # --- Feature importance for LR-9 ---
    importance = metrics_lr9["importance"]
    feature_cols = metrics_lr9["feature_cols"]
    n_feat = len(feature_cols)
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(n_feat)
    colors = ["#1f77b4"] * (n_feat - 2) + ["#d62728", "#d62728"]
    ax.bar(x, importance, color=colors, edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(feature_cols, rotation=30, ha="right")
    ax.set_ylabel("Mean |coefficient| across folds")
    ax.set_title(
        "LR-9 binary feature importance (red bars = 2 new SpO2-aware features)"
    )
    ax.grid(True, axis="y", alpha=0.3)
    # Highlight new features
    for new_idx in [feature_cols.index(f) for f in FEATURE_COLS_NEW]:
        ax.axvspan(new_idx - 0.5, new_idx + 0.5, color="yellow", alpha=0.15, zorder=0)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_spo2_aware_lr_feature_importance.pdf")
    plt.close(fig)
    print(f"[fig] wrote {FIGURES_DIR/'fig_spo2_aware_lr_feature_importance.pdf'}")

    # --- Comparison: LR-7 vs LR-9 vs Day5 LR pct_in_94_100 per (dataset, tier) ---
    label_sources = ["lr7_binary", "lr9_binary", "day5_lr_3class"]
    titles = ["LR-7 (baseline)", "LR-9 (with 2 SpO2 feat)", " LR (3-class, ref only)"]
    datasets = sorted(summary["dataset"].unique())
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.4), sharey=True)
    bar_w = 0.35

    for ax, ls, title in zip(axes, label_sources, titles):
        # Per-row tier ordering depends on label source
        tier_ord = LABEL_ORDER if ls == "day5_lr_3class" else BINARY_LABEL_ORDER
        for di, ds in enumerate(datasets):
            sub = summary[
                (summary.label_source == ls) & (summary.dataset == ds)
            ].set_index("tier").reindex(tier_ord)
            pct_in = sub["pct_in_94_100"].to_numpy(dtype=float)
            ci_lo = sub["pct_in_94_100_ci_lo"].to_numpy(dtype=float)
            ci_hi = sub["pct_in_94_100_ci_hi"].to_numpy(dtype=float)
            xs = np.arange(len(tier_ord)) + di * bar_w
            heights = np.nan_to_num(pct_in, nan=0.0)
            err_lo = np.clip(np.nan_to_num(pct_in - ci_lo, nan=0.0), 0.0, None)
            err_hi = np.clip(np.nan_to_num(ci_hi - pct_in, nan=0.0), 0.0, None)
            ax.bar(
                xs, heights, bar_w,
                label=ds, edgecolor="black", linewidth=0.4, alpha=0.7,
                yerr=[err_lo, err_hi], capsize=3,
            )
            for x_v, p, n in zip(xs, pct_in, sub["n_total"].to_numpy()):
                if not np.isnan(p):
                    ax.text(x_v, p + 2.5, f"N={int(n)}", ha="center", fontsize=7)
        ax.set_xticks(np.arange(len(tier_ord)) + bar_w * (len(datasets) - 1) / 2)
        ax.set_xticklabels(tier_ord)
        ax.set_title(title)
        ax.set_xlabel("LR-predicted tier")
        ax.set_ylim(0, 110)
        ax.grid(axis="y", linestyle=":", alpha=0.4)
    axes[0].set_ylabel("% SpO2 in [94, 100]% (Wilson 95% CI bars)")
    axes[0].legend(loc="lower left", fontsize=8)
    fig.suptitle("LR-7 vs LR-9 vs  LR — pct SpO2 in normoxic range per tier")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_spo2_aware_lr_comparison.pdf")
    plt.close(fig)
    print(f"[fig] wrote {FIGURES_DIR/'fig_spo2_aware_lr_comparison.pdf'}")


# ============================================================
# Honest framing
# ============================================================
def print_honest_framing(
    feat_df: pd.DataFrame,
    n_subjects: int,
    feature_corr_r_ratio_vs_corr: float,
) -> None:
    print()
    print("=" * 80)
    print("HONEST FRAMING NOTES (+ SpO2-aware LR)")
    print("=" * 80)
    print(f"""
1. SMALL N CAVEAT: Only {n_subjects} subjects (14 BP + 4 S09) and
   {len(feat_df)} windows. This is small for ML training.
   GroupKFold k=5 may produce unstable folds (some folds may have only
   3-4 subjects in test set). Treat AUC point estimates with care.

2. CROSS-DATASET GENERALIZATION NOT TESTED: BP-protocol and S09-SQA
   only. No held-out external dataset (BIDMC and PPG-DaLiA cannot be
   used because they lack the Red channel). For cross-dataset
   generalization the production classifier remains the  LR
   (7 features, 43 subjects, 11850 windows, LODO macro AUC 0.58).

3. ENGINEERING-MOTIVATED FEATURES:
   - r_ratio_std: indirect industry support via Berwal et al. 2022
     IEEE Sensors 22(12):11653 (R-ratio stability for SpO2 SQA
     feedback). Not validated standalone in PPG SQA classifier
     literature.
   - red_ir_correlation: cross-channel template-matching extension
     of Orphanidou 2015. No direct PPG SQA literature precedent at
     time of writing.

4. CIRCULAR DEPENDENCY note: r_ratio_std reuses calculate_spo2_v23
   internals (AC/DC + R-ratio). High r_ratio_std implies the SpO2
   estimator is unstable across beats within a window. Using this
   to predict whether SpO2 will be valid is partially auto-correlated
   with the outcome metric. The honest interpretation is "windows
   where the SpO2 estimator is unstable beat-to-beat are flagged
   as unfit by this LR" — useful, even if not a strict independent
   ML evaluation.

5. APPLES-TO-APPLES BINARY COMPARISON: LR-7 and LR-9 are both trained
   on the SAME 1033 BP+S09 windows × {n_subjects} subjects, with
   GroupKFold k=5 and class_weight=balanced, and both are BINARY
   (excellent vs unfit; 'acceptable' is dropped because BP+S09 has
   0 acceptable windows by  finding). The previous attempt that
   compared a 9-feature LR against the  LR (3-class, 43 subjects,
   11,850 windows incl. BIDMC + DaLiA) was an INVALID SCOPE MISMATCH —
   that comparison is preserved as 'day5_lr_3class' rows in the
   summary CSV for reference only, NOT as primary headline.

6. FEATURE REDUNDANCY: r_ratio_std and red_ir_correlation are
   anti-correlated on real data (Pearson r = {feature_corr_r_ratio_vs_corr:+.3f}).
   Including both adds limited extra signal. We retain both for
   transparency: r_ratio_std for SpO2-specific R-ratio stability
   (Berwal 2022); red_ir_correlation for cross-channel coherence
   (Orphanidou 2015 extension). Per-feature importance attributes
   weight to whichever gradient the LR finds easier to use, the
   sum of |coef| of the two is the genuine joint information gain.

7. SMALL-N FOLD INSTABILITY: 18 subjects with k=5 GroupKFold splits
   into roughly 3-4 subjects per test fold. AUC and F1 point
   estimates are sensitive to fold composition; bootstrap CIs are
   not provided here. Treat the LR-7 vs LR-9 delta as a comparison
   under a fixed random_state=42 splitter, not as a population claim.
""")
    print("=" * 80)


# ============================================================
# Main
# ============================================================
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-cache", action="store_true",
                    help="Force feature recompute (ignore cache pickle)")
    args = ap.parse_args()

    if not PSEUDO_LABELS_CSV.exists():
        print(f"[error] {PSEUDO_LABELS_CSV} not found.")
        return 1
    if not DAY5_RESULTS_CSV.exists():
        print(f"[error] {DAY5_RESULTS_CSV} not found. Run eval_sqa_classifier.py first.")
        return 1

    np.random.seed(42)

    df_labels_all = pd.read_csv(PSEUDO_LABELS_CSV)
    df_labels = df_labels_all[df_labels_all["dataset"].isin(ELIGIBLE_DATASETS)].reset_index(drop=True)
    n_subjects = df_labels["subject_id"].nunique()
    print(f"[load] pseudo_labels.csv: {len(df_labels_all)} total -> "
          f"{len(df_labels)} after filter to {sorted(ELIGIBLE_DATASETS)}")
    print(f"       n_subjects={n_subjects}, datasets={df_labels['dataset'].value_counts().to_dict()}")
    print(f"       label distribution: {df_labels['label'].value_counts().to_dict()}")

    feat_df = build_feature_matrix(df_labels, use_cache=not args.no_cache)
    if feat_df.empty:
        print("[error] no features computed")
        return 2

    n_pre = len(feat_df)
    mask = np.isfinite(feat_df[FEATURE_COLS].to_numpy()).all(axis=1)
    feat_df = feat_df[mask].reset_index(drop=True)
    if len(feat_df) < n_pre:
        print(f"[clean] dropped {n_pre - len(feat_df)} rows with non-finite features")

    # Filter to binary (drop 'acceptable'). BP+S09 has 0 acceptable per .
    n_pre_bin = len(feat_df)
    feat_df = feat_df[feat_df["label"].isin(BINARY_LABEL_ORDER)].reset_index(drop=True)
    if len(feat_df) < n_pre_bin:
        print(f"[clean] dropped {n_pre_bin - len(feat_df)} 'acceptable' rows for binary LR")
    n_subjects_train = feat_df["subject_id"].nunique()
    print(f"[binary] train set: {len(feat_df)} windows × {n_subjects_train} subjects, "
          f"labels={feat_df['label'].value_counts().to_dict()}")

    # ---- Train LR-7 (baseline) and LR-9 (treatment) on SAME windows ----
    res_lr7 = train_binary_lr(feat_df, FEATURE_COLS_OLD, tag="LR-7")
    metrics_lr7 = report_binary_metrics(res_lr7)
    res_lr9 = train_binary_lr(feat_df, FEATURE_COLS, tag="LR-9")
    metrics_lr9 = report_binary_metrics(res_lr9)

    # Refit LR-9 on ALL data and save final model bundle
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    final_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            solver="lbfgs",
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        )),
    ])
    X9 = feat_df[FEATURE_COLS].to_numpy(dtype=float)
    y_bin = feat_df["label"].astype(str).to_numpy()
    final_pipe.fit(X9, y_bin)
    joblib.dump({
        "pipeline": final_pipe,
        "feature_cols": FEATURE_COLS,
        "feature_cols_old": FEATURE_COLS_OLD,
        "feature_cols_new": FEATURE_COLS_NEW,
        "label_order": BINARY_LABEL_ORDER,
        "is_binary": True,
        "training_n": int(len(X9)),
        "training_subjects": int(n_subjects_train),
    }, MODEL_PKL)
    print(f"[ok] wrote final LR-9 binary model -> {MODEL_PKL.name}")

    # -- -- SpO2 ablation re-run ----
    spo2_df = compute_spo2_per_window(feat_df)

    # Pull  LR predictions (3-class) for reference rows
    day5_df = pd.read_csv(DAY5_RESULTS_CSV)
    day5_key = day5_df[["dataset", "file", "window_idx", "label_pred_lr"]].copy()
    feat_key = feat_df[["dataset", "file", "window_idx"]].copy()
    feat_key["_row_idx"] = np.arange(len(feat_key))
    merged = feat_key.merge(
        day5_key, on=["dataset", "file", "window_idx"], how="left"
    ).sort_values("_row_idx").reset_index(drop=True)
    if merged["label_pred_lr"].isna().any():
        n_missing = int(merged["label_pred_lr"].isna().sum())
        print(f"[warn] {n_missing} rows missing  LR prediction; "
              "filling with 'unfit' (defensive)")
        merged["label_pred_lr"] = merged["label_pred_lr"].fillna("unfit")
    day5_pred = merged["label_pred_lr"].to_numpy()

    pred_columns = {
        "lr7_binary": res_lr7["y_pred"],
        "lr9_binary": res_lr9["y_pred"],
        "day5_lr_3class": day5_pred,
    }
    tier_orders = {
        "lr7_binary": BINARY_LABEL_ORDER,
        "lr9_binary": BINARY_LABEL_ORDER,
        "day5_lr_3class": LABEL_ORDER,
    }

    summary = build_summary(spo2_df, pred_columns, tier_orders)
    summary.to_csv(SUMMARY_CSV, index=False)
    print(f"[ok] wrote {len(summary)} rows -> {SUMMARY_CSV.name}")

    # Per-window results CSV: include both LR predictions + features + spo2 outcome
    out_df = feat_df.copy()
    out_df["label_pred_lr7"] = res_lr7["y_pred"]
    out_df["proba_excellent_lr7"] = res_lr7["proba_excellent"]
    out_df["label_pred_lr9"] = res_lr9["y_pred"]
    out_df["proba_excellent_lr9"] = res_lr9["proba_excellent"]
    out_df["label_pred_day5"] = day5_pred
    out_df["spo2_pred"] = spo2_df["spo2_pred"].to_numpy()
    out_df["spo2_valid"] = spo2_df["valid_flag"].to_numpy()
    out_df["spo2_ratio_r"] = spo2_df["ratio_r"].to_numpy()
    out_df["spo2_rejection_reason"] = spo2_df["rejection_reason"].to_numpy()
    out_df.to_csv(RESULTS_CSV, index=False)
    print(f"[ok] wrote {len(out_df)} rows -> {RESULTS_CSV.name}")

    # ---- Headline + verdict ----
    verdict = print_headline_table(summary, metrics_lr7, metrics_lr9)

    # ---- Figures ----
    make_figures(res_lr7, res_lr9, metrics_lr7, metrics_lr9, summary)

    # ---- Honest framing (with measured feature redundancy) ----
    feat_clean = feat_df[FEATURE_COLS_NEW].dropna()
    feat_corr = (
        float(feat_clean[FEATURE_COLS_NEW[0]].corr(feat_clean[FEATURE_COLS_NEW[1]]))
        if len(feat_clean) > 1 else float("nan")
    )
    print_honest_framing(feat_df, n_subjects_train, feat_corr)

    print()
    print("=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"LR-7 binary AUC : {metrics_lr7['auc']:.4f}    F1 : {metrics_lr7['f1_macro']:.4f}")
    print(f"LR-9 binary AUC : {metrics_lr9['auc']:.4f}    F1 : {metrics_lr9['f1_macro']:.4f}")
    print(f"Top 3 features (LR-9): "
          + ", ".join(f"{FEATURE_COLS[i]}={metrics_lr9['importance'][i]:.2f}"
                      for i in metrics_lr9["rank_idx"][:3]))
    print(f"Verdict          : {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
