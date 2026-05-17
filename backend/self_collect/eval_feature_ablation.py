"""
Experiment 2: Feature Ablation via Permutation Importance.

Goal: Identify low-signal features in current 38-feature set, drop them to reduce
overfitting on small N (17 self + 219 PPG-BP = 236 total).

Method:
  1. Train SVR/RF on full 38 features + subject-level GroupKFold (the LOSO protocol)
  2. Run sklearn.inspection.permutation_importance (n_repeats=30) to measure
     each feature's contribution to MAE
  3. Plot top-K importance bar chart per target (SBP, DBP)
  4. Sweep reduced-feature CV: K=10, 15, 20, 25, 30, 38 → MAE curve
  5. Report: which features to drop + expected MAE gain

Reference:
  - sklearn permutation_importance: model-agnostic, more reliable than RF built-in
    importance for small N (recommended by sklearn docs)
  - PMC12511243: feature reduction 57→15 via ReliefF improved BP MAE 6.97 mmHg

Usage:
  python eval_feature_ablation.py [--target sbp|dbp|both] [--n-repeats 30]
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ML_DIR = PROJECT_ROOT / "ML code" / "ml"
COLLECTED_DIR = PROJECT_ROOT / "collected_data"
OUTPUT_DIR = PROJECT_ROOT / "Thesis" / "figures"
RESULTS_DIR = Path(__file__).resolve().parent

os.environ.setdefault("MODEL_DIR", str(ML_DIR / "models"))
sys.path.insert(0, str(ML_DIR))

from train_ppg_bp import load_features_and_labels, load_labels  # noqa: E402

# Local utility — duplicate logic from eval_subject_holdout_calibration to keep self-contained
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_subject_holdout_calibration import load_self_sessions_separate  # noqa: E402

# 38-feature names (mapping from ml_models.py extract_*_features functions)
FEATURE_NAMES = [
    # Time domain (18)
    "td_mean", "td_std", "td_skewness", "td_kurtosis", "td_min", "td_max",
    "td_ptp", "td_rms", "td_avg_ppi", "td_std_ppi", "td_rmssd", "td_pnn50",
    "td_num_peaks", "td_peak_rate", "td_zero_cross_rate", "td_slope_mean",
    "td_slope_std", "td_energy",
    # Frequency (8)
    "fd_vlf", "fd_lf", "fd_hf", "fd_lf_hf_ratio", "fd_total_power",
    "fd_dominant_freq", "fd_spectral_entropy", "fd_spectral_centroid",
    # Morphological (12)
    "morph_systolic_amp_mean", "morph_systolic_amp_std",
    "morph_rise_time", "morph_fall_time",
    "morph_pulse_width_mean", "morph_pulse_width_std",
    "morph_area_ratio",
    "morph_deriv1_max", "morph_deriv1_min",
    "morph_deriv2_max", "morph_deriv2_min",
    "morph_waveform_symmetry",
]
assert len(FEATURE_NAMES) == 38, f"Expected 38 feature names, got {len(FEATURE_NAMES)}"


def build_pipeline() -> Pipeline:
    """Standard SVR pipeline used in train_ppg_bp.py"""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVR(kernel="rbf", C=10.0, epsilon=1.5, gamma="scale")),
    ])


def load_combined_data() -> Tuple[np.ndarray, Dict[str, np.ndarray], List[str]]:
    """
    Combine PPG-BP (transmission, 219 subjects) + self-collected (reflectance, 17
    sessions). Each session = one row (no aggregate mean per subject).

    Returns (X, y_dict, subject_ids):
      X: (N, 38) feature matrix
      y_dict: {"sbp": (N,), "dbp": (N,)}
      subject_ids: list[str], len N — for GroupKFold
    """
    # PPG-BP (aggregate=mean per subject, 219 rows)
    print("[1] Loading PPG-BP dataset...")
    labels_df = load_labels()
    X_ppgbp, y_ppgbp, sids_ppgbp = load_features_and_labels(labels_df, aggregate="mean")
    print(f"    PPG-BP: {X_ppgbp.shape[0]} subjects, {X_ppgbp.shape[1]} features")

    # Self-collected (each session separate row)
    print("[2] Loading self-collected sessions...")
    df_self = load_self_sessions_separate(COLLECTED_DIR)
    print(f"    Self-collected: {len(df_self)} sessions, "
          f"{df_self['subject_id'].nunique()} unique subjects")

    X_self = np.vstack(df_self["features"].to_list())
    y_sbp_self = df_self["sbp_baseline"].to_numpy()
    y_dbp_self = df_self["dbp_baseline"].to_numpy()
    sids_self = df_self["subject_id"].tolist()

    # Combine — prefix PPG-BP subject ids to disambiguate (avoid string collision với self)
    X_all = np.vstack([X_ppgbp, X_self])
    y_sbp_all = np.concatenate([y_ppgbp["sbp"], y_sbp_self])
    y_dbp_all = np.concatenate([y_ppgbp["dbp"], y_dbp_self])
    sids_all = [f"ppgbp_{int(s)}" for s in sids_ppgbp] + sids_self

    return X_all, {"sbp": y_sbp_all, "dbp": y_dbp_all}, sids_all


def aggregate_importance_across_folds(
    X: np.ndarray, y: np.ndarray, subject_ids: List[str],
    n_folds: int = 5, n_repeats: int = 30, random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Run permutation importance per fold + aggregate (mean across folds).

    Returns:
      importance_mean: (38,) average importance across folds
      importance_std:  (38,) std of importance across folds
      cv_mae: average test-fold MAE on full feature set
    """
    gkf = GroupKFold(n_splits=n_folds)
    importance_per_fold: List[np.ndarray] = []
    fold_maes: List[float] = []

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=subject_ids)):
        pipeline = build_pipeline()
        pipeline.fit(X[train_idx], y[train_idx])
        y_pred = pipeline.predict(X[test_idx])
        fold_mae = mean_absolute_error(y[test_idx], y_pred)
        fold_maes.append(fold_mae)

        # Permutation importance trên test fold
        result = permutation_importance(
            pipeline, X[test_idx], y[test_idx],
            n_repeats=n_repeats,
            random_state=random_state + fold_idx,
            scoring="neg_mean_absolute_error",
            n_jobs=1,  # sklearn permutation has its own parallelism issues with pipeline
        )
        # importances_mean is the mean reduction in score across n_repeats permutations
        # for permutation_importance with neg_mean_absolute_error: positive value = feature important
        importance_per_fold.append(result.importances_mean)
        print(f"    Fold {fold_idx+1}/{n_folds}: MAE={fold_mae:.2f}, "
              f"top features by importance: "
              f"{[FEATURE_NAMES[i] for i in np.argsort(result.importances_mean)[-3:][::-1]]}")

    importance_arr = np.array(importance_per_fold)  # (n_folds, 38)
    importance_mean = importance_arr.mean(axis=0)
    importance_std = importance_arr.std(axis=0)
    cv_mae = float(np.mean(fold_maes))

    return importance_mean, importance_std, cv_mae


def reduced_feature_cv(
    X: np.ndarray, y: np.ndarray, subject_ids: List[str],
    feature_indices: List[int], n_folds: int = 5,
) -> float:
    """Run subject-level CV using only specified feature indices. Returns mean MAE."""
    X_reduced = X[:, feature_indices]
    gkf = GroupKFold(n_splits=n_folds)
    fold_maes = []
    for train_idx, test_idx in gkf.split(X_reduced, y, groups=subject_ids):
        pipeline = build_pipeline()
        pipeline.fit(X_reduced[train_idx], y[train_idx])
        y_pred = pipeline.predict(X_reduced[test_idx])
        fold_maes.append(mean_absolute_error(y[test_idx], y_pred))
    return float(np.mean(fold_maes))


def plot_importance(
    feature_names: List[str], importance_mean: np.ndarray, importance_std: np.ndarray,
    target: str, output_path: Path, top_k: int = 38,
):
    """Plot horizontal bar chart of permutation importance, sorted descending."""
    sort_idx = np.argsort(importance_mean)[::-1][:top_k]
    names_sorted = [feature_names[i] for i in sort_idx]
    means_sorted = importance_mean[sort_idx]
    stds_sorted = importance_std[sort_idx]

    fig, ax = plt.subplots(figsize=(10, max(6, top_k * 0.25)), constrained_layout=True)
    colors = ["green" if m > 0 else "red" for m in means_sorted]
    y_pos = np.arange(len(names_sorted))
    ax.barh(y_pos, means_sorted, xerr=stds_sorted, color=colors, alpha=0.7, edgecolor="black")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names_sorted, fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color="black", lw=0.5)
    ax.set_xlabel(f"Permutation importance Δ MAE ({target}, mmHg)", fontsize=11)
    ax.set_title(
        f"{target.upper()} Feature Importance (permutation, mean ± std across 5 CV folds)\n"
        f"Positive = dropping hurts model. Negative = dropping helps (noise feature).",
        fontsize=11,
    )
    ax.text(
        0.5, -0.06,
        "Combined dataset: PPG-BP N=219 + self-collected N=15 (N_total=234), 5-fold subject-level GroupKFold CV, n_repeats=30",
        transform=ax.transAxes, ha="center", fontsize=9, color="gray", style="italic",
    )
    ax.grid(True, alpha=0.3, axis="x")

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


def plot_mae_curve(
    k_values: List[int], maes_sbp: List[float], maes_dbp: List[float],
    output_path: Path,
):
    """Plot MAE vs number of features kept, for SBP + DBP."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    panels = [
        (axes[0], "SBP", maes_sbp, "C0", "o", 12.93, 12.42),
        (axes[1], "DBP", maes_dbp, "C1", "s", 8.24, 6.69),
    ]
    for ax, label, maes, color, marker, naive, loso in panels:
        ax.plot(k_values, maes, f"{marker}-", label=f"{label} ablation MAE", color=color, lw=2)
        ax.axhline(
            naive, color="purple", linestyle="--", lw=1.5,
            label=f"Naive LOO baseline ({label}={naive:.2f})",
        )
        ax.axhline(
            loso, color="brown", linestyle="-.", lw=1.5,
            label=f"True LOSO MAE ({label}={loso:.2f}, N=15)",
        )
        ax.set_xlabel("Number of features kept (top-K by importance)", fontsize=11)
        ax.set_ylabel("CV MAE (mmHg)", fontsize=11)
        ax.set_title(f"{label} — Feature Reduction Impact on CV MAE",
                     fontsize=12, weight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc="best")
        for k, m in zip(k_values, maes):
            ax.annotate(f"{m:.2f}", (k, m), textcoords="offset points",
                        xytext=(0, 5), ha="center", fontsize=8)

    fig.suptitle(
        "Reference lines: Naive LOO baseline (eval_naive_baseline.py) | "
        "True LOSO MAE (eval_true_loso_report.csv)",
        fontsize=10, color="gray", style="italic",
    )
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


def report_drop_candidates(
    feature_names: List[str], importance_mean: np.ndarray, target: str
) -> List[Tuple[str, float]]:
    """Return list of (name, importance) sorted ascending — candidates to drop first."""
    sort_idx = np.argsort(importance_mean)
    drops = [(feature_names[i], importance_mean[i]) for i in sort_idx if importance_mean[i] <= 0.05]
    return drops


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=["sbp", "dbp", "both"], default="both")
    parser.add_argument("--n-repeats", type=int, default=30,
                        help="Permutation importance repeats per fold (default 30)")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip permutation importance computation; load cached "
                             "_feature_ablation_summary.csv and only re-render figures.")
    args = parser.parse_args()

    np.random.seed(args.seed)
    print(f"{'='*64}")
    print(f"Experiment 2: Feature Ablation via Permutation Importance")
    print(f"  n_repeats={args.n_repeats}, n_folds={args.n_folds}, seed={args.seed}")
    print(f"{'='*64}\n")

    X, y_dict, subject_ids = load_combined_data()
    print(f"\n  Combined: X.shape={X.shape}, "
          f"unique subjects={len(set(subject_ids))}\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    targets = ["sbp", "dbp"] if args.target == "both" else [args.target]
    all_results = {}

    cached_df = None
    if args.plot_only:
        cached_csv = RESULTS_DIR / "_feature_ablation_summary.csv"
        if not cached_csv.exists():
            print(f"ERROR: --plot-only requires {cached_csv.name} (run full mode first).")
            return 1
        cached_df = pd.read_csv(cached_csv)
        print(f"[plot-only] Loaded cached importance from {cached_csv.name}")

    for target in targets:
        if args.plot_only:
            print(f"\n[3] (plot-only) Loading cached importance for {target.upper()}...")
            y = y_dict[target]
            importance_mean = cached_df[f"importance_{target}"].to_numpy()
            importance_std = cached_df[f"importance_std_{target}"].to_numpy()
            # Recompute full-feature CV MAE (fast, no permutation) for reporting
            cv_mae = reduced_feature_cv(
                X, y, subject_ids, list(range(X.shape[1])), n_folds=args.n_folds,
            )
        else:
            print(f"\n[3] Computing permutation importance for {target.upper()}...")
            y = y_dict[target]
            importance_mean, importance_std, cv_mae = aggregate_importance_across_folds(
                X, y, subject_ids,
                n_folds=args.n_folds, n_repeats=args.n_repeats, random_state=args.seed,
            )
        print(f"    Full feature set CV MAE: {cv_mae:.2f} mmHg")

        # Save importance plot
        plot_path = OUTPUT_DIR / f"fig_feature_importance_{target}.pdf"
        plot_importance(FEATURE_NAMES, importance_mean, importance_std,
                        target, plot_path)

        # Reduced-feature CV sweep
        print(f"\n[4] Reduced-feature CV sweep for {target.upper()}...")
        sort_idx = np.argsort(importance_mean)[::-1]  # descending importance
        k_values = [10, 15, 20, 25, 30, 38]
        maes_k = []
        for k in k_values:
            top_k_indices = sort_idx[:k].tolist()
            mae_k = reduced_feature_cv(X, y, subject_ids, top_k_indices, n_folds=args.n_folds)
            maes_k.append(mae_k)
            print(f"    K={k}: MAE={mae_k:.2f} mmHg")

        # Find optimal K (lowest MAE)
        opt_k_idx = int(np.argmin(maes_k))
        opt_k = k_values[opt_k_idx]
        opt_mae = maes_k[opt_k_idx]
        print(f"    >> Optimal K={opt_k} -> MAE {opt_mae:.2f} mmHg "
              f"(vs full 38: {cv_mae:.2f}, delta {opt_mae - cv_mae:+.2f})")

        # Drop candidates
        drops = report_drop_candidates(FEATURE_NAMES, importance_mean, target)
        print(f"\n[5] Drop candidates (importance <= 0.05) for {target.upper()}:")
        for name, imp in drops[:10]:
            print(f"    {name}: {imp:+.4f}")

        all_results[target] = {
            "importance_mean": importance_mean.tolist(),
            "importance_std": importance_std.tolist(),
            "cv_mae_full": cv_mae,
            "k_values": k_values,
            "maes_k": maes_k,
            "optimal_k": opt_k,
            "optimal_mae": opt_mae,
            "drop_candidates": [(n, float(i)) for n, i in drops],
        }

    # Combined MAE curve plot
    if args.target == "both":
        plot_path = OUTPUT_DIR / "fig_feature_ablation_mae_curve.pdf"
        plot_mae_curve(
            all_results["sbp"]["k_values"],
            all_results["sbp"]["maes_k"],
            all_results["dbp"]["maes_k"],
            plot_path,
        )

    # Save summary CSV
    summary_rows = []
    for i, fname in enumerate(FEATURE_NAMES):
        row = {"feature": fname, "index": i}
        for target in targets:
            row[f"importance_{target}"] = all_results[target]["importance_mean"][i]
            row[f"importance_std_{target}"] = all_results[target]["importance_std"][i]
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = RESULTS_DIR / "_feature_ablation_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"\n[6] Saved feature importance CSV: {summary_csv.name}")

    # Final recommendation
    print(f"\n{'='*64}")
    print(f"=== FINAL RECOMMENDATION ===")
    print(f"{'='*64}")
    for target in targets:
        r = all_results[target]
        print(f"\n{target.upper()}:")
        print(f"  Full 38 features CV MAE: {r['cv_mae_full']:.2f} mmHg")
        print(f"  Optimal K={r['optimal_k']} CV MAE: {r['optimal_mae']:.2f} mmHg "
              f"(delta {r['optimal_mae'] - r['cv_mae_full']:+.2f})")
        if r["optimal_k"] < 38:
            n_drop = 38 - r["optimal_k"]
            print(f"  -> Drop bottom {n_drop} features -> reduced model")
        if r["drop_candidates"]:
            top_drops = [n for n, _ in r["drop_candidates"][:5]]
            print(f"  Top 5 drop candidates: {top_drops}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
