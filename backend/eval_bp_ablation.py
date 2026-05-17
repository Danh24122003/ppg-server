"""
 Deep SQA — BP ablation: BP estimation MAE stratified by per-person SQA score.

Question answered:
  Does per-person SQA quality (computed from  LR predictions across all
  windows of all sessions) predict BP estimation error under TRUE LOSO?
  If high-quality persons have lower MAE than low-quality persons -> SQA gating
  provides downstream BP value. If MAE is flat across strata -> SQA does not
  predict BP error (honest negative).

Methodology:
  1. From eval_sqa_classifier_results.csv (), filter to dataset == "BP-protocol".
     Group by subject_id; compute aggregate SQA quality:
       - pct_excellent: % windows labeled excellent by  LR
       - pct_unfit:     % windows labeled unfit
       - mean_proba_excellent: mean of prob_excellent
       - n_windows_total
  2. Run 15-fold LOSO (refit SVR per fold) on self-collected cohort.
     For each held-out person: train on PPG-BP 219 + 14 others, predict.
     This mirrors eval_true_loso.py exactly.
  3. Tag each held-out prediction with the person's SQA stratum:
       - Scheme A (tercile by pct_excellent): top-5 / mid-5 / bot-5
       - Scheme B (threshold pct_excellent >= 80%): high / low
  4. Aggregate MAE per stratum vs naive baselines (pop mean, LOO subject mean).

Excluded:
  - PPG-BP 219 subjects: no Red+IR raw, no SQA labels.
  - Persons with 0 windows in eval_sqa_classifier_results.csv (post-classifier
    files like S08_01v2): warn + assign stratum=NaN, kept in LOSO numerator
    only when both schemes resolve to a stratum.

Outputs:
  backend/eval_bp_ablation_results.csv
    Per-person: person, sbp_true, sbp_pred, sbp_err, dbp_true, dbp_pred, dbp_err,
    n_windows, pct_excellent, pct_unfit, mean_proba_excellent,
    stratum_tercile, stratum_threshold.

  backend/eval_bp_ablation_summary.csv
    Per (scheme x stratum): n_persons, sbp_mae, sbp_me, sbp_sd,
    dbp_mae, dbp_me, dbp_sd, naive_pop_sbp_mae, naive_pop_dbp_mae.

  backend/figures/fig_bp_ablation_mae_per_stratum.pdf
  backend/figures/fig_bp_ablation_quality_vs_error.pdf
  backend/figures/fig_bp_ablation_box_per_stratum.pdf

Usage:
  python "backend/eval_bp_ablation.py"

Time: ~3-5 min (15-fold SVR refit, identical to eval_true_loso.py).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

ROOT = Path(__file__).resolve().parent  # backend/
PROJECT_ROOT = ROOT.parent
ML_DIR = PROJECT_ROOT / "ML code" / "ml"
COLLECTED_DIR = PROJECT_ROOT / "collected_data"
sys.path.insert(0, str(ML_DIR))

from train_ppg_bp import (  # noqa: E402
    load_features_and_labels,
    load_labels,
    load_self_collected_features,
)

CLASSIFIER_CSV = ROOT / "eval_sqa_classifier_results.csv"
OUT_RESULTS_CSV = ROOT / "eval_bp_ablation_results.csv"
OUT_SUMMARY_CSV = ROOT / "eval_bp_ablation_summary.csv"
FIG_DIR = ROOT / "figures"

# Population mean baseline from PPG-BP training set (per eval_true_loso.py).
PPG_BP_MEAN_SBP = 118.3
PPG_BP_MEAN_DBP = 75.0

# Threshold scheme split point (per task spec).
PCT_EXCELLENT_THRESHOLD = 80.0

RANDOM_SEED = 42

STRATUM_COLORS = {
    "high": "#2ecc71",
    "medium": "#f1c40f",
    "low": "#e74c3c",
}


# ============================================================
# SQA per-person aggregation
# ============================================================
def compute_per_person_sqa(classifier_csv: Path) -> pd.DataFrame:
    """Aggregate  LR predictions per subject_id (BP-protocol only).

    Returns DataFrame indexed by subject_id with columns:
      n_windows_total, pct_excellent, pct_unfit, mean_proba_excellent.
    """
    if not classifier_csv.exists():
        raise FileNotFoundError(f"Missing {classifier_csv} -- run  first")

    df = pd.read_csv(classifier_csv)
    bp = df[df["dataset"] == "BP-protocol"].copy()
    if bp.empty:
        raise RuntimeError("No BP-protocol rows in classifier CSV")

    rows = []
    for sid, g in bp.groupby("subject_id"):
        n = len(g)
        n_exc = int((g["label_pred_lr"] == "excellent").sum())
        n_unf = int((g["label_pred_lr"] == "unfit").sum())
        rows.append({
            "subject_id": sid,
            "n_windows_total": n,
            "pct_excellent": 100.0 * n_exc / n if n else float("nan"),
            "pct_unfit": 100.0 * n_unf / n if n else float("nan"),
            "mean_proba_excellent": float(g["prob_excellent"].mean()),
        })
    out = pd.DataFrame(rows).set_index("subject_id")
    return out


def assign_strata(per_person: pd.DataFrame) -> pd.DataFrame:
    """Add stratum_tercile and stratum_threshold columns based on pct_excellent.

    Tercile: rank by pct_excellent, top third = high, mid third = medium,
             bottom third = low. Ties broken by ascending subject_id (stable).
    Threshold: pct_excellent >= 80 -> high, else low.
    NaN pct_excellent (no SQA windows) -> stratum = NaN.
    """
    df = per_person.copy()
    df["stratum_tercile"] = pd.NA
    df["stratum_threshold"] = pd.NA

    valid = df.dropna(subset=["pct_excellent"]).copy()
    # Tiebreak by subject_id ascending (stable across runs).
    valid = (valid.reset_index()
             .sort_values(["pct_excellent", "subject_id"], ascending=[False, True])
             .set_index("subject_id"))
    n_valid = len(valid)
    if n_valid == 0:
        return df

    # Tercile cuts: equal-size groups; if remainder, give to "medium" first.
    third = n_valid // 3
    rem = n_valid - 3 * third
    n_high = third + (1 if rem >= 2 else 0)
    n_mid = third + (1 if rem >= 1 else 0)
    # n_low = n_valid - n_high - n_mid
    sids = valid.index.tolist()
    high_ids = set(sids[:n_high])
    mid_ids = set(sids[n_high:n_high + n_mid])
    # remaining = low
    for sid in df.index:
        pe = df.at[sid, "pct_excellent"]
        if pd.isna(pe):
            continue
        if sid in high_ids:
            df.at[sid, "stratum_tercile"] = "high"
        elif sid in mid_ids:
            df.at[sid, "stratum_tercile"] = "medium"
        else:
            df.at[sid, "stratum_tercile"] = "low"
        df.at[sid, "stratum_threshold"] = "high" if pe >= PCT_EXCELLENT_THRESHOLD else "low"
    return df


# ============================================================
# LOSO BP estimation (mirrors eval_true_loso.py)
# ============================================================
def build_svr_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svr", SVR(kernel="rbf", C=10.0, gamma="scale")),
    ])


def run_loso(
    X_bp: np.ndarray,
    y_bp: Dict[str, np.ndarray],
    X_sc: np.ndarray,
    y_sc: Dict[str, np.ndarray],
    sid_sc: List[str],
) -> pd.DataFrame:
    """15-fold LOSO. Returns DataFrame with per-person rows."""
    n = len(sid_sc)
    rows = []
    for i, person in enumerate(sid_sc):
        keep = np.array([j != i for j in range(n)])
        X_train = np.vstack([X_bp, X_sc[keep]])
        y_sbp_train = np.concatenate([y_bp["sbp"], y_sc["sbp"][keep]])
        y_dbp_train = np.concatenate([y_bp["dbp"], y_sc["dbp"][keep]])

        pipe_sbp = build_svr_pipeline().fit(X_train, y_sbp_train)
        pipe_dbp = build_svr_pipeline().fit(X_train, y_dbp_train)

        X_held = X_sc[i:i + 1]
        sbp_pred = float(pipe_sbp.predict(X_held)[0])
        dbp_pred = float(pipe_dbp.predict(X_held)[0])
        sbp_true = float(y_sc["sbp"][i])
        dbp_true = float(y_sc["dbp"][i])
        rows.append({
            "person": person,
            "n_train": int(len(X_train)),
            "sbp_true": sbp_true,
            "sbp_pred": sbp_pred,
            "sbp_err": sbp_pred - sbp_true,
            "dbp_true": dbp_true,
            "dbp_pred": dbp_pred,
            "dbp_err": dbp_pred - dbp_true,
        })
        print(f"      [{i+1:2d}/{n}] {person:10s} | "
              f"SBP {sbp_true:5.1f} -> {sbp_pred:6.2f} (err {sbp_pred-sbp_true:+6.2f}) | "
              f"DBP {dbp_true:4.1f} -> {dbp_pred:5.2f} (err {dbp_pred-dbp_true:+5.2f})")
    return pd.DataFrame(rows)


# ============================================================
# Aggregation per stratum
# ============================================================
def aggregate_per_stratum(
    detail: pd.DataFrame,
    scheme_col: str,
    scheme_name: str,
    confounders: Dict[str, float] | None = None,
    degenerate_note: str | None = None,
) -> pd.DataFrame:
    """Compute per-stratum aggregate metrics for a given scheme column.

    confounders: optional dict with keys r_sbp_confounder, r_dbp_confounder,
                 r_quality_sbp, r_quality_dbp -- written to the ALL row.
    degenerate_note: optional string -- written to all rows in this scheme
                     (e.g. when threshold "low" stratum is empty).
    """
    rows = []
    naive_pop_sbp_mae = float((detail["sbp_true"] - PPG_BP_MEAN_SBP).abs().mean())
    naive_pop_dbp_mae = float((detail["dbp_true"] - PPG_BP_MEAN_DBP).abs().mean())

    sub = detail.dropna(subset=[scheme_col])
    for stratum, g in sub.groupby(scheme_col):
        sbp_abs = g["sbp_err"].abs()
        dbp_abs = g["dbp_err"].abs()
        np_sbp = float((g["sbp_true"] - PPG_BP_MEAN_SBP).abs().mean())
        np_dbp = float((g["dbp_true"] - PPG_BP_MEAN_DBP).abs().mean())
        rows.append({
            "scheme": scheme_name,
            "stratum": stratum,
            "n_persons": int(len(g)),
            "sbp_mae": float(sbp_abs.mean()),
            "sbp_me": float(g["sbp_err"].mean()),
            "sbp_sd": float(g["sbp_err"].std(ddof=1)) if len(g) > 1 else float("nan"),
            "dbp_mae": float(dbp_abs.mean()),
            "dbp_me": float(g["dbp_err"].mean()),
            "dbp_sd": float(g["dbp_err"].std(ddof=1)) if len(g) > 1 else float("nan"),
            "naive_pop_sbp_mae": np_sbp,
            "naive_pop_dbp_mae": np_dbp,
            "r_sbp_confounder": float("nan"),
            "r_dbp_confounder": float("nan"),
            "r_quality_sbp": float("nan"),
            "r_quality_dbp": float("nan"),
            "degenerate_note": degenerate_note if degenerate_note else "",
        })
    # Always-on row across all persons (any stratum).
    sbp_abs = detail["sbp_err"].abs()
    dbp_abs = detail["dbp_err"].abs()
    all_row = {
        "scheme": scheme_name,
        "stratum": "ALL",
        "n_persons": int(len(detail)),
        "sbp_mae": float(sbp_abs.mean()),
        "sbp_me": float(detail["sbp_err"].mean()),
        "sbp_sd": float(detail["sbp_err"].std(ddof=1)) if len(detail) > 1 else float("nan"),
        "dbp_mae": float(dbp_abs.mean()),
        "dbp_me": float(detail["dbp_err"].mean()),
        "dbp_sd": float(detail["dbp_err"].std(ddof=1)) if len(detail) > 1 else float("nan"),
        "naive_pop_sbp_mae": naive_pop_sbp_mae,
        "naive_pop_dbp_mae": naive_pop_dbp_mae,
        "r_sbp_confounder": float(confounders["r_sbp_confounder"]) if confounders else float("nan"),
        "r_dbp_confounder": float(confounders["r_dbp_confounder"]) if confounders else float("nan"),
        "r_quality_sbp": float(confounders["r_quality_sbp"]) if confounders else float("nan"),
        "r_quality_dbp": float(confounders["r_quality_dbp"]) if confounders else float("nan"),
        "degenerate_note": degenerate_note if degenerate_note else "",
    }
    rows.append(all_row)
    return pd.DataFrame(rows)


# ============================================================
# Plotting
# ============================================================
def plot_mae_per_stratum(summary: pd.DataFrame, out: Path) -> None:
    """Bar chart 2 panels (SBP, DBP) x stratum, separated by scheme."""
    schemes = ["tercile", "threshold"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for col, target in enumerate(["sbp", "dbp"]):
        for row, scheme in enumerate(schemes):
            ax = axes[row, col]
            s = summary[(summary["scheme"] == scheme) & (summary["stratum"] != "ALL")].copy()
            order = ["high", "medium", "low"] if scheme == "tercile" else ["high", "low"]
            present = [x for x in order if x in set(s["stratum"].tolist())]
            s = s.set_index("stratum").reindex(present).reset_index()
            s = s.dropna(subset=[f"{target}_mae"])
            colors = [STRATUM_COLORS.get(x, "#888888") for x in s["stratum"]]
            bars = ax.bar(s["stratum"], s[f"{target}_mae"], color=colors, edgecolor="black")
            for b, n in zip(bars, s["n_persons"]):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.2,
                        f"N={int(n)}", ha="center", va="bottom", fontsize=9)
            naive = float(s[f"naive_pop_{target}_mae"].iloc[0]) if len(s) else float("nan")
            if np.isfinite(naive):
                ax.axhline(naive, color="black", linestyle="--", linewidth=1,
                           label=f"naive pop = {naive:.2f}")
                ax.legend(loc="upper right", fontsize=8)
            ax.set_title(f"{target.upper()} MAE per {scheme} stratum")
            ax.set_xlabel("stratum (by pct_excellent)")
            ax.set_ylabel("MAE (mmHg)")
            ax.grid(axis="y", alpha=0.3)
    fig.suptitle(" BP ablation -- MAE per SQA quality stratum (LOSO N=15)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out)
    plt.close(fig)
    print(f"      [saved] {out.relative_to(PROJECT_ROOT)}")


def plot_quality_vs_error(detail: pd.DataFrame, out: Path) -> None:
    """Scatter pct_excellent vs |err|, separate panels SBP/DBP."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, target in zip(axes, ["sbp", "dbp"]):
        d = detail.dropna(subset=["pct_excellent"]).copy()
        d["abs_err"] = d[f"{target}_err"].abs()
        ax.scatter(d["pct_excellent"], d["abs_err"], s=60, color="#3498db",
                   edgecolor="black", alpha=0.85)
        for _, r in d.iterrows():
            ax.annotate(r["person"], (r["pct_excellent"], r["abs_err"]),
                        fontsize=7, xytext=(3, 3), textcoords="offset points")
        # Linear fit if >= 3 points.
        if len(d) >= 3:
            x = d["pct_excellent"].to_numpy()
            y = d["abs_err"].to_numpy()
            slope, intercept = np.polyfit(x, y, 1)
            xs = np.linspace(x.min(), x.max(), 50)
            ax.plot(xs, slope * xs + intercept, color="#e74c3c", linewidth=1,
                    label=f"slope={slope:+.3f}")
            ax.legend(loc="upper right", fontsize=9)
        ax.set_xlabel("pct_excellent (% LR-classified excellent windows)")
        ax.set_ylabel(f"|{target}_err| (mmHg)")
        ax.set_title(f"{target.upper()} -- per-person SQA quality vs |err|")
        ax.grid(alpha=0.3)
    fig.suptitle(" BP ablation -- SQA quality vs LOSO error (per person)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out)
    plt.close(fig)
    print(f"      [saved] {out.relative_to(PROJECT_ROOT)}")


def plot_box_per_stratum(detail: pd.DataFrame, out: Path) -> None:
    """Box plot |err| per tercile stratum, side by side SBP+DBP."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    order = ["high", "medium", "low"]
    d = detail.dropna(subset=["stratum_tercile"]).copy()
    for ax, target in zip(axes, ["sbp", "dbp"]):
        data = []
        labels = []
        for s in order:
            sub = d[d["stratum_tercile"] == s][f"{target}_err"].abs().to_numpy()
            if sub.size > 0:
                data.append(sub)
                labels.append(f"{s}\n(N={sub.size})")
        if data:
            bp = ax.boxplot(data, tick_labels=labels, patch_artist=True,
                            showmeans=True, meanprops={"marker": "D", "markerfacecolor": "red"})
            for patch, s in zip(bp["boxes"], order[:len(data)]):
                patch.set_facecolor(STRATUM_COLORS.get(s, "#888888"))
                patch.set_alpha(0.6)
        ax.set_ylabel(f"|{target}_err| (mmHg)")
        ax.set_title(f"{target.upper()} -- |err| per tercile stratum")
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(" BP ablation -- |err| distribution per stratum (tercile)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out)
    plt.close(fig)
    print(f"      [saved] {out.relative_to(PROJECT_ROOT)}")


# ============================================================
# Main
# ============================================================
def main() -> int:
    np.random.seed(RANDOM_SEED)
    print("=" * 78)
    print(" BP ablation -- LOSO MAE stratified by per-person SQA score")
    print("=" * 78)

    # ----- Step 1: per-person SQA aggregation -----
    print("\n[1/5] Aggregating  LR predictions per subject_id (BP-protocol only)...")
    per_person = compute_per_person_sqa(CLASSIFIER_CSV)
    print(f"      {len(per_person)} subject_ids in classifier CSV (BP-protocol)")
    print(per_person.sort_values("pct_excellent", ascending=False).to_string(
        float_format=lambda x: f"{x:.2f}"))

    # ----- Step 2: load features -----
    print("\n[2/5] Loading PPG-BP labels + features (219 subjects)...")
    labels_df = load_labels()
    X_bp, y_bp, _ = load_features_and_labels(labels_df, aggregate="mean")
    print(f"      X_bp.shape = {X_bp.shape}")

    print("\n[3/5] Loading self-collected features (per person aggregation)...")
    X_sc, y_sc, sid_sc = load_self_collected_features(COLLECTED_DIR)
    n_persons = len(sid_sc)
    print(f"      X_sc.shape = {X_sc.shape}, persons = {sid_sc}")

    if n_persons < 3:
        print("ERROR: need >= 3 persons for LOSO")
        return 1

    # Filter LOSO cohort to persons with SQA windows (per task spec: warn + exclude
    # from ablation when a person has 0 windows in eval_sqa_classifier_results.csv).
    sqa_ids = set(per_person.index.tolist())
    missing = [p for p in sid_sc if p not in sqa_ids]
    if missing:
        print(f"      WARN: {len(missing)} self-collected person(s) missing from SQA "
              f"classifier CSV: {missing}")
        print(f"            -> excluded from ablation (no per-person SQA score available)")
        keep_idx = [i for i, p in enumerate(sid_sc) if p in sqa_ids]
        sid_sc = [sid_sc[i] for i in keep_idx]
        X_sc = X_sc[keep_idx]
        y_sc = {"sbp": y_sc["sbp"][keep_idx], "dbp": y_sc["dbp"][keep_idx]}
        n_persons = len(sid_sc)
        print(f"            -> ablation cohort: N = {n_persons} persons")

    # ----- Step 3: LOSO -----
    print(f"\n[4/5] Running {n_persons}-fold LOSO (refit SVR each fold)...")
    detail = run_loso(X_bp, y_bp, X_sc, y_sc, sid_sc)

    # ----- Step 4: stratify -----
    strata = assign_strata(per_person)

    # Join SQA scores + strata onto LOSO detail.
    detail = detail.merge(
        strata.reset_index().rename(columns={"subject_id": "person"}),
        on="person",
        how="left",
    )

    # Save per-person detail.
    cols = ["person", "sbp_true", "sbp_pred", "sbp_err",
            "dbp_true", "dbp_pred", "dbp_err",
            "n_windows_total", "pct_excellent", "pct_unfit", "mean_proba_excellent",
            "stratum_tercile", "stratum_threshold"]
    detail_out = detail.reindex(columns=cols).rename(columns={"n_windows_total": "n_windows"})
    detail_out.to_csv(OUT_RESULTS_CSV, index=False)
    print(f"      [saved] {OUT_RESULTS_CSV.relative_to(PROJECT_ROOT)}")

    # ----- Confounder analysis (Fix 1 P1) -----
    detail_with_features = detail_out.dropna(subset=["pct_excellent"]).copy()
    detail_with_features["sbp_bp_dist"] = (detail_with_features["sbp_true"] - PPG_BP_MEAN_SBP).abs()
    detail_with_features["dbp_bp_dist"] = (detail_with_features["dbp_true"] - PPG_BP_MEAN_DBP).abs()
    detail_with_features["sbp_err_abs"] = detail_with_features["sbp_err"].abs()
    detail_with_features["dbp_err_abs"] = detail_with_features["dbp_err"].abs()

    r_sbp_confounder = float(np.corrcoef(
        detail_with_features["sbp_bp_dist"], detail_with_features["sbp_err_abs"])[0, 1])
    r_dbp_confounder = float(np.corrcoef(
        detail_with_features["dbp_bp_dist"], detail_with_features["dbp_err_abs"])[0, 1])
    r_quality_sbp = float(np.corrcoef(
        detail_with_features["pct_excellent"], detail_with_features["sbp_err_abs"])[0, 1])
    r_quality_dbp = float(np.corrcoef(
        detail_with_features["pct_excellent"], detail_with_features["dbp_err_abs"])[0, 1])

    confounders = {
        "r_sbp_confounder": r_sbp_confounder,
        "r_dbp_confounder": r_dbp_confounder,
        "r_quality_sbp": r_quality_sbp,
        "r_quality_dbp": r_quality_dbp,
    }

    # ----- Threshold scheme degenerate check (Fix 2 P1) -----
    threshold_low_count = int((detail_out["stratum_threshold"] == "low").sum())
    threshold_degenerate_note: str | None = None
    if threshold_low_count == 0:
        threshold_degenerate_note = "all persons >= 80% threshold, scheme degenerate"

    # ----- Step 5: per-stratum aggregation -----
    print("\n[5/5] Per-stratum aggregation + figures...")
    summary_tercile = aggregate_per_stratum(
        detail, "stratum_tercile", "tercile", confounders=confounders)
    summary_threshold = aggregate_per_stratum(
        detail, "stratum_threshold", "threshold",
        confounders=confounders, degenerate_note=threshold_degenerate_note)
    summary = pd.concat([summary_tercile, summary_threshold], ignore_index=True)
    summary.to_csv(OUT_SUMMARY_CSV, index=False)
    print(f"      [saved] {OUT_SUMMARY_CSV.relative_to(PROJECT_ROOT)}")

    # ----- Figures -----
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_mae_per_stratum(summary, FIG_DIR / "fig_bp_ablation_mae_per_stratum.pdf")
    plot_quality_vs_error(detail_out, FIG_DIR / "fig_bp_ablation_quality_vs_error.pdf")
    plot_box_per_stratum(detail_out, FIG_DIR / "fig_bp_ablation_box_per_stratum.pdf")

    # ----- Console headline -----
    print()
    print("=" * 78)
    print(f"HEADLINE -- BP MAE per SQA stratum (LOSO N={n_persons}, refit SVR per fold)")
    print("=" * 78)
    print()
    naive_pop_sbp = float(summary[summary["stratum"] == "ALL"]["naive_pop_sbp_mae"].iloc[0])
    naive_pop_dbp = float(summary[summary["stratum"] == "ALL"]["naive_pop_dbp_mae"].iloc[0])
    header = (f"{'Scheme':9s} | {'Stratum':7s} | {'N':>2s} | {'SBP MAE':>8s} | "
              f"{'DBP MAE':>8s} | {'Naive SBP':>9s} | {'delta vs naive':>14s}")
    print(header)
    print("-" * len(header))
    for _, r in summary.iterrows():
        if r["stratum"] == "ALL":
            continue
        delta = naive_pop_sbp - r["sbp_mae"]
        print(f"{r['scheme']:9s} | {r['stratum']:7s} | {int(r['n_persons']):>2d} | "
              f"{r['sbp_mae']:8.2f} | {r['dbp_mae']:8.2f} | "
              f"{r['naive_pop_sbp_mae']:9.2f} | {delta:+13.2f}")

    # All-row summary for reference.
    all_sbp = float(summary[(summary["scheme"] == "tercile") &
                            (summary["stratum"] == "ALL")]["sbp_mae"].iloc[0])
    all_dbp = float(summary[(summary["scheme"] == "tercile") &
                            (summary["stratum"] == "ALL")]["dbp_mae"].iloc[0])
    print()
    print(f"  Across all N={n_persons}: SBP MAE = {all_sbp:.2f} | DBP MAE = {all_dbp:.2f}")
    print(f"  Naive pop mean        : SBP MAE = {naive_pop_sbp:.2f} | DBP MAE = {naive_pop_dbp:.2f}")

    # ----- Threshold degenerate warning (Fix 2 P1) -----
    if threshold_degenerate_note is not None:
        min_pe = float(detail_out["pct_excellent"].min())
        min_person = detail_out.loc[detail_out["pct_excellent"].idxmin(), "person"]
        print()
        print("=" * 78)
        print("WARNING: Threshold scheme DEGENERATE")
        print("=" * 78)
        print(f"All {len(detail_out)} persons have pct_excellent >= 80% (lowest = "
              f"{min_pe:.1f}%, person={min_person}).")
        print("Threshold at 80% does not separate this controlled-resting cohort.")
        print("Scheme B numbers in summary CSV are NOT meaningful for ablation.")
        print("Use tercile scheme as primary stratification.")

    # ----- Confounder analysis (Fix 1 P1) -----
    print()
    print("=" * 78)
    print("CONFOUNDER ANALYSIS")
    print("=" * 78)
    print(f"\nCorrelation r(|BP_true - 118.3|, |sbp_err|) = {r_sbp_confounder:.3f}")
    print(f"Correlation r(|BP_true - 75.0 |, |dbp_err|) = {r_dbp_confounder:.3f}")
    print(f"Correlation r(pct_excellent,    |sbp_err|) = {r_quality_sbp:+.3f}")
    print(f"Correlation r(pct_excellent,    |dbp_err|) = {r_quality_dbp:+.3f}")
    print()
    if r_sbp_confounder > 0.7:
        print(f"-> STRONG confounder: BP-distance-from-mean dominates SBP error (r > 0.7)")
    elif r_sbp_confounder > 0.5:
        print(f"-> MODERATE confounder: BP-distance-from-mean partially explains SBP error (0.5 < r < 0.7)")
    else:
        print(f"-> WEAK confounder: BP-distance-from-mean does NOT explain SBP error (r < 0.5)")
    if abs(r_quality_sbp) < 0.3:
        print(f"-> SQA quality has NO predictive power for SBP error (|r| < 0.3)")

    # ----- Verdict -----
    print()
    print("=" * 78)
    print(f"VERDICT (directional, N={n_persons} underpowered)")
    print("=" * 78)
    terc = summary[(summary["scheme"] == "tercile") & (summary["stratum"] != "ALL")].set_index("stratum")
    if {"high", "low"}.issubset(set(terc.index)):
        sbp_h = float(terc.loc["high", "sbp_mae"])
        sbp_l = float(terc.loc["low", "sbp_mae"])
        dbp_h = float(terc.loc["high", "dbp_mae"])
        dbp_l = float(terc.loc["low", "dbp_mae"])
        print(f"  SBP: tercile high MAE={sbp_h:.2f} vs low MAE={sbp_l:.2f} "
              f"(delta = {sbp_h - sbp_l:+.2f} mmHg)")
        if sbp_h < sbp_l:
            print(f"       -> high-quality persons have LOWER SBP MAE -- SQA gating helps")
        else:
            print(f"       -> high-quality persons NOT lower MAE -- SQA does not predict SBP error")
        print(f"  DBP: tercile high MAE={dbp_h:.2f} vs low MAE={dbp_l:.2f} "
              f"(delta = {dbp_h - dbp_l:+.2f} mmHg)")
        if dbp_h < dbp_l:
            print(f"       -> high-quality persons have LOWER DBP MAE -- SQA gating helps")
        else:
            print(f"       -> high-quality persons NOT lower MAE -- SQA does not predict DBP error")

    # ----- Honest framing -----
    print()
    print("=" * 78)
    print("HONEST FRAMING (cite alongside numbers in thesis)")
    print("=" * 78)
    print(
        f"1. Sample size N={n_persons} unique persons (15 LOSO cohort minus {len(missing)}\n"
        f"   missing from SQA classifier CSV: {missing if missing else 'none'}).\n"
        "   Tercile split unbalanced when ties exist at boundary.\n"
        "   Statistical power LOW; finding directional, not powered.\n"
        "2. SQA quality score is per-person aggregate (across all sessions x all windows).\n"
        "   Multi-session subjects: S03, S07, S10, S09.\n"
        "3. PPG-BP 219 subjects do NOT have SQA labels (no Red+IR raw).\n"
        f"   Only self-collected {n_persons} persons stratified.\n"
        "4. LOSO per-person held-out -- leakage-free per Cheung 2024 IEEE TBME.\n"
        "5. SVR refit per fold (~3-5 min total run time). Random seed 42.\n"
        "6. Naive baselines: pop mean (118.3/75.0) + LOO subject mean.\n"
        "   If SBP MAE high stratum < pop mean MAE -> SQA-gating provides value.\n"
        "   If high stratum MAE ~= low stratum MAE -> SQA does not predict BP error."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
