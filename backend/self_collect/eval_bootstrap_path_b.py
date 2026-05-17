"""
Bootstrap paired test (Experiment A) + Path B baseline (Experiment B).

Eval-only: reads existing LOSO + drift CSVs, performs bootstrap CI / paired
comparison and Path B (drop pop model) ablation. No production code modified.

Inputs:
  - eval_true_loso_report.csv         (15 LOSO per-subject results)
  - _drift_results_sbp.csv             (drift sessions SBP)
  - _drift_results_dbp.csv             (drift sessions DBP)

Outputs:
  - eval_bootstrap_path_b_results.csv  (per-row + summary CSV)
  - stdout report (ASCII safe for Windows CP1252)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).parent
LOSO_PATH = HERE / "eval_true_loso_report.csv"
DRIFT_SBP_PATH = HERE / "_drift_results_sbp.csv"
DRIFT_DBP_PATH = HERE / "_drift_results_dbp.csv"
OUTPUT_PATH = HERE / "eval_bootstrap_path_b_results.csv"

N_BOOTSTRAP = 10_000
SEED = 42

# Expected reference numbers from the project guide (sanity check)
# Updated 2026-05-10 đêm post epsilon correction sweep (Section II): SVR(epsilon=1.5) canonical
# Old (epsilon=0.1 default): sbp_loso 14.11 / dbp_loso 7.18 / hybrid_sbp 8.61 / hybrid_dbp 7.26 — superseded
EXPECTED = {
    "sbp_loso_mae": 12.42,
    "dbp_loso_mae": 6.69,
    "naive_sbp_mae": 12.93,
    "naive_dbp_mae": 8.24,
    "hybrid_sbp_mae": 9.39,
    "hybrid_dbp_mae": 6.80,
}


def _check_files() -> None:
    for p in (LOSO_PATH, DRIFT_SBP_PATH, DRIFT_DBP_PATH):
        if not p.exists():
            print(f"[ERROR] missing input file: {p}")
            sys.exit(1)


def _format_pass(actual: float, expected: float, tol: float = 0.05) -> str:
    return "PASS" if abs(actual - expected) <= tol else "FAIL"


def _loo_subject_mean(truth: np.ndarray) -> np.ndarray:
    """For each i, return mean(truth[j] for j != i)."""
    n = len(truth)
    total = truth.sum()
    return (total - truth) / (n - 1)


def _bootstrap_paired(
    err_model: np.ndarray, err_naive: np.ndarray, n_boot: int, seed: int
) -> tuple[float, float, float, float, np.ndarray]:
    """
    Returns (delta_mean, ci_lo, ci_hi, p_one_sided, delta_distribution).
    delta = MAE_naive - MAE_model  (positive => model better than naive).
    p_one_sided = fraction of bootstrap deltas <= 0 (model NOT better).
    """
    rng = np.random.default_rng(seed)
    n = len(err_model)
    deltas = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        mae_m = err_model[idx].mean()
        mae_n = err_naive[idx].mean()
        deltas[b] = mae_n - mae_m
    delta_mean = float(deltas.mean())
    ci_lo = float(np.percentile(deltas, 2.5))
    ci_hi = float(np.percentile(deltas, 97.5))
    p_one = float((deltas <= 0).mean())
    return delta_mean, ci_lo, ci_hi, p_one, deltas


def experiment_a() -> tuple[dict, pd.DataFrame]:
    """Bootstrap paired test on LOSO file (model vs naive LOO subject-mean)."""
    df = pd.read_csv(LOSO_PATH)
    n = len(df)

    sbp_true = df["sbp_true"].to_numpy(dtype=float)
    sbp_pred = df["sbp_pred"].to_numpy(dtype=float)
    dbp_true = df["dbp_true"].to_numpy(dtype=float)
    dbp_pred = df["dbp_pred"].to_numpy(dtype=float)

    sbp_naive = _loo_subject_mean(sbp_true)
    dbp_naive = _loo_subject_mean(dbp_true)

    err_sbp_model = np.abs(sbp_true - sbp_pred)
    err_sbp_naive = np.abs(sbp_true - sbp_naive)
    err_dbp_model = np.abs(dbp_true - dbp_pred)
    err_dbp_naive = np.abs(dbp_true - dbp_naive)

    mae_sbp_model = float(err_sbp_model.mean())
    mae_sbp_naive = float(err_sbp_naive.mean())
    mae_dbp_model = float(err_dbp_model.mean())
    mae_dbp_naive = float(err_dbp_naive.mean())

    sbp_delta, sbp_lo, sbp_hi, sbp_p, _ = _bootstrap_paired(
        err_sbp_model, err_sbp_naive, N_BOOTSTRAP, SEED
    )
    dbp_delta, dbp_lo, dbp_hi, dbp_p, _ = _bootstrap_paired(
        err_dbp_model, err_dbp_naive, N_BOOTSTRAP, SEED
    )

    summary = {
        "n_subjects": n,
        "mae_sbp_model": mae_sbp_model,
        "mae_sbp_naive": mae_sbp_naive,
        "mae_dbp_model": mae_dbp_model,
        "mae_dbp_naive": mae_dbp_naive,
        "sbp_delta_mean": sbp_delta,
        "sbp_ci_lo": sbp_lo,
        "sbp_ci_hi": sbp_hi,
        "sbp_p_one_sided": sbp_p,
        "dbp_delta_mean": dbp_delta,
        "dbp_ci_lo": dbp_lo,
        "dbp_ci_hi": dbp_hi,
        "dbp_p_one_sided": dbp_p,
    }

    rows = pd.DataFrame(
        {
            "experiment": "A_bootstrap",
            "subject": df["person"].to_numpy(),
            "sbp_true": sbp_true,
            "sbp_model_pred": sbp_pred,
            "sbp_naive_pred": sbp_naive,
            "sbp_err_model": err_sbp_model,
            "sbp_err_naive": err_sbp_naive,
            "dbp_true": dbp_true,
            "dbp_model_pred": dbp_pred,
            "dbp_naive_pred": dbp_naive,
            "dbp_err_model": err_dbp_model,
            "dbp_err_naive": err_dbp_naive,
        }
    )
    return summary, rows


def _path_b_for_metric(drift_df: pd.DataFrame, metric_label: str) -> tuple[dict, pd.DataFrame]:
    """
    Compare hybrid (pred_calibrated) vs Path B (cuff_anchor) on genuine drift
    sessions (included_in_drift_eval=True AND days_from_anchor>0).
    """
    # anchors (one row per subject with is_anchor=True)
    anchors = drift_df[drift_df["is_anchor"]].copy()
    anchor_map = dict(zip(anchors["subject_id"], anchors["cuff_truth"].astype(float)))

    # genuine drift = flagged AND days>0
    drift = drift_df[
        (drift_df["included_in_drift_eval"]) & (drift_df["days_from_anchor"] > 0)
    ].copy()

    if len(drift) == 0:
        return (
            {
                "metric": metric_label,
                "n_genuine_drift": 0,
                "mae_hybrid": float("nan"),
                "mae_path_b": float("nan"),
                "delta_hybrid_minus_path_b": float("nan"),
            },
            pd.DataFrame(),
        )

    drift["cuff_anchor"] = drift["subject_id"].map(anchor_map)
    if drift["cuff_anchor"].isna().any():
        missing = drift.loc[drift["cuff_anchor"].isna(), "subject_id"].unique().tolist()
        print(f"[WARN] {metric_label}: drift subjects without anchor: {missing}")
        drift = drift.dropna(subset=["cuff_anchor"])

    drift["hybrid_pred"] = drift["pred_calibrated"].astype(float)
    drift["path_b_pred"] = drift["cuff_anchor"].astype(float)
    drift["hybrid_err"] = drift["abs_err_calibrated"].astype(float)
    drift["path_b_err"] = (drift["cuff_truth"].astype(float) - drift["path_b_pred"]).abs()
    drift["diff_err"] = drift["hybrid_err"] - drift["path_b_err"]

    mae_hybrid = float(drift["hybrid_err"].mean())
    mae_path_b = float(drift["path_b_err"].mean())

    # Bootstrap paired test: hybrid vs path_b on same drift sessions.
    # Reuse _bootstrap_paired with (err_path_b, err_hybrid) so that
    # delta = MAE_hybrid - MAE_path_b (positive => hybrid worse, path_b better).
    # p_one_sided = fraction of bootstrap deltas <= 0 = fraction where path_b NOT better.
    hybrid_errs = drift["hybrid_err"].to_numpy(dtype=float)
    path_b_errs = drift["path_b_err"].to_numpy(dtype=float)
    if len(hybrid_errs) >= 2:
        b_delta, b_lo, b_hi, b_p, _ = _bootstrap_paired(
            path_b_errs, hybrid_errs, N_BOOTSTRAP, SEED
        )
    else:
        b_delta, b_lo, b_hi, b_p = float("nan"), float("nan"), float("nan"), float("nan")

    summary = {
        "metric": metric_label,
        "n_genuine_drift": int(len(drift)),
        "mae_hybrid": mae_hybrid,
        "mae_path_b": mae_path_b,
        "delta_hybrid_minus_path_b": mae_hybrid - mae_path_b,
        "boot_delta_mean": b_delta,
        "boot_ci_lo": b_lo,
        "boot_ci_hi": b_hi,
        "boot_p_one_sided": b_p,
    }

    rows = pd.DataFrame(
        {
            "experiment": f"B_path_b_{metric_label}",
            "subject": drift["subject_id"].to_numpy(),
            "days_from_anchor": drift["days_from_anchor"].astype(float).to_numpy(),
            "cuff_truth": drift["cuff_truth"].astype(float).to_numpy(),
            "hybrid_pred": drift["hybrid_pred"].to_numpy(),
            "hybrid_err": drift["hybrid_err"].to_numpy(),
            "path_b_pred": drift["path_b_pred"].to_numpy(),
            "path_b_err": drift["path_b_err"].to_numpy(),
            "diff_hybrid_minus_path_b": drift["diff_err"].to_numpy(),
            "metric": metric_label,
        }
    )
    return summary, rows


def experiment_b() -> tuple[dict, dict, pd.DataFrame]:
    sbp_drift = pd.read_csv(DRIFT_SBP_PATH)
    dbp_drift = pd.read_csv(DRIFT_DBP_PATH)

    sbp_summary, sbp_rows = _path_b_for_metric(sbp_drift, "sbp")
    dbp_summary, dbp_rows = _path_b_for_metric(dbp_drift, "dbp")

    rows = pd.concat([sbp_rows, dbp_rows], ignore_index=True)
    return sbp_summary, dbp_summary, rows


def _avg_anchor_for_metric(
    drift_df: pd.DataFrame, metric_label: str
) -> tuple[list[dict], dict]:
    """
    Experiment C: averaged-anchor methodology on multi-anchor subjects.

    Identify subjects with >= 3 sessions. For each, treat S1 + S2 as anchors
    (mean cuff truth + mean pred_raw) and evaluate test sessions (idx >= 2)
    under 4 methods:
      M1 Hybrid 1-anchor:  cal = pred_raw + (cuff_S1 - pred_raw_S1)
      M2 Hybrid AVG:       cal = pred_raw + (mean_cuff - mean_pred_raw)
      M3 Path B 1-anchor:  cal = cuff_S1
      M4 Path B AVG:       cal = mean_cuff

    Returns (per_session_records, aggregate_summary).
    """
    df = drift_df.sort_values(["subject_id", "session_idx"]).reset_index(drop=True)

    # subjects with at least 3 sessions
    counts = df.groupby("subject_id")["session_idx"].count()
    multi = counts[counts >= 3].index.tolist()

    per_session: list[dict] = []
    for sid in multi:
        sub = df[df["subject_id"] == sid].sort_values("session_idx").reset_index(drop=True)
        if len(sub) < 3:
            continue
        s1 = sub.iloc[0]
        s2 = sub.iloc[1]
        cuff_s1 = float(s1["cuff_truth"])
        pred_s1 = float(s1["pred_raw"])
        cuff_s2 = float(s2["cuff_truth"])
        pred_s2 = float(s2["pred_raw"])
        mean_cuff = (cuff_s1 + cuff_s2) / 2.0
        mean_pred = (pred_s1 + pred_s2) / 2.0

        for k in range(2, len(sub)):
            test = sub.iloc[k]
            truth = float(test["cuff_truth"])
            pred_raw = float(test["pred_raw"])
            cal_m1 = pred_raw + (cuff_s1 - pred_s1)
            cal_m2 = pred_raw + (mean_cuff - mean_pred)
            cal_m3 = cuff_s1
            cal_m4 = mean_cuff
            per_session.append(
                {
                    "subject_id": sid,
                    "session_idx": int(test["session_idx"]),
                    "days_from_anchor": float(test["days_from_anchor"]),
                    "cuff_truth": truth,
                    "pred_raw": pred_raw,
                    "anchor_cuff_s1": cuff_s1,
                    "anchor_pred_s1": pred_s1,
                    "avg_cuff": mean_cuff,
                    "avg_pred": mean_pred,
                    "m1_pred": cal_m1,
                    "m2_pred": cal_m2,
                    "m3_pred": cal_m3,
                    "m4_pred": cal_m4,
                    "m1_err": abs(cal_m1 - truth),
                    "m2_err": abs(cal_m2 - truth),
                    "m3_err": abs(cal_m3 - truth),
                    "m4_err": abs(cal_m4 - truth),
                    "metric": metric_label,
                }
            )

    if not per_session:
        return [], {
            "metric": metric_label,
            "n_paired": 0,
            "mae_m1": float("nan"),
            "mae_m2": float("nan"),
            "mae_m3": float("nan"),
            "mae_m4": float("nan"),
        }

    arr = pd.DataFrame(per_session)
    err_m1 = arr["m1_err"].to_numpy(dtype=float)
    err_m2 = arr["m2_err"].to_numpy(dtype=float)
    err_m3 = arr["m3_err"].to_numpy(dtype=float)
    err_m4 = arr["m4_err"].to_numpy(dtype=float)

    mae_m1 = float(err_m1.mean())
    mae_m2 = float(err_m2.mean())
    mae_m3 = float(err_m3.mean())
    mae_m4 = float(err_m4.mean())

    # Bootstrap deltas (positive => first method has higher MAE => second better).
    # Reuse _bootstrap_paired with (err_better_candidate, err_worse_candidate)
    # so its delta = mean(err_naive)-mean(err_model). We label semantics manually.
    if len(err_m1) >= 2:
        d12, lo12, hi12, p12, _ = _bootstrap_paired(err_m2, err_m1, N_BOOTSTRAP, SEED)
        d14, lo14, hi14, p14, _ = _bootstrap_paired(err_m4, err_m1, N_BOOTSTRAP, SEED)
        d34, lo34, hi34, p34, _ = _bootstrap_paired(err_m4, err_m3, N_BOOTSTRAP, SEED)
    else:
        d12 = lo12 = hi12 = p12 = float("nan")
        d14 = lo14 = hi14 = p14 = float("nan")
        d34 = lo34 = hi34 = p34 = float("nan")

    summary = {
        "metric": metric_label,
        "n_paired": int(len(arr)),
        "subjects": sorted(arr["subject_id"].unique().tolist()),
        "mae_m1": mae_m1,
        "mae_m2": mae_m2,
        "mae_m3": mae_m3,
        "mae_m4": mae_m4,
        "delta_m1_m2": mae_m1 - mae_m2,
        "ci_lo_m1_m2": lo12,
        "ci_hi_m1_m2": hi12,
        "p_m1_m2": p12,
        "delta_m1_m4": mae_m1 - mae_m4,
        "ci_lo_m1_m4": lo14,
        "ci_hi_m1_m4": hi14,
        "p_m1_m4": p14,
        "delta_m3_m4": mae_m3 - mae_m4,
        "ci_lo_m3_m4": lo34,
        "ci_hi_m3_m4": hi34,
        "p_m3_m4": p34,
    }
    return per_session, summary


def experiment_c() -> tuple[dict, dict, pd.DataFrame]:
    sbp_drift = pd.read_csv(DRIFT_SBP_PATH)
    dbp_drift = pd.read_csv(DRIFT_DBP_PATH)

    sbp_records, sbp_summary = _avg_anchor_for_metric(sbp_drift, "sbp")
    dbp_records, dbp_summary = _avg_anchor_for_metric(dbp_drift, "dbp")

    rows_sbp = pd.DataFrame(sbp_records) if sbp_records else pd.DataFrame()
    rows_dbp = pd.DataFrame(dbp_records) if dbp_records else pd.DataFrame()
    if not rows_sbp.empty:
        rows_sbp["experiment"] = "C_avg_anchor_sbp"
    if not rows_dbp.empty:
        rows_dbp["experiment"] = "C_avg_anchor_dbp"
    rows = pd.concat([rows_sbp, rows_dbp], ignore_index=True, sort=False)
    return sbp_summary, dbp_summary, rows


def _verdict_bootstrap(delta: float, ci_lo: float, ci_hi: float, p: float) -> str:
    """Significance verdict for one-sided test alpha=0.025."""
    # delta>0 means model better; significant if CI lower bound > 0
    if ci_lo > 0:
        return "SIGNIFICANT (model > naive)"
    if ci_hi < 0:
        return "SIGNIFICANT (model < naive)"
    return "NOT SIGNIFICANT"


def _verdict_path_b(delta: float) -> str:
    if abs(delta) < 0.5:
        return "pop model adds NOTHING (drop defensible)"
    if delta > 0:
        return "pop model adds WORSE (Path B better)"
    if delta <= -1.0:
        return "pop model adds VALUE"
    return "pop model adds MARGINAL"


def main() -> int:
    _check_files()

    print("=" * 68)
    print("BOOTSTRAP PAIRED TEST + PATH B BASELINE")
    print("=" * 68)
    print()

    print(f"[Experiment A] Bootstrap paired test (n={N_BOOTSTRAP}, seed={SEED})")
    print()
    a_summary, a_rows = experiment_a()

    print("Reproducing the project guide numbers (sanity check):")
    print(
        f"  SBP LOSO MAE = {a_summary['mae_sbp_model']:.2f} "
        f"(expected {EXPECTED['sbp_loso_mae']:.2f})  "
        f"[{_format_pass(a_summary['mae_sbp_model'], EXPECTED['sbp_loso_mae'])}]"
    )
    print(
        f"  DBP LOSO MAE = {a_summary['mae_dbp_model']:.2f}  "
        f"(expected {EXPECTED['dbp_loso_mae']:.2f})   "
        f"[{_format_pass(a_summary['mae_dbp_model'], EXPECTED['dbp_loso_mae'])}]"
    )
    print(
        f"  Naive LOO SBP = {a_summary['mae_sbp_naive']:.2f}                   "
        f"[{_format_pass(a_summary['mae_sbp_naive'], EXPECTED['naive_sbp_mae'])}]"
    )
    print(
        f"  Naive LOO DBP = {a_summary['mae_dbp_naive']:.2f}                    "
        f"[{_format_pass(a_summary['mae_dbp_naive'], EXPECTED['naive_dbp_mae'])}]"
    )
    print()

    print("DBP test (model vs naive):")
    print(
        f"  delta = {a_summary['dbp_delta_mean']:.2f} mmHg "
        f"[95% CI: {a_summary['dbp_ci_lo']:.2f}, {a_summary['dbp_ci_hi']:.2f}]"
    )
    print(f"  p-value (1-sided) = {a_summary['dbp_p_one_sided']:.4f}")
    dbp_verdict = _verdict_bootstrap(
        a_summary["dbp_delta_mean"],
        a_summary["dbp_ci_lo"],
        a_summary["dbp_ci_hi"],
        a_summary["dbp_p_one_sided"],
    )
    print(f"  Verdict: {dbp_verdict}")
    if a_summary["dbp_ci_lo"] > 0:
        dbp_frame = "DBP win defensible (CI excludes 0)"
    else:
        dbp_frame = "DBP win directional only (CI crosses 0, n=15 underpowered)"
    print(f"  Defense framing: {dbp_frame}")
    print()

    print("SBP test (model vs naive):")
    print(
        f"  delta = {a_summary['sbp_delta_mean']:.2f} mmHg "
        f"[95% CI: {a_summary['sbp_ci_lo']:.2f}, {a_summary['sbp_ci_hi']:.2f}]"
    )
    print(f"  p-value (1-sided) = {a_summary['sbp_p_one_sided']:.4f}")
    sbp_verdict = _verdict_bootstrap(
        a_summary["sbp_delta_mean"],
        a_summary["sbp_ci_lo"],
        a_summary["sbp_ci_hi"],
        a_summary["sbp_p_one_sided"],
    )
    print(f"  Verdict: {sbp_verdict}")
    print()

    print(f"[Experiment B] Path B baseline (drop pop model) -- bootstrap n={N_BOOTSTRAP}, seed={SEED}")
    print()
    b_sbp, b_dbp, b_rows = experiment_b()

    print(f"Genuine drift sessions (SBP): n={b_sbp['n_genuine_drift']}")
    print(f"Genuine drift sessions (DBP): n={b_dbp['n_genuine_drift']}")
    print()
    print("  Method                | SBP MAE | DBP MAE")
    print("  ----------------------|---------|--------")
    print(
        f"  Hybrid (current)      | {b_sbp['mae_hybrid']:>7.2f} | {b_dbp['mae_hybrid']:>6.2f}"
    )
    print(
        f"  Path B (anchor only)  | {b_sbp['mae_path_b']:>7.2f} | {b_dbp['mae_path_b']:>6.2f}"
    )
    print(
        f"  Delta (Hybrid-PathB)  | "
        f"{b_sbp['delta_hybrid_minus_path_b']:>+7.2f} | "
        f"{b_dbp['delta_hybrid_minus_path_b']:>+6.2f}"
    )
    print()
    print("Bootstrap paired test (Hybrid vs Path B):")
    print(
        f"  SBP: delta = {b_sbp['boot_delta_mean']:+.2f} mmHg "
        f"[95% CI: {b_sbp['boot_ci_lo']:+.2f}, {b_sbp['boot_ci_hi']:+.2f}]  "
        f"p(1-sided) = {b_sbp['boot_p_one_sided']:.4f}"
    )
    print(
        f"  DBP: delta = {b_dbp['boot_delta_mean']:+.2f} mmHg "
        f"[95% CI: {b_dbp['boot_ci_lo']:+.2f}, {b_dbp['boot_ci_hi']:+.2f}]  "
        f"p(1-sided) = {b_dbp['boot_p_one_sided']:.4f}"
    )
    print("  (delta = MAE_hybrid - MAE_path_b; positive => hybrid worse = path_b better)")
    print()
    print("Sanity check vs the project guide:")
    print(
        f"  Hybrid SBP MAE = {b_sbp['mae_hybrid']:.2f} "
        f"(expected {EXPECTED['hybrid_sbp_mae']:.2f})  "
        f"[{_format_pass(b_sbp['mae_hybrid'], EXPECTED['hybrid_sbp_mae'], tol=0.5)}]"
    )
    print(
        f"  Hybrid DBP MAE = {b_dbp['mae_hybrid']:.2f}  "
        f"(expected {EXPECTED['hybrid_dbp_mae']:.2f})   "
        f"[{_format_pass(b_dbp['mae_hybrid'], EXPECTED['hybrid_dbp_mae'], tol=0.5)}]"
    )
    print()

    sbp_b_verdict = _verdict_path_b(b_sbp["delta_hybrid_minus_path_b"])
    dbp_b_verdict = _verdict_path_b(b_dbp["delta_hybrid_minus_path_b"])

    def _sig_label(ci_lo: float, ci_hi: float) -> str:
        # delta = MAE_hybrid - MAE_path_b; significant "path_b better" iff CI lo > 0.
        if np.isnan(ci_lo) or np.isnan(ci_hi):
            return "NOT TESTABLE (n<2)"
        if ci_lo > 0:
            return "SIGNIFICANT (path_b better)"
        if ci_hi < 0:
            return "SIGNIFICANT (hybrid better)"
        return "NOT SIGNIFICANT (CI crosses 0)"

    sbp_sig = _sig_label(b_sbp["boot_ci_lo"], b_sbp["boot_ci_hi"])
    dbp_sig = _sig_label(b_dbp["boot_ci_lo"], b_dbp["boot_ci_hi"])
    print("Verdict per metric:")
    print(f"  SBP: {sbp_b_verdict}  [{sbp_sig}]")
    print(f"  DBP: {dbp_b_verdict}  [{dbp_sig}]")
    print()

    print("=" * 68)
    print("KEY DECISION FOR THESIS")
    print("=" * 68)
    if a_summary["dbp_ci_lo"] > 0:
        dbp_decision = "defensible"
    else:
        dbp_decision = "fragile (CI crosses 0)"
    print(f"  - DBP win: {dbp_decision}")
    if abs(b_sbp["delta_hybrid_minus_path_b"]) < 0.5 and abs(b_dbp["delta_hybrid_minus_path_b"]) < 0.5:
        pop_decision = "drop (no contribution either metric)"
    elif b_sbp["delta_hybrid_minus_path_b"] > 0.5 and b_dbp["delta_hybrid_minus_path_b"] > 0.5:
        pop_decision = "drop (Path B beats hybrid both metrics)"
    elif b_sbp["delta_hybrid_minus_path_b"] <= -1.0 and b_dbp["delta_hybrid_minus_path_b"] <= -1.0:
        pop_decision = "keep (clear value both metrics)"
    else:
        pop_decision = "marginal (per-metric decision)"
    print(f"  - Pop model: {pop_decision}")

    framing_lines = []
    if a_summary["dbp_ci_lo"] > 0:
        framing_lines.append("DBP outperforms naive with CI excluding 0")
    else:
        framing_lines.append(
            "DBP win directional (delta="
            f"{a_summary['dbp_delta_mean']:+.2f}, CI crosses 0, n=15 underpowered)"
        )
    framing_lines.append(
        f"SBP delta={a_summary['sbp_delta_mean']:+.2f} mmHg, model "
        f"{'beats' if a_summary['sbp_delta_mean'] > 0 else 'loses to'} naive"
    )
    framing_lines.append(
        "Path B SBP delta="
        f"{b_sbp['delta_hybrid_minus_path_b']:+.2f}, "
        "DBP delta="
        f"{b_dbp['delta_hybrid_minus_path_b']:+.2f}"
    )
    print("  - Recommended framing:")
    for line in framing_lines:
        print(f"      * {line}")
    print()

    # Write CSV: per-row + summary block
    summary_rows = pd.DataFrame(
        [
            {
                "experiment": "A_summary",
                "metric": "sbp",
                "n": a_summary["n_subjects"],
                "mae_model": a_summary["mae_sbp_model"],
                "mae_naive_or_path_b": a_summary["mae_sbp_naive"],
                "delta": a_summary["sbp_delta_mean"],
                "ci_lo": a_summary["sbp_ci_lo"],
                "ci_hi": a_summary["sbp_ci_hi"],
                "p_one_sided": a_summary["sbp_p_one_sided"],
            },
            {
                "experiment": "A_summary",
                "metric": "dbp",
                "n": a_summary["n_subjects"],
                "mae_model": a_summary["mae_dbp_model"],
                "mae_naive_or_path_b": a_summary["mae_dbp_naive"],
                "delta": a_summary["dbp_delta_mean"],
                "ci_lo": a_summary["dbp_ci_lo"],
                "ci_hi": a_summary["dbp_ci_hi"],
                "p_one_sided": a_summary["dbp_p_one_sided"],
            },
            {
                "experiment": "B_summary",
                "metric": "sbp",
                "n": b_sbp["n_genuine_drift"],
                "mae_model": b_sbp["mae_hybrid"],
                "mae_naive_or_path_b": b_sbp["mae_path_b"],
                "delta": b_sbp["delta_hybrid_minus_path_b"],
                "ci_lo": b_sbp["boot_ci_lo"],
                "ci_hi": b_sbp["boot_ci_hi"],
                "p_one_sided": b_sbp["boot_p_one_sided"],
            },
            {
                "experiment": "B_summary",
                "metric": "dbp",
                "n": b_dbp["n_genuine_drift"],
                "mae_model": b_dbp["mae_hybrid"],
                "mae_naive_or_path_b": b_dbp["mae_path_b"],
                "delta": b_dbp["delta_hybrid_minus_path_b"],
                "ci_lo": b_dbp["boot_ci_lo"],
                "ci_hi": b_dbp["boot_ci_hi"],
                "p_one_sided": b_dbp["boot_p_one_sided"],
            },
        ]
    )

    # ---------------- Experiment C: averaged-anchor methodology ----------------
    print("=" * 68)
    print(
        f"[Experiment C] Averaged-anchor methodology bootstrap (n={N_BOOTSTRAP}, seed={SEED})"
    )
    print("=" * 68)
    print()

    c_sbp, c_dbp, c_rows = experiment_c()

    if c_sbp["n_paired"] == 0:
        print("[WARN] No multi-anchor subjects (>=3 sessions) found. Skipping Experiment C.")
        print()
        c_summary_rows = pd.DataFrame()
    else:
        subjects = c_sbp.get("subjects", [])
        print(
            f"Subjects: {', '.join(subjects)} -- multi-anchor sessions "
            f"(n_paired SBP={c_sbp['n_paired']}, DBP={c_dbp['n_paired']})"
        )
        print()

        # Per-session breakdown table (SBP)
        if not c_rows.empty:
            sbp_rows_only = c_rows[c_rows["metric"] == "sbp"].copy()
            sbp_rows_only = sbp_rows_only.sort_values(["subject_id", "session_idx"])
            print("Per-session breakdown (SBP, idx = session_idx in CSV):")
            print(
                "  Subject  | idx | Days  | Truth | M1     | M2     | M3     | M4"
            )
            print(
                "  ---------|-----|-------|-------|--------|--------|--------|-------"
            )
            for _, r in sbp_rows_only.iterrows():
                print(
                    f"  {r['subject_id']:<8s} | {int(r['session_idx']):>3d} | "
                    f"{r['days_from_anchor']:>5.2f} | {r['cuff_truth']:>5.0f} | "
                    f"{r['m1_err']:>6.2f} | {r['m2_err']:>6.2f} | "
                    f"{r['m3_err']:>6.2f} | {r['m4_err']:>5.2f}"
                )
            print()

        print("Aggregate MAE:")
        print("  Method                    | SBP MAE | DBP MAE")
        print("  --------------------------|---------|--------")
        print(
            f"  M1 Hybrid 1-anchor (curr) | {c_sbp['mae_m1']:>7.2f} | {c_dbp['mae_m1']:>6.2f}"
        )
        print(
            f"  M2 Hybrid AVG 2-anchor    | {c_sbp['mae_m2']:>7.2f} | {c_dbp['mae_m2']:>6.2f}"
        )
        print(
            f"  M3 Path B 1-anchor        | {c_sbp['mae_m3']:>7.2f} | {c_dbp['mae_m3']:>6.2f}"
        )
        print(
            f"  M4 Path B AVG 2-anchor    | {c_sbp['mae_m4']:>7.2f} | {c_dbp['mae_m4']:>6.2f}"
        )
        print()

        print(f"Bootstrap paired tests (n={N_BOOTSTRAP}, seed={SEED}):")
        print("  M1 vs M2 (Hybrid AVG anchor benefit; positive delta => M2 better):")
        print(
            f"    SBP: delta = {c_sbp['delta_m1_m2']:+.2f} "
            f"[CI: {c_sbp['ci_lo_m1_m2']:+.2f}, {c_sbp['ci_hi_m1_m2']:+.2f}], "
            f"p = {c_sbp['p_m1_m2']:.4f}  -> [{_sig_label(c_sbp['ci_lo_m1_m2'], c_sbp['ci_hi_m1_m2'])}]"
        )
        print(
            f"    DBP: delta = {c_dbp['delta_m1_m2']:+.2f} "
            f"[CI: {c_dbp['ci_lo_m1_m2']:+.2f}, {c_dbp['ci_hi_m1_m2']:+.2f}], "
            f"p = {c_dbp['p_m1_m2']:.4f}  -> [{_sig_label(c_dbp['ci_lo_m1_m2'], c_dbp['ci_hi_m1_m2'])}]"
        )
        print("  M1 vs M4 (Path B AVG anchor vs Hybrid 1-anchor):")
        print(
            f"    SBP: delta = {c_sbp['delta_m1_m4']:+.2f} "
            f"[CI: {c_sbp['ci_lo_m1_m4']:+.2f}, {c_sbp['ci_hi_m1_m4']:+.2f}], "
            f"p = {c_sbp['p_m1_m4']:.4f}  -> [{_sig_label(c_sbp['ci_lo_m1_m4'], c_sbp['ci_hi_m1_m4'])}]"
        )
        print(
            f"    DBP: delta = {c_dbp['delta_m1_m4']:+.2f} "
            f"[CI: {c_dbp['ci_lo_m1_m4']:+.2f}, {c_dbp['ci_hi_m1_m4']:+.2f}], "
            f"p = {c_dbp['p_m1_m4']:.4f}  -> [{_sig_label(c_dbp['ci_lo_m1_m4'], c_dbp['ci_hi_m1_m4'])}]"
        )
        print("  M3 vs M4 (Path B AVG vs Path B 1-anchor):")
        print(
            f"    SBP: delta = {c_sbp['delta_m3_m4']:+.2f} "
            f"[CI: {c_sbp['ci_lo_m3_m4']:+.2f}, {c_sbp['ci_hi_m3_m4']:+.2f}], "
            f"p = {c_sbp['p_m3_m4']:.4f}  -> [{_sig_label(c_sbp['ci_lo_m3_m4'], c_sbp['ci_hi_m3_m4'])}]"
        )
        print(
            f"    DBP: delta = {c_dbp['delta_m3_m4']:+.2f} "
            f"[CI: {c_dbp['ci_lo_m3_m4']:+.2f}, {c_dbp['ci_hi_m3_m4']:+.2f}], "
            f"p = {c_dbp['p_m3_m4']:.4f}  -> [{_sig_label(c_dbp['ci_lo_m3_m4'], c_dbp['ci_hi_m3_m4'])}]"
        )
        print()
        print("Verdict:")
        print(f"  - n={c_sbp['n_paired']} EXTREMELY underpowered, CI will be very wide")
        print(
            f"  - Headline: M2 vs M1 SBP delta = {c_sbp['delta_m1_m2']:+.2f} mmHg "
            f"on {', '.join(subjects)} subset"
        )
        if c_sbp["ci_lo_m1_m2"] > 0:
            print("  - SBP M2>M1 CI excludes 0 -> defensible")
        else:
            print(
                f"  - SBP M2>M1 CI [{c_sbp['ci_lo_m1_m2']:+.2f}, "
                f"{c_sbp['ci_hi_m1_m2']:+.2f}] crosses 0 -> directional only n={c_sbp['n_paired']}"
            )
        print()

        # Build summary rows for CSV (3 comparisons x 2 metrics = 6 rows)
        c_summary_rows = pd.DataFrame(
            [
                {
                    "experiment": "C_summary",
                    "metric": "sbp",
                    "comparison": "m1_vs_m2",
                    "n": c_sbp["n_paired"],
                    "mae_a": c_sbp["mae_m1"],
                    "mae_b": c_sbp["mae_m2"],
                    "delta": c_sbp["delta_m1_m2"],
                    "ci_lo": c_sbp["ci_lo_m1_m2"],
                    "ci_hi": c_sbp["ci_hi_m1_m2"],
                    "p_one_sided": c_sbp["p_m1_m2"],
                },
                {
                    "experiment": "C_summary",
                    "metric": "dbp",
                    "comparison": "m1_vs_m2",
                    "n": c_dbp["n_paired"],
                    "mae_a": c_dbp["mae_m1"],
                    "mae_b": c_dbp["mae_m2"],
                    "delta": c_dbp["delta_m1_m2"],
                    "ci_lo": c_dbp["ci_lo_m1_m2"],
                    "ci_hi": c_dbp["ci_hi_m1_m2"],
                    "p_one_sided": c_dbp["p_m1_m2"],
                },
                {
                    "experiment": "C_summary",
                    "metric": "sbp",
                    "comparison": "m1_vs_m4",
                    "n": c_sbp["n_paired"],
                    "mae_a": c_sbp["mae_m1"],
                    "mae_b": c_sbp["mae_m4"],
                    "delta": c_sbp["delta_m1_m4"],
                    "ci_lo": c_sbp["ci_lo_m1_m4"],
                    "ci_hi": c_sbp["ci_hi_m1_m4"],
                    "p_one_sided": c_sbp["p_m1_m4"],
                },
                {
                    "experiment": "C_summary",
                    "metric": "dbp",
                    "comparison": "m1_vs_m4",
                    "n": c_dbp["n_paired"],
                    "mae_a": c_dbp["mae_m1"],
                    "mae_b": c_dbp["mae_m4"],
                    "delta": c_dbp["delta_m1_m4"],
                    "ci_lo": c_dbp["ci_lo_m1_m4"],
                    "ci_hi": c_dbp["ci_hi_m1_m4"],
                    "p_one_sided": c_dbp["p_m1_m4"],
                },
                {
                    "experiment": "C_summary",
                    "metric": "sbp",
                    "comparison": "m3_vs_m4",
                    "n": c_sbp["n_paired"],
                    "mae_a": c_sbp["mae_m3"],
                    "mae_b": c_sbp["mae_m4"],
                    "delta": c_sbp["delta_m3_m4"],
                    "ci_lo": c_sbp["ci_lo_m3_m4"],
                    "ci_hi": c_sbp["ci_hi_m3_m4"],
                    "p_one_sided": c_sbp["p_m3_m4"],
                },
                {
                    "experiment": "C_summary",
                    "metric": "dbp",
                    "comparison": "m3_vs_m4",
                    "n": c_dbp["n_paired"],
                    "mae_a": c_dbp["mae_m3"],
                    "mae_b": c_dbp["mae_m4"],
                    "delta": c_dbp["delta_m3_m4"],
                    "ci_lo": c_dbp["ci_lo_m3_m4"],
                    "ci_hi": c_dbp["ci_hi_m3_m4"],
                    "p_one_sided": c_dbp["p_m3_m4"],
                },
            ]
        )

    a_rows_aug = a_rows.copy()
    a_rows_aug["metric"] = ""  # placeholder so columns align
    out = pd.concat(
        [a_rows_aug, b_rows, c_rows, summary_rows, c_summary_rows],
        ignore_index=True,
        sort=False,
    )
    out.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved: {OUTPUT_PATH.name} ({len(out)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
