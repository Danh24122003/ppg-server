"""
Direct measurement of Path B 1-anchor calibration MAE on current cohort.

Eval-only. Reads existing drift CSVs (already encode pred_raw, cuff_truth,
session_idx, anchor flags). For each genuine drift session (days>0):

    path_b_1anchor_pred = cuff_S1                       (anchor cuff truth only)
    err = |cuff_truth - path_b_1anchor_pred|

Then computes:
  - Direct MAE (SBP + DBP) on n=14 drift datapoints
  - Per-subject breakdown
  - Bootstrap 95% CI (10,000 resamples, seed=42)
  - Cross-check vs derived estimate (Hybrid MAE + (PathB - Hybrid) delta)

No production code modified. No model retrained. Production bundle untouched.

Inputs:
  - _drift_results_sbp.csv
  - _drift_results_dbp.csv

Outputs (under eval_path_b_1anchor_rerun_2026-05-14/):
  - per_subject_breakdown.csv
  - bootstrap_summary.csv
  - report.txt
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).parent
DRIFT_SBP_PATH = HERE / "_drift_results_sbp.csv"
DRIFT_DBP_PATH = HERE / "_drift_results_dbp.csv"
OUTPUT_DIR = HERE / "eval_path_b_1anchor_rerun_2026-05-14"

N_BOOTSTRAP = 10_000
SEED = 42


def _check_files() -> None:
    for p in (DRIFT_SBP_PATH, DRIFT_DBP_PATH):
        if not p.exists():
            print(f"[ERROR] missing input file: {p}")
            sys.exit(1)
    OUTPUT_DIR.mkdir(exist_ok=True)


def _path_b_1anchor(drift_df: pd.DataFrame, metric_label: str) -> pd.DataFrame:
    """Compute Path B 1-anchor predictions on genuine drift sessions."""
    anchors = drift_df[drift_df["is_anchor"]].copy()
    anchor_map = dict(zip(anchors["subject_id"], anchors["cuff_truth"].astype(float)))

    drift = drift_df[
        (drift_df["included_in_drift_eval"]) & (drift_df["days_from_anchor"] > 0)
    ].copy()
    drift["cuff_anchor"] = drift["subject_id"].map(anchor_map)
    if drift["cuff_anchor"].isna().any():
        missing = drift.loc[drift["cuff_anchor"].isna(), "subject_id"].unique().tolist()
        print(f"[WARN] {metric_label}: drift subjects without anchor: {missing}")
        drift = drift.dropna(subset=["cuff_anchor"])

    drift["path_b_pred"] = drift["cuff_anchor"].astype(float)
    drift["path_b_err"] = (drift["cuff_truth"].astype(float) - drift["path_b_pred"]).abs()
    drift["hybrid_err"] = drift["abs_err_calibrated"].astype(float)
    drift["metric"] = metric_label
    return drift[[
        "subject_id",
        "session_idx",
        "days_from_anchor",
        "cuff_truth",
        "cuff_anchor",
        "path_b_pred",
        "path_b_err",
        "hybrid_err",
        "metric",
    ]].reset_index(drop=True)


def _bootstrap_ci(values: np.ndarray, n_boot: int, seed: int) -> tuple[float, float, float]:
    """Returns (mean, ci_lo, ci_hi) for MAE via percentile bootstrap."""
    rng = np.random.default_rng(seed)
    n = len(values)
    boots = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[b] = values[idx].mean()
    return float(values.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main() -> int:
    _check_files()

    sbp_drift_raw = pd.read_csv(DRIFT_SBP_PATH)
    dbp_drift_raw = pd.read_csv(DRIFT_DBP_PATH)

    sbp_rows = _path_b_1anchor(sbp_drift_raw, "sbp")
    dbp_rows = _path_b_1anchor(dbp_drift_raw, "dbp")

    n_sbp = len(sbp_rows)
    n_dbp = len(dbp_rows)

    # Direct MAE
    sbp_mae = float(sbp_rows["path_b_err"].mean())
    dbp_mae = float(dbp_rows["path_b_err"].mean())
    sbp_hybrid = float(sbp_rows["hybrid_err"].mean())
    dbp_hybrid = float(dbp_rows["hybrid_err"].mean())

    # Bootstrap CIs (10,000 resamples, seed=42)
    sbp_mean, sbp_lo, sbp_hi = _bootstrap_ci(
        sbp_rows["path_b_err"].to_numpy(dtype=float), N_BOOTSTRAP, SEED
    )
    dbp_mean, dbp_lo, dbp_hi = _bootstrap_ci(
        dbp_rows["path_b_err"].to_numpy(dtype=float), N_BOOTSTRAP, SEED
    )

    # Cross-check vs derived
    derived_sbp = sbp_hybrid + (sbp_mae - sbp_hybrid)  # tautology by construction
    derived_dbp = dbp_hybrid + (dbp_mae - dbp_hybrid)
    # The "derived" reference in the project guide is 7.43 + 0.93 = 8.36 (SBP), 6.90 + 0.97 = 7.87 (DBP)
    claude_md_hybrid_sbp = 7.43
    claude_md_hybrid_dbp = 6.90
    claude_md_delta_sbp = 0.93
    claude_md_delta_dbp = 0.97
    claude_md_derived_pathB_sbp = claude_md_hybrid_sbp + claude_md_delta_sbp
    claude_md_derived_pathB_dbp = claude_md_hybrid_dbp + claude_md_delta_dbp

    # Per-subject table
    per_subject_sbp = (
        sbp_rows.groupby("subject_id")["path_b_err"].agg(["count", "mean"]).reset_index()
    )
    per_subject_sbp.columns = ["subject_id", "n_drift_sessions", "sbp_mae"]
    per_subject_dbp = (
        dbp_rows.groupby("subject_id")["path_b_err"].agg(["count", "mean"]).reset_index()
    )
    per_subject_dbp.columns = ["subject_id", "n_drift_sessions", "dbp_mae"]
    per_subject = per_subject_sbp.merge(per_subject_dbp, on=["subject_id", "n_drift_sessions"], how="outer")
    per_subject = per_subject.sort_values("subject_id").reset_index(drop=True)
    per_subject.to_csv(OUTPUT_DIR / "per_subject_breakdown.csv", index=False)

    # Also write per-row results (n=14 SBP + n=14 DBP, traceable)
    rows_combined = pd.concat([sbp_rows, dbp_rows], ignore_index=True)
    rows_combined.to_csv(OUTPUT_DIR / "per_row_path_b_1anchor.csv", index=False)

    # Bootstrap summary
    summary = pd.DataFrame(
        [
            {
                "metric": "sbp",
                "n_drift": n_sbp,
                "path_b_mae_direct": sbp_mae,
                "ci_lo_95": sbp_lo,
                "ci_hi_95": sbp_hi,
                "hybrid_mae_direct": sbp_hybrid,
                "delta_pathB_minus_hybrid": sbp_mae - sbp_hybrid,
                "claude_md_derived_pathB": claude_md_derived_pathB_sbp,
                "diff_direct_vs_derived": sbp_mae - claude_md_derived_pathB_sbp,
            },
            {
                "metric": "dbp",
                "n_drift": n_dbp,
                "path_b_mae_direct": dbp_mae,
                "ci_lo_95": dbp_lo,
                "ci_hi_95": dbp_hi,
                "hybrid_mae_direct": dbp_hybrid,
                "delta_pathB_minus_hybrid": dbp_mae - dbp_hybrid,
                "claude_md_derived_pathB": claude_md_derived_pathB_dbp,
                "diff_direct_vs_derived": dbp_mae - claude_md_derived_pathB_dbp,
            },
        ]
    )
    summary.to_csv(OUTPUT_DIR / "bootstrap_summary.csv", index=False)

    # Build text report
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("PATH B 1-ANCHOR DIRECT MEASUREMENT (drift n=14 each metric)")
    lines.append("=" * 72)
    lines.append("")
    lines.append("Formula: path_b_pred = cuff_S1 (anchor cuff truth only)")
    lines.append(f"Inputs:  _drift_results_sbp.csv (mtime 11/5) + _drift_results_dbp.csv (mtime 11/5)")
    lines.append(f"Cohort:  n_sbp={n_sbp}, n_dbp={n_dbp} drift datapoints")
    lines.append(
        f"Subjects: {', '.join(sorted(sbp_rows['subject_id'].unique().tolist()))}"
    )
    lines.append("")
    lines.append("=" * 72)
    lines.append("DIRECT MAE")
    lines.append("=" * 72)
    lines.append("")
    lines.append("  Metric | n  | Path B MAE | 95% CI (boot)        | Hybrid MAE (ref) | Path B - Hybrid")
    lines.append("  -------|----|------------|----------------------|------------------|----------------")
    lines.append(
        f"  SBP    | {n_sbp:>2d} | {sbp_mae:>9.2f}  | [{sbp_lo:>5.2f}, {sbp_hi:>5.2f}]        | "
        f"{sbp_hybrid:>15.2f}  | {sbp_mae - sbp_hybrid:>+14.2f}"
    )
    lines.append(
        f"  DBP    | {n_dbp:>2d} | {dbp_mae:>9.2f}  | [{dbp_lo:>5.2f}, {dbp_hi:>5.2f}]        | "
        f"{dbp_hybrid:>15.2f}  | {dbp_mae - dbp_hybrid:>+14.2f}"
    )
    lines.append("")
    lines.append("=" * 72)
    lines.append("CROSS-CHECK: direct vs the project guide derived estimate")
    lines.append("=" * 72)
    lines.append("")
    lines.append(
        f"  SBP: direct = {sbp_mae:.2f}  vs derived (7.43 + 0.93) = "
        f"{claude_md_derived_pathB_sbp:.2f}  diff = {sbp_mae - claude_md_derived_pathB_sbp:+.2f}"
    )
    lines.append(
        f"  DBP: direct = {dbp_mae:.2f}  vs derived (6.90 + 0.97) = "
        f"{claude_md_derived_pathB_dbp:.2f}  diff = {dbp_mae - claude_md_derived_pathB_dbp:+.2f}"
    )
    lines.append("")
    lines.append("=" * 72)
    lines.append("PER-SUBJECT BREAKDOWN")
    lines.append("=" * 72)
    lines.append("")
    lines.append("  Subject    | n  | SBP MAE | DBP MAE")
    lines.append("  -----------|----|---------|--------")
    for _, r in per_subject.iterrows():
        sbp_v = r.get("sbp_mae", float("nan"))
        dbp_v = r.get("dbp_mae", float("nan"))
        lines.append(
            f"  {r['subject_id']:<10s} | {int(r['n_drift_sessions']):>2d} | "
            f"{sbp_v:>7.2f} | {dbp_v:>6.2f}"
        )
    lines.append("")
    lines.append("=" * 72)
    lines.append("PER-ROW (n=14 SBP detail)")
    lines.append("=" * 72)
    lines.append("")
    lines.append(
        "  Subject   | idx | Days  | Truth | Anchor S1 | Path B err"
    )
    lines.append(
        "  ----------|-----|-------|-------|-----------|-----------"
    )
    for _, r in sbp_rows.sort_values(["subject_id", "session_idx"]).iterrows():
        lines.append(
            f"  {r['subject_id']:<9s} | {int(r['session_idx']):>3d} | "
            f"{r['days_from_anchor']:>5.2f} | {r['cuff_truth']:>5.0f} | "
            f"{r['cuff_anchor']:>9.0f} | {r['path_b_err']:>9.2f}"
        )
    lines.append("")
    lines.append("=" * 72)
    lines.append("PER-ROW (n=14 DBP detail)")
    lines.append("=" * 72)
    lines.append("")
    lines.append(
        "  Subject   | idx | Days  | Truth | Anchor S1 | Path B err"
    )
    lines.append(
        "  ----------|-----|-------|-------|-----------|-----------"
    )
    for _, r in dbp_rows.sort_values(["subject_id", "session_idx"]).iterrows():
        lines.append(
            f"  {r['subject_id']:<9s} | {int(r['session_idx']):>3d} | "
            f"{r['days_from_anchor']:>5.2f} | {r['cuff_truth']:>5.0f} | "
            f"{r['cuff_anchor']:>9.0f} | {r['path_b_err']:>9.2f}"
        )
    lines.append("")
    lines.append("=" * 72)
    lines.append("ABSTRACT-READY SENTENCE")
    lines.append("=" * 72)
    lines.append("")
    lines.append(
        f"  Path B 1-anchor per-subject calibration (n={n_sbp} drift sessions, "
        f"15 unique subjects, 8 multi-session):"
    )
    lines.append(
        f"      SBP MAE = {sbp_mae:.2f} mmHg [95% CI {sbp_lo:.2f}, {sbp_hi:.2f}]"
    )
    lines.append(
        f"      DBP MAE = {dbp_mae:.2f} mmHg [95% CI {dbp_lo:.2f}, {dbp_hi:.2f}]"
    )
    lines.append("")

    text = "\n".join(lines)
    print(text)
    (OUTPUT_DIR / "report.txt").write_text(text, encoding="utf-8")
    print(f"Saved: {OUTPUT_DIR.name}/report.txt + per_subject_breakdown.csv + bootstrap_summary.csv + per_row_path_b_1anchor.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
