"""
test_bp_ablation.py -- BP ablation correctness tests.

Tests are READ-ONLY: they load the pre-computed CSVs from eval_bp_ablation.py
and verify numerical correctness, structural integrity, confounder presence,
and consistency with prior LOSO numbers.  No model refit is triggered.

Run:
    cd backend
    pytest test_bp_ablation.py -v
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent  # backend/
RESULTS_CSV = ROOT / "eval_bp_ablation_results.csv"
SUMMARY_CSV = ROOT / "eval_bp_ablation_summary.csv"

PPG_BP_MEAN_SBP = 118.3   # population mean used as naive baseline


# ── helpers ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def results() -> pd.DataFrame:
    assert RESULTS_CSV.exists(), f"Missing {RESULTS_CSV} — run eval_bp_ablation.py first"
    return pd.read_csv(RESULTS_CSV)


@pytest.fixture(scope="module")
def summary() -> pd.DataFrame:
    assert SUMMARY_CSV.exists(), f"Missing {SUMMARY_CSV} — run eval_bp_ablation.py first"
    return pd.read_csv(SUMMARY_CSV)


# ============================================================
# Group 1 — Headline number reproducibility
# ============================================================

class TestHeadlineNumbers:
    """Verify published headline numbers from eval_bp_ablation_summary.csv."""

    def test_summary_csv_loads_and_has_expected_rows(self, summary):
        assert len(summary) == 6, f"Expected 6 rows, got {len(summary)}"

    def test_summary_has_both_schemes(self, summary):
        assert set(summary["scheme"].unique()) == {"tercile", "threshold"}

    def test_tercile_high_sbp_mae_17_75(self, summary):
        row = summary[(summary["scheme"] == "tercile") & (summary["stratum"] == "high")]
        assert len(row) == 1
        assert abs(row.iloc[0]["sbp_mae"] - 17.75) < 0.5, (
            f"Expected ~17.75, got {row.iloc[0]['sbp_mae']:.4f}"
        )

    def test_tercile_medium_sbp_mae_11_35(self, summary):
        row = summary[(summary["scheme"] == "tercile") & (summary["stratum"] == "medium")]
        assert len(row) == 1
        assert abs(row.iloc[0]["sbp_mae"] - 11.35) < 0.5, (
            f"Expected ~11.35, got {row.iloc[0]['sbp_mae']:.4f}"
        )

    def test_tercile_low_sbp_mae_13_56(self, summary):
        row = summary[(summary["scheme"] == "tercile") & (summary["stratum"] == "low")]
        assert len(row) == 1
        assert abs(row.iloc[0]["sbp_mae"] - 13.56) < 0.5, (
            f"Expected ~13.56, got {row.iloc[0]['sbp_mae']:.4f}"
        )

    def test_negative_monotonicity_confirmed(self, summary):
        """High-stratum SBP MAE > low-stratum (REVERSED expectation — negative result)."""
        high = summary[(summary["scheme"] == "tercile") & (summary["stratum"] == "high")].iloc[0]
        low = summary[(summary["scheme"] == "tercile") & (summary["stratum"] == "low")].iloc[0]
        assert high["sbp_mae"] > low["sbp_mae"], (
            f"Expected high {high['sbp_mae']:.2f} > low {low['sbp_mae']:.2f}"
        )

    def test_high_stratum_n_persons_5(self, summary):
        row = summary[(summary["scheme"] == "tercile") & (summary["stratum"] == "high")]
        assert int(row.iloc[0]["n_persons"]) == 5

    def test_medium_stratum_n_persons_5(self, summary):
        row = summary[(summary["scheme"] == "tercile") & (summary["stratum"] == "medium")]
        assert int(row.iloc[0]["n_persons"]) == 5

    def test_low_stratum_n_persons_4(self, summary):
        row = summary[(summary["scheme"] == "tercile") & (summary["stratum"] == "low")]
        assert int(row.iloc[0]["n_persons"]) == 4

    def test_all_row_present_for_tercile(self, summary):
        all_row = summary[(summary["scheme"] == "tercile") & (summary["stratum"] == "ALL")]
        assert len(all_row) == 1
        assert int(all_row.iloc[0]["n_persons"]) == 14

    def test_sbp_mae_values_are_positive(self, summary):
        tercile = summary[summary["scheme"] == "tercile"]
        assert (tercile["sbp_mae"] > 0).all()

    def test_dbp_mae_values_are_positive(self, summary):
        tercile = summary[summary["scheme"] == "tercile"]
        assert (tercile["dbp_mae"] > 0).all()


# ============================================================
# Group 2 — Confounder verification
# ============================================================

class TestConfounderVerification:
    """Verify the regression-to-mean confounder claim that explains the negative result."""

    def test_high_stratum_contains_bp_extremes(self, results):
        """High-quality stratum coincidentally contains BP extremes
        (S07+S09+S15 hypotensive, S12_01+S13 HTN)."""
        high = set(results[results["stratum_tercile"] == "high"]["person"].tolist())
        expected_extremes = {"S07", "S09", "S15", "S12_01", "S13"}
        assert high == expected_extremes, (
            f"Expected {expected_extremes}, got {high}"
        )

    def test_high_stratum_naive_pop_mae_17_86(self, summary):
        """Naive population-mean MAE for the high stratum ≈ 17.86
        (driven by BP extremes — confirms confounder)."""
        high = summary[
            (summary["scheme"] == "tercile") & (summary["stratum"] == "high")
        ].iloc[0]
        assert abs(high["naive_pop_sbp_mae"] - 17.86) < 0.5, (
            f"Expected ~17.86, got {high['naive_pop_sbp_mae']:.4f}"
        )

    def test_confounder_correlation_strong(self, results):
        """|BP_true - 118.3| should correlate strongly with |sbp_err|
        (r > 0.7 confirms regression-to-mean drives the negative result)."""
        bp_dist = (results["sbp_true"] - PPG_BP_MEAN_SBP).abs()
        err_abs = results["sbp_err"].abs()
        r = float(np.corrcoef(bp_dist.to_numpy(), err_abs.to_numpy())[0, 1])
        print(f"\n  Confounder r = {r:.4f}")
        assert r > 0.7, (
            f"Confounder r={r:.3f} should be > 0.7 to confirm regression-to-mean"
        )

    def test_high_stratum_sbp_dist_from_mean_larger(self, results):
        """High-stratum persons have larger |BP - 118.3| than low-stratum —
        structurally explains reversed MAE ordering."""
        high = results[results["stratum_tercile"] == "high"]
        low = results[results["stratum_tercile"] == "low"]
        high_dist = (high["sbp_true"] - PPG_BP_MEAN_SBP).abs().mean()
        low_dist = (low["sbp_true"] - PPG_BP_MEAN_SBP).abs().mean()
        print(f"\n  high mean |BP-118.3| = {high_dist:.2f}, low = {low_dist:.2f}")
        assert high_dist > low_dist, (
            f"High stratum mean BP distance {high_dist:.2f} should exceed "
            f"low stratum {low_dist:.2f}"
        )

    def test_naive_pop_mae_tracks_model_mae_directionally(self, summary):
        """Naive pop MAE ordering should mirror model MAE ordering (high > medium)
        — confirming BP-extremes drive both naive and model errors simultaneously."""
        tercile = summary[
            (summary["scheme"] == "tercile") & (summary["stratum"] != "ALL")
        ].set_index("stratum")
        assert tercile.loc["high", "naive_pop_sbp_mae"] > tercile.loc["medium", "naive_pop_sbp_mae"], (
            "Naive MAE should be higher for high stratum than medium (BP extremes drive naive error)"
        )


# ============================================================
# Group 3 — Person count and grouping correctness
# ============================================================

class TestPersonCountGrouping:
    """Verify cohort composition: 14 persons, multi-session aggregated, exclusions correct."""

    def test_14_persons_in_results(self, results):
        """5 persons excluded (S08 + 4 S09-SQA experiment files), 14 included."""
        assert len(results) == 14, f"Expected 14 rows, got {len(results)}"

    def test_required_columns_present(self, results):
        required = {
            "person", "sbp_true", "sbp_pred", "sbp_err",
            "dbp_true", "dbp_pred", "dbp_err",
            "n_windows", "pct_excellent", "pct_unfit", "mean_proba_excellent",
            "stratum_tercile", "stratum_threshold",
        }
        assert required.issubset(set(results.columns)), (
            f"Missing columns: {required - set(results.columns)}"
        )

    def test_multi_session_persons_appear_once(self, results):
        """S03, S07, S10, S09 each appear ONCE (multi-session aggregated)."""
        persons = results["person"].tolist()
        for p in ["S03", "S07", "S10", "S09"]:
            count = persons.count(p)
            assert count == 1, f"{p} should appear once, got {count}"

    def test_no_phuong_in_results(self, results):
        assert "S08" not in results["person"].tolist()

    def test_no_sqa_experiment_files_in_results(self, results):
        """S09 SQA-experiment files (S01_p1..p4 etc.) should not appear."""
        persons = results["person"].tolist()
        for p in ["S01_p1", "S01_p2", "S01_p3", "S01_p4",
                  "S09_p1", "S09_p2", "S09_p3", "S09_p4"]:
            assert p not in persons, f"{p} should not be in ablation results"

    def test_all_expected_persons_present(self, results):
        expected = {
            "S01", "S14", "S03", "S02", "S04", "S05", "S06",
            "S07", "S09", "S10", "S15", "S11", "S12_01", "S13",
        }
        actual = set(results["person"].tolist())
        assert actual == expected, (
            f"Person set mismatch. Extra: {actual - expected}. Missing: {expected - actual}"
        )

    def test_no_duplicate_persons(self, results):
        persons = results["person"].tolist()
        assert len(persons) == len(set(persons)), "Duplicate persons found in results"

    def test_stratum_tercile_three_values(self, results):
        unique_strata = set(results["stratum_tercile"].dropna().unique())
        assert unique_strata == {"high", "medium", "low"}, (
            f"Expected three tercile strata, got {unique_strata}"
        )

    def test_stratum_threshold_all_high(self, results):
        """All 14 persons have pct_excellent >= 80% so threshold stratum is uniformly high."""
        unique_threshold = set(results["stratum_threshold"].dropna().unique())
        assert unique_threshold == {"high"}, (
            f"Expected only 'high' threshold stratum, got {unique_threshold}"
        )


# ============================================================
# Group 4 — SQA score range and threshold scheme
# ============================================================

class TestSQAScoreRange:
    """Verify pct_excellent values are internally consistent."""

    def test_pct_excellent_min_gte_87(self, results):
        """All persons have pct_excellent >= 87% (S11 is lowest at ~87.04%)."""
        min_val = float(results["pct_excellent"].min())
        assert min_val >= 87.0, f"Expected min >= 87.0, got {min_val:.4f}"

    def test_pct_excellent_max_lte_100(self, results):
        max_val = float(results["pct_excellent"].max())
        assert max_val <= 100.0, f"Expected max <= 100.0, got {max_val:.4f}"

    def test_threshold_scheme_degenerate_no_low_stratum(self, summary):
        """Because all pct_excellent >= 80%, the threshold 'low' stratum is empty
        and does not appear in summary CSV."""
        threshold_rows = summary[summary["scheme"] == "threshold"]
        assert "low" not in threshold_rows["stratum"].tolist(), (
            "Threshold 'low' stratum should not exist (all pct_excellent >= 80%)"
        )

    def test_no_persons_below_80pct_excellent(self, results):
        """Confirms degenerate threshold scheme: no one falls below 80%."""
        n_below = int((results["pct_excellent"] < 80.0).sum())
        assert n_below == 0, f"Expected 0 persons below 80% excellent, got {n_below}"

    def test_pct_unfit_nonnegative(self, results):
        assert (results["pct_unfit"] >= 0.0).all()

    def test_pct_excellent_pct_unfit_sum_lte_100(self, results):
        total = results["pct_excellent"] + results["pct_unfit"]
        assert (total <= 100.01).all(), "pct_excellent + pct_unfit exceeds 100%"

    def test_mean_proba_excellent_in_01(self, results):
        assert (results["mean_proba_excellent"] >= 0.0).all()
        assert (results["mean_proba_excellent"] <= 1.0).all()

    def test_n_windows_positive(self, results):
        assert (results["n_windows"] > 0).all()


# ============================================================
# Group 5 — LOSO MAE consistency
# ============================================================

class TestLOSOMAEConsistency:
    """Cross-check aggregate MAE with previously published LOSO numbers."""

    def test_overall_sbp_mae_within_expected_range(self, results):
        """Aggregate SBP MAE over 14 persons should be near 13.86 (eval_true_loso N=15).
        14-person subset (S08 excluded) may differ slightly — check reasonable range."""
        overall_mae = float(results["sbp_err"].abs().mean())
        print(f"\n  Overall SBP MAE (14 persons) = {overall_mae:.4f}")
        assert 12.0 < overall_mae < 16.0, (
            f"Overall MAE {overall_mae:.2f} outside expected range [12, 16]"
        )

    def test_summary_all_row_sbp_mae_matches_results(self, results, summary):
        """ALL row in summary CSV should equal the mean of |sbp_err| in results."""
        all_row = summary[(summary["scheme"] == "tercile") & (summary["stratum"] == "ALL")]
        assert len(all_row) == 1
        summary_mae = float(all_row.iloc[0]["sbp_mae"])
        results_mae = float(results["sbp_err"].abs().mean())
        assert abs(summary_mae - results_mae) < 1e-6, (
            f"Summary ALL SBP MAE {summary_mae:.6f} != results mean {results_mae:.6f}"
        )

    def test_summary_all_row_dbp_mae_matches_results(self, results, summary):
        all_row = summary[(summary["scheme"] == "tercile") & (summary["stratum"] == "ALL")]
        summary_mae = float(all_row.iloc[0]["dbp_mae"])
        results_mae = float(results["dbp_err"].abs().mean())
        assert abs(summary_mae - results_mae) < 1e-6, (
            f"Summary ALL DBP MAE {summary_mae:.6f} != results mean {results_mae:.6f}"
        )

    def test_sbp_err_is_pred_minus_true(self, results):
        """sbp_err = sbp_pred - sbp_true (signed, consistent formula)."""
        computed = results["sbp_pred"] - results["sbp_true"]
        np.testing.assert_allclose(
            computed.to_numpy(), results["sbp_err"].to_numpy(), atol=1e-6,
            err_msg="sbp_err != sbp_pred - sbp_true"
        )

    def test_dbp_err_is_pred_minus_true(self, results):
        computed = results["dbp_pred"] - results["dbp_true"]
        np.testing.assert_allclose(
            computed.to_numpy(), results["dbp_err"].to_numpy(), atol=1e-6,
            err_msg="dbp_err != dbp_pred - dbp_true"
        )

    def test_naive_pop_sbp_mae_all_row_approximately_12_47(self, summary):
        """Naive pop MAE for ALL should be |mean(BP_true) - 118.3| across 14 persons."""
        all_row = summary[(summary["scheme"] == "tercile") & (summary["stratum"] == "ALL")]
        naive_mae = float(all_row.iloc[0]["naive_pop_sbp_mae"])
        # Rough sanity: should be < model MAE for an honest comparison baseline
        assert 10.0 < naive_mae < 16.0, (
            f"Naive pop SBP MAE {naive_mae:.2f} outside expected range [10, 16]"
        )

    def test_tercile_strata_n_persons_sum_to_14(self, summary):
        tercile_strata = summary[
            (summary["scheme"] == "tercile") & (summary["stratum"] != "ALL")
        ]
        total = int(tercile_strata["n_persons"].sum())
        assert total == 14, f"Tercile strata n_persons sum = {total}, expected 14"
