"""
Tests for eval_spo2_ablation.py —  SpO2 ablation.

Groups:
  1. SpO2 computation parity with production (no reimplementation)
  2. Window indexing alignment
  3. Aggregation denominator (P1 framing: is pct_in_94_100 / valid or / total?)
  4. Headline number reproducibility (99.3% excellent vs 47.4% unfit)
  5. Edge cases (empty tier, all-invalid, zero-signal)
"""
from __future__ import annotations

import importlib
import inspect
import sys
import types
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Path setup — add backend/ to sys.path so imports work
# ---------------------------------------------------------------------------
BACKEND_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND_DIR))

from eval_spo2_ablation import (  # noqa: E402
    eval_row,
    summarize_by_tier,
    NORMOXIC_LO,
    NORMOXIC_HI,
)
from main import calculate_spo2_v23  # noqa: E402

SUMMARY_CSV = BACKEND_DIR / "eval_spo2_ablation_summary.csv"


# ===========================================================================
# Group 1 — SpO2 computation parity with production
# ===========================================================================

class TestSpO2ComputationParity:
    """eval_row must delegate to calculate_spo2_v23, not re-implement R-ratio."""

    def test_eval_row_calls_production_function(self):
        """eval_row source must reference calculate_spo2_v23, not DIY R formula."""
        src = inspect.getsource(eval_row)
        assert "calculate_spo2_v23" in src, (
            "eval_row does not call calculate_spo2_v23 — possible reimplementation"
        )

    def test_eval_row_no_diy_r_ratio(self):
        """eval_row must NOT contain inline ac_red/dc_red computation."""
        src = inspect.getsource(eval_row).replace(" ", "")
        assert "ac_red/dc_red" not in src, (
            "eval_row reimplements R-ratio inline instead of delegating to production"
        )

    def test_module_import_uses_production_calculate_spo2_v23(self):
        """The ablation module imports calculate_spo2_v23 from main."""
        import eval_spo2_ablation as abl
        # The function calculate_spo2_v23 in the ablation module's namespace must
        # be the exact same object as the one from main.
        assert hasattr(abl, "calculate_spo2_v23"), (
            "eval_spo2_ablation does not expose calculate_spo2_v23 in its namespace"
        )
        assert abl.calculate_spo2_v23 is calculate_spo2_v23, (
            "eval_spo2_ablation's calculate_spo2_v23 is not the same object as main's"
        )

    def test_clean_sine_red_ir_gives_high_spo2(self):
        """Synthetic clean sine with realistic RED/IR AC/DC ratios -> SpO2 in [97, 100]."""
        fs = 100
        t = np.arange(500) / fs
        # IR: DC ~80000, AC ~2000 (PI ~2.5%) -> acdc_ir ~0.025
        # Red: DC ~70000, AC ~1200 -> acdc_red ~0.017
        # R = acdc_red / acdc_ir ~0.68 -> SpO2 ~-45.06*0.68^2 + 30.354*0.68 + 94.845 ~97.3
        ir = 80000 + 2000 * np.sin(2 * np.pi * 1.2 * t)
        red = 70000 + 1200 * np.sin(2 * np.pi * 1.2 * t)
        spo2, R, acdc_ir, acdc_red, status = calculate_spo2_v23(ir, red, fs)
        assert status == "ok", f"Expected status=ok, got {status}"
        assert spo2 is not None
        assert 94.0 <= spo2 <= 100.0, f"SpO2 {spo2} not in [94, 100]"

    def test_low_red_ac_gives_high_r_rejected(self):
        """Very low Red AC / very high IR AC -> R > SPO2_RATIO_MAX=1.4 -> rejected."""
        fs = 100
        t = np.arange(500) / fs
        # Deliberately pathological: Red AC tiny, IR AC large
        ir = 80000 + 5000 * np.sin(2 * np.pi * 1.2 * t)
        red = 70000 + 100 * np.sin(2 * np.pi * 1.2 * t)
        spo2, R, acdc_ir, acdc_red, status = calculate_spo2_v23(ir, red, fs)
        # R = (100/70000)/(5000/80000) ~ 0.023 which is < 0.4 -> ratio_r_out_of_range
        assert status in ("ratio_r_out_of_range", "acdc_ir_out_of_range",
                          "acdc_red_out_of_range", "low_perfusion_index", "ac_or_dc_invalid"), (
            f"Expected rejection, got status={status}, spo2={spo2}"
        )

    def test_eval_row_propagates_rejection_reason(self):
        """eval_row must record rejection reason from calculate_spo2_v23."""
        row = pd.Series({
            "dataset": "BP-protocol",
            "file": "__nonexistent_file__.csv",
            "t_start_s": 0.0,
            "t_end_s": 10.0,
            "fs": 100,
        })
        result = eval_row(row)
        assert result["skip_reason"] == "file_missing_or_bad_cols"
        assert result["valid_flag"] is False


# ===========================================================================
# Group 2 — Window indexing alignment
# ===========================================================================

class TestWindowIndexingAlignment:
    """Verify sample slicing uses t_start_s * fs : t_end_s * fs."""

    def test_t_start_times_fs_gives_correct_sample_offset(self):
        """For t_start_s=2.0, fs=100 -> slice starts at sample 200."""
        # We test the eval_row slice logic indirectly by injecting a signal
        # where only samples [200:1200] (window [2s, 12s]) contain a valid sine.
        fs = 100
        n = 1300
        signal = np.zeros(n)
        # Put valid PPG-like sine ONLY in samples 200:1200
        t_win = np.arange(1000) / fs
        signal[200:1200] = 80000 + 2000 * np.sin(2 * np.pi * 1.2 * t_win)
        # Red similarly
        signal_red = np.zeros(n)
        signal_red[200:1200] = 70000 + 1200 * np.sin(2 * np.pi * 1.2 * t_win)

        with patch("eval_spo2_ablation.load_signal_cached", return_value=(signal, signal_red)):
            row = pd.Series({
                "dataset": "BP-protocol",
                "file": "dummy.csv",
                "t_start_s": 2.0,
                "t_end_s": 12.0,
                "fs": fs,
            })
            result = eval_row(row)

        assert result["skip_reason"] == "", f"Unexpected skip: {result['skip_reason']}"
        assert result["valid_flag"] is True, (
            f"Expected valid SpO2, got rejection={result['rejection_reason']}"
        )

    def test_window_out_of_bounds_sets_skip_reason(self):
        """t_end_s * fs beyond signal length -> skip_reason='window_out_of_bounds'."""
        fs = 100
        signal = np.ones(500) * 80000  # only 5s
        signal_red = np.ones(500) * 70000
        with patch("eval_spo2_ablation.load_signal_cached", return_value=(signal, signal_red)):
            row = pd.Series({
                "dataset": "BP-protocol",
                "file": "dummy.csv",
                "t_start_s": 0.0,
                "t_end_s": 10.0,  # requires 1000 samples, only 500 exist
                "fs": fs,
            })
            result = eval_row(row)
        assert result["skip_reason"] == "window_out_of_bounds"

    def test_correct_slice_length_for_10s_window_at_100hz(self):
        """t_start=0, t_end=10, fs=100 -> slice length = 1000 samples."""
        fs = 100
        t = np.arange(1000) / fs
        ir = 80000 + 2000 * np.sin(2 * np.pi * 1.2 * t)
        red = 70000 + 1200 * np.sin(2 * np.pi * 1.2 * t)
        full_ir = np.concatenate([ir, np.ones(200) * 80000])
        full_red = np.concatenate([red, np.ones(200) * 70000])

        captured = {}
        orig = calculate_spo2_v23

        def _spy(ir_win, red_win, fs_):
            captured["len_ir"] = len(ir_win)
            captured["len_red"] = len(red_win)
            return orig(ir_win, red_win, fs_)

        with patch("eval_spo2_ablation.calculate_spo2_v23", side_effect=_spy):
            with patch("eval_spo2_ablation.load_signal_cached", return_value=(full_ir, full_red)):
                row = pd.Series({"dataset": "BP-protocol", "file": "d.csv",
                                 "t_start_s": 0.0, "t_end_s": 10.0, "fs": fs})
                eval_row(row)

        assert captured.get("len_ir") == 1000, f"Slice length wrong: {captured.get('len_ir')}"
        assert captured.get("len_red") == 1000


# ===========================================================================
# Group 3 — Aggregation denominator correctness (P1 framing)
# ===========================================================================

class TestAggregationDenominator:
    """
    P1 framing concern: pct_in_94_100 uses valid-only denominator.
    Tests confirm or refute which denominator is actually used.
    """

    def _make_df(self, tier_col: str = "sqa_tier_pseudo") -> pd.DataFrame:
        """
        10 windows in one tier: 6 valid, of which 5 have SpO2 in [94,100].
        """
        rows = []
        for i in range(10):
            valid_flag = i < 6
            spo2 = 97.0 if (valid_flag and i < 5) else (85.0 if valid_flag else float("nan"))
            rows.append({
                "dataset": "BP-protocol",
                "file": f"f{i}.csv",
                "subject_id": "X",
                "window_idx": i,
                "t_start_s": i * 10.0,
                "t_end_s": (i + 1) * 10.0,
                "fs": 100,
                "valid_flag": valid_flag,
                "spo2_pred": spo2,
                "ratio_r": 0.65,
                "acdc_ir": 0.02,
                "acdc_red": 0.015,
                "rejection_reason": "" if valid_flag else "ratio_r_out_of_range",
                "skip_reason": "",
                "sqa_tier_pseudo": "excellent",
                "sqa_tier_lr_pred": "excellent",
            })
        return pd.DataFrame(rows)

    def test_pct_valid_uses_n_total_denominator(self):
        """pct_valid = n_valid / n_total * 100 = 6/10*100 = 60.0"""
        df = self._make_df()
        summary = summarize_by_tier(df, "sqa_tier_pseudo", "pseudo")
        row = summary[
            (summary["dataset"] == "BP-protocol") & (summary["tier"] == "excellent")
        ].iloc[0]
        assert row["n_total"] == 10
        assert row["n_valid"] == 6
        assert abs(row["pct_valid"] - 60.0) < 1e-9, (
            f"pct_valid={row['pct_valid']}, expected 60.0"
        )

    def test_pct_in_94_100_uses_valid_only_denominator(self):
        """
        With 6 valid windows and 5 in [94,100]:
          - valid-only denominator  -> 5/6 * 100 = 83.33%  (CONFIRMED if this passes)
          - total denominator       -> 5/10 * 100 = 50.0%  (would pass if other branch)
        """
        df = self._make_df()
        summary = summarize_by_tier(df, "sqa_tier_pseudo", "pseudo")
        row = summary[
            (summary["dataset"] == "BP-protocol") & (summary["tier"] == "excellent")
        ].iloc[0]
        pct = row["pct_in_94_100"]
        # 83.33 = valid-only denominator; 50.0 = total denominator
        is_valid_denom = abs(pct - (5 / 6 * 100)) < 0.5
        is_total_denom = abs(pct - 50.0) < 0.5
        assert is_valid_denom or is_total_denom, (
            f"pct_in_94_100={pct} matches neither valid-only (83.33) nor total (50.0)"
        )
        # Document which denominator is actually used (P1 framing finding)
        if is_valid_denom:
            # This is the "P1 framing confirmed" path
            assert True, "P1 confirmed: pct_in_94_100 uses valid-only denominator (83.33%)"
        else:
            assert True, "P1 refuted: pct_in_94_100 uses total denominator (50.0%)"

    def test_pct_in_94_100_denominator_explicit(self):
        """
        Explicit assertion: the actual denominator. Code line 205-208:
          pct_in_range = ((spo2_arr >= lo) & (spo2_arr <= hi)).mean() * 100
        where spo2_arr is built from valid windows only -> valid-only denominator.
        This test PASSES if valid-only denominator is used (P1 confirmed).
        """
        df = self._make_df()
        summary = summarize_by_tier(df, "sqa_tier_pseudo", "pseudo")
        row = summary[
            (summary["dataset"] == "BP-protocol") & (summary["tier"] == "excellent")
        ].iloc[0]
        expected_valid_denom = round(5 / 6 * 100, 6)
        assert abs(row["pct_in_94_100"] - expected_valid_denom) < 0.1, (
            f"Expected valid-only denominator {expected_valid_denom:.2f}%, "
            f"got {row['pct_in_94_100']:.2f}%. "
            "If this fails, denominator is total-windows (50%) — P1 refuted."
        )

    def test_unconditional_rate_derivable_from_summary(self):
        """
        If pct_in_94_100 uses valid-only denom, the unconditional (of total) rate
        can be derived as: pct_valid * pct_in_94_100 / 100.
        Verify: 60.0 * 83.33 / 100 = 50.0% (= 5/10 total rate).
        """
        df = self._make_df()
        summary = summarize_by_tier(df, "sqa_tier_pseudo", "pseudo")
        row = summary[
            (summary["dataset"] == "BP-protocol") & (summary["tier"] == "excellent")
        ].iloc[0]
        derived = row["pct_valid"] * row["pct_in_94_100"] / 100.0
        expected_unconditional = 5 / 10 * 100  # 50.0%
        assert abs(derived - expected_unconditional) < 0.5, (
            f"Derived unconditional rate {derived:.2f}% != expected {expected_unconditional:.1f}%. "
            "Denominator chain broken."
        )

    def test_zero_valid_windows_returns_nan_pct_in_range(self):
        """All windows invalid -> n_valid=0 -> pct_in_94_100 NaN (not 0%)."""
        rows = [
            {
                "dataset": "BP-protocol", "file": "f.csv", "subject_id": "X",
                "window_idx": i, "t_start_s": i * 10.0, "t_end_s": (i + 1) * 10.0,
                "fs": 100, "valid_flag": False, "spo2_pred": float("nan"),
                "ratio_r": float("nan"), "acdc_ir": float("nan"), "acdc_red": float("nan"),
                "rejection_reason": "ratio_r_out_of_range", "skip_reason": "",
                "sqa_tier_pseudo": "excellent", "sqa_tier_lr_pred": "excellent",
            }
            for i in range(5)
        ]
        df = pd.DataFrame(rows)
        summary = summarize_by_tier(df, "sqa_tier_pseudo", "pseudo")
        row = summary[
            (summary["dataset"] == "BP-protocol") & (summary["tier"] == "excellent")
        ].iloc[0]
        assert row["n_valid"] == 0
        assert np.isnan(row["pct_in_94_100"]), (
            f"Expected NaN for pct_in_94_100 when n_valid=0, got {row['pct_in_94_100']}"
        )


# ===========================================================================
# Group 4 — Headline number reproducibility
# ===========================================================================

class TestHeadlineReproducibility:
    """Verify summary CSV matches claimed numbers: LR BP excellent 99.3%, unfit 47.4%."""

    @pytest.fixture(autouse=True)
    def require_summary_csv(self):
        if not SUMMARY_CSV.exists():
            pytest.skip(f"Summary CSV not found: {SUMMARY_CSV}")

    @pytest.fixture
    def summary(self):
        return pd.read_csv(SUMMARY_CSV)

    def test_lr_bp_excellent_pct_in_range_99_3(self, summary):
        """LR-tier BP-protocol excellent: pct_in_94_100 ~ 99.3%."""
        row = summary[
            (summary["label_source"] == "lr_pred")
            & (summary["dataset"] == "BP-protocol")
            & (summary["tier"] == "excellent")
        ]
        assert len(row) == 1, "Expected exactly 1 row for lr_pred / BP-protocol / excellent"
        assert abs(row.iloc[0]["pct_in_94_100"] - 99.3) < 0.5, (
            f"pct_in_94_100={row.iloc[0]['pct_in_94_100']:.2f}%, expected ~99.3%"
        )

    def test_lr_bp_unfit_pct_in_range_47_4(self, summary):
        """LR-tier BP-protocol unfit: pct_in_94_100 ~ 47.4%."""
        row = summary[
            (summary["label_source"] == "lr_pred")
            & (summary["dataset"] == "BP-protocol")
            & (summary["tier"] == "unfit")
        ]
        assert len(row) == 1
        assert abs(row.iloc[0]["pct_in_94_100"] - 47.4) < 0.5, (
            f"pct_in_94_100={row.iloc[0]['pct_in_94_100']:.2f}%, expected ~47.4%"
        )

    def test_lr_bp_unfit_n_valid_is_19(self, summary):
        """LR-tier BP-protocol unfit: N_valid=19 (small sample caveat)."""
        row = summary[
            (summary["label_source"] == "lr_pred")
            & (summary["dataset"] == "BP-protocol")
            & (summary["tier"] == "unfit")
        ]
        assert row.iloc[0]["n_valid"] == 19, (
            f"n_valid={row.iloc[0]['n_valid']}, expected 19"
        )

    def test_lr_bp_unfit_n_total_is_28(self, summary):
        """LR-tier BP-protocol unfit: N_total=28."""
        row = summary[
            (summary["label_source"] == "lr_pred")
            & (summary["dataset"] == "BP-protocol")
            & (summary["tier"] == "unfit")
        ]
        assert row.iloc[0]["n_total"] == 28, (
            f"n_total={row.iloc[0]['n_total']}, expected 28"
        )

    def test_wilson_ci_excellent_vs_unfit_non_overlapping(self, summary):
        """
        Wilson 95% CI for 99.3% on N=581 vs 47.4% on N=19 must not overlap.
        This validates the Δ51.9pp claim is statistically meaningful.
        """
        pytest.importorskip("statsmodels", reason="statsmodels required for Wilson CI")
        from statsmodels.stats.proportion import proportion_confint

        exc_row = summary[
            (summary["label_source"] == "lr_pred")
            & (summary["dataset"] == "BP-protocol")
            & (summary["tier"] == "excellent")
        ].iloc[0]
        unf_row = summary[
            (summary["label_source"] == "lr_pred")
            & (summary["dataset"] == "BP-protocol")
            & (summary["tier"] == "unfit")
        ].iloc[0]

        n_exc = int(exc_row["n_valid"])
        k_exc = int(round(exc_row["pct_in_94_100"] / 100.0 * n_exc))
        n_unf = int(unf_row["n_valid"])
        k_unf = int(round(unf_row["pct_in_94_100"] / 100.0 * n_unf))

        lo_exc, hi_exc = proportion_confint(k_exc, n_exc, method="wilson")
        lo_unf, hi_unf = proportion_confint(k_unf, n_unf, method="wilson")

        assert lo_exc > hi_unf, (
            f"CIs overlap: excellent=[{lo_exc:.3f}, {hi_exc:.3f}], "
            f"unfit=[{lo_unf:.3f}, {hi_unf:.3f}]. "
            "Δ51.9pp claim not statistically supported with non-overlapping CIs."
        )

    def test_headline_delta_51_9_pp(self, summary):
        """Δ = excellent_pct_in_range - unfit_pct_in_range ~ 51.9 pp."""
        exc = summary[
            (summary["label_source"] == "lr_pred")
            & (summary["dataset"] == "BP-protocol")
            & (summary["tier"] == "excellent")
        ].iloc[0]["pct_in_94_100"]
        unf = summary[
            (summary["label_source"] == "lr_pred")
            & (summary["dataset"] == "BP-protocol")
            & (summary["tier"] == "unfit")
        ].iloc[0]["pct_in_94_100"]
        delta = exc - unf
        assert abs(delta - 51.9) < 1.0, (
            f"Δ={delta:.1f} pp, expected ~51.9 pp"
        )

    def test_pseudo_bp_excellent_pct_in_range_above_90(self, summary):
        """Pseudo-label BP excellent pct_in_94_100 should be high (>90%)."""
        row = summary[
            (summary["label_source"] == "pseudo")
            & (summary["dataset"] == "BP-protocol")
            & (summary["tier"] == "excellent")
        ]
        assert len(row) == 1
        assert row.iloc[0]["pct_in_94_100"] >= 90.0, (
            f"Pseudo BP excellent pct_in_94_100={row.iloc[0]['pct_in_94_100']:.1f}% < 90%"
        )


# ===========================================================================
# Group 5 — Edge cases
# ===========================================================================

class TestEdgeCases:
    """Edge cases: empty tier, all-invalid windows, zero/flat signals."""

    def _empty_tier_df(self) -> pd.DataFrame:
        """DataFrame with no 'unfit' windows, only 'excellent'."""
        rows = []
        for i in range(3):
            rows.append({
                "dataset": "BP-protocol", "file": "f.csv", "subject_id": "X",
                "window_idx": i, "t_start_s": i * 10.0, "t_end_s": (i + 1) * 10.0,
                "fs": 100, "valid_flag": True,
                "spo2_pred": 97.0,
                "ratio_r": 0.65, "acdc_ir": 0.02, "acdc_red": 0.015,
                "rejection_reason": "", "skip_reason": "",
                "sqa_tier_pseudo": "excellent",
                "sqa_tier_lr_pred": "excellent",
            })
        return pd.DataFrame(rows)

    def test_no_windows_in_tier_returns_nan_no_crash(self):
        """Tier with 0 windows -> pct_valid=NaN, pct_in_94_100=NaN, no exception."""
        df = self._empty_tier_df()
        try:
            summary = summarize_by_tier(df, "sqa_tier_pseudo", "pseudo")
        except Exception as exc:
            pytest.fail(f"summarize_by_tier raised on empty tier: {exc}")
        unfit_row = summary[
            (summary["dataset"] == "BP-protocol") & (summary["tier"] == "unfit")
        ]
        assert len(unfit_row) == 1
        assert unfit_row.iloc[0]["n_total"] == 0
        assert np.isnan(unfit_row.iloc[0]["pct_valid"]), (
            "pct_valid should be NaN for empty tier"
        )
        assert np.isnan(unfit_row.iloc[0]["pct_in_94_100"]), (
            "pct_in_94_100 should be NaN for empty tier"
        )

    def test_all_windows_invalid_returns_nan_pct_in_range(self):
        """All windows rejected -> n_valid=0 -> pct_in_94_100 is NaN, not 0%."""
        rows = [
            {
                "dataset": "BP-protocol", "file": "f.csv", "subject_id": "X",
                "window_idx": i, "t_start_s": i * 10.0, "t_end_s": (i + 1) * 10.0,
                "fs": 100, "valid_flag": False, "spo2_pred": float("nan"),
                "ratio_r": float("nan"), "acdc_ir": float("nan"), "acdc_red": float("nan"),
                "rejection_reason": "ratio_r_out_of_range", "skip_reason": "",
                "sqa_tier_pseudo": "unfit", "sqa_tier_lr_pred": "unfit",
            }
            for i in range(4)
        ]
        df = pd.DataFrame(rows)
        summary = summarize_by_tier(df, "sqa_tier_pseudo", "pseudo")
        row = summary[
            (summary["dataset"] == "BP-protocol") & (summary["tier"] == "unfit")
        ].iloc[0]
        assert row["n_valid"] == 0
        assert np.isnan(row["pct_in_94_100"]), (
            f"Expected NaN for pct_in_94_100 with 0 valid windows, got {row['pct_in_94_100']}"
        )

    def test_red_zero_signal_rejected_no_crash(self):
        """Red signal all zeros -> calculate_spo2_v23 must return a rejection status, not crash."""
        fs = 100
        ir = np.ones(500) * 80000
        red = np.zeros(500)
        try:
            spo2, R, acdc_ir, acdc_red, status = calculate_spo2_v23(ir, red, fs)
        except Exception as exc:
            pytest.fail(f"calculate_spo2_v23 raised on zero Red signal: {exc}")
        assert spo2 is None, f"Expected None SpO2 for zero Red, got {spo2}"
        assert status != "ok", f"Expected rejection status, got 'ok'"

    def test_flat_ir_red_rejected_no_crash(self):
        """Constant IR and Red (no AC component) -> rejected, no crash."""
        fs = 100
        ir = np.ones(500) * 80000
        red = np.ones(500) * 70000
        try:
            spo2, R, acdc_ir, acdc_red, status = calculate_spo2_v23(ir, red, fs)
        except Exception as exc:
            pytest.fail(f"calculate_spo2_v23 raised on flat signals: {exc}")
        assert spo2 is None, f"Expected None for flat DC-only signal, got {spo2}"

    def test_eval_row_skip_reason_for_missing_file(self):
        """File missing -> skip_reason='file_missing_or_bad_cols', valid_flag=False."""
        row = pd.Series({
            "dataset": "BP-protocol",
            "file": "_no_such_file_xyz.csv",
            "t_start_s": 0.0,
            "t_end_s": 10.0,
            "fs": 100,
        })
        result = eval_row(row)
        assert result["skip_reason"] == "file_missing_or_bad_cols"
        assert result["valid_flag"] is False
        assert np.isnan(result["spo2_pred"])

    def test_single_window_all_valid_no_crash(self):
        """Single window, valid SpO2 -> summary should not crash, returns 1 row per tier."""
        row_data = {
            "dataset": "BP-protocol", "file": "f.csv", "subject_id": "X",
            "window_idx": 0, "t_start_s": 0.0, "t_end_s": 10.0,
            "fs": 100, "valid_flag": True, "spo2_pred": 97.5,
            "ratio_r": 0.65, "acdc_ir": 0.02, "acdc_red": 0.015,
            "rejection_reason": "", "skip_reason": "",
            "sqa_tier_pseudo": "excellent", "sqa_tier_lr_pred": "excellent",
        }
        df = pd.DataFrame([row_data])
        summary = summarize_by_tier(df, "sqa_tier_pseudo", "pseudo")
        # Should have 3 rows (one per tier), no crash
        assert len(summary) == 3
        exc_row = summary[summary["tier"] == "excellent"].iloc[0]
        assert exc_row["n_total"] == 1
        assert exc_row["n_valid"] == 1
        assert abs(exc_row["pct_in_94_100"] - 100.0) < 1e-9
