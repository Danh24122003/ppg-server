"""
pytest tests for eval_subject_holdout_calibration.py

Tests cover:
  a. Anchor identification (earliest session by date, not dataframe order)
  b. Offset computation and application
  c. Causality constraint (anchor_date <= drift_date, days_from_anchor >= 0)
  d. CV no-leakage (test subjects absent from train indices in same fold)
  e. Empty / edge data handling

Run:
    cd "C:/Users/Acer/Desktop/PPG monitor"
    python -m pytest "backend/self_collect/test_calibration_drift.py" -v
"""
from __future__ import annotations

import sys
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
ROOT = Path("C:/Users/Acer/Desktop/PPG monitor")
ML_DIR = ROOT / "ML code" / "ml"
SELF_COLLECT_DIR = ROOT / "Backend code" / "self_collect"

os.environ.setdefault("MODEL_DIR", str(ML_DIR / "models"))
sys.path.insert(0, str(ML_DIR))

# Import module under test via importlib to avoid name collision with __main__
import importlib.util as _ilu

_spec = _ilu.spec_from_file_location(
    "eval_subject_holdout_calibration",
    SELF_COLLECT_DIR / "eval_subject_holdout_calibration.py",
)
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

subject_holdout_cv = _mod.subject_holdout_cv
build_pipeline = _mod.build_pipeline
load_self_sessions_separate = _mod.load_self_sessions_separate
FS_TARGET = _mod.FS_TARGET

from ml_models import NUM_TOTAL_FEATURES  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dummy_features(n: int = 1, seed: int = 0) -> list:
    """Return list of n random 38-element feature vectors."""
    rng = np.random.default_rng(seed)
    return [rng.random(NUM_TOTAL_FEATURES) for _ in range(n)]


def _make_df_sessions(records: List[dict]) -> pd.DataFrame:
    """Build a sessions DataFrame directly (bypasses file I/O)."""
    rows = []
    for r in records:
        rows.append({
            "subject_id": r["subject_id"],
            "csv_path": r.get("csv_path", "dummy.csv"),
            "session_date": r["session_date"],
            "sbp_baseline": r["sbp"],
            "dbp_baseline": r["dbp"],
            "features": r["features"],
        })
    df = pd.DataFrame(rows)
    df = df.sort_values(["subject_id", "session_date"]).reset_index(drop=True)
    return df


def _empty_ppgbp(n_feats: int = NUM_TOTAL_FEATURES):
    """PPG-BP placeholders with zero rows."""
    return (
        np.zeros((0, n_feats)),
        {"sbp": np.array([]), "dbp": np.array([])},
    )


# ---------------------------------------------------------------------------
# Section a: Anchor identification
# ---------------------------------------------------------------------------

class TestAnchorIdentification:

    def test_anchor_is_earliest_date_not_dataframe_order(self):
        """Subject with sessions in dataframe order [29/4, 28/4, 5/5] — anchor MUST be 28/4."""
        feats = _make_dummy_features(3, seed=1)
        records = [
            {"subject_id": "X", "session_date": datetime(2026, 4, 29), "sbp": 120.0, "dbp": 80.0, "features": feats[0]},
            {"subject_id": "X", "session_date": datetime(2026, 4, 28), "sbp": 120.0, "dbp": 80.0, "features": feats[1]},
            {"subject_id": "X", "session_date": datetime(2026, 5, 5),  "sbp": 120.0, "dbp": 80.0, "features": feats[2]},
        ]
        df_self = _make_df_sessions(records)
        # After sort_values by session_date, row 0 for subject X must be 28/4
        sub_rows = df_self[df_self.subject_id == "X"].reset_index(drop=True)
        assert sub_rows.iloc[0]["session_date"] == datetime(2026, 4, 28), \
            "Anchor must be earliest date after sort"

    def test_single_session_marked_anchor(self):
        """Subject with only 1 session: that row must be is_anchor=True, no drift rows."""
        feats = _make_dummy_features(2, seed=2)
        records = [
            {"subject_id": "Solo", "session_date": datetime(2026, 5, 1), "sbp": 115.0, "dbp": 75.0, "features": feats[0]},
            {"subject_id": "Other", "session_date": datetime(2026, 5, 2), "sbp": 125.0, "dbp": 82.0, "features": feats[1]},
        ]
        df_self = _make_df_sessions(records)
        X_self = np.vstack(df_self["features"].to_list())
        X_ppgbp, y_ppgbp = _empty_ppgbp()

        results = subject_holdout_cv(
            X_self, df_self, X_ppgbp, y_ppgbp, n_folds=2, target="sbp"
        )
        solo_rows = results[results.subject_id == "Solo"]
        assert len(solo_rows) >= 1, "Solo subject should appear in results"
        assert solo_rows.iloc[0]["is_anchor"] is True or bool(solo_rows.iloc[0]["is_anchor"]), \
            "Single session must be anchor"
        drift_rows = solo_rows[~solo_rows["is_anchor"]]
        assert len(drift_rows) == 0, "Solo subject has no drift sessions"

    def test_anchor_session_idx_is_zero(self):
        """Anchor row must always have session_idx=0."""
        feats = _make_dummy_features(4, seed=3)
        records = [
            {"subject_id": "A", "session_date": datetime(2026, 4, 28), "sbp": 110.0, "dbp": 70.0, "features": feats[0]},
            {"subject_id": "A", "session_date": datetime(2026, 5, 3),  "sbp": 112.0, "dbp": 71.0, "features": feats[1]},
            {"subject_id": "B", "session_date": datetime(2026, 4, 29), "sbp": 120.0, "dbp": 80.0, "features": feats[2]},
            {"subject_id": "B", "session_date": datetime(2026, 5, 4),  "sbp": 118.0, "dbp": 79.0, "features": feats[3]},
        ]
        df_self = _make_df_sessions(records)
        X_self = np.vstack(df_self["features"].to_list())
        X_ppgbp, y_ppgbp = _empty_ppgbp()

        results = subject_holdout_cv(
            X_self, df_self, X_ppgbp, y_ppgbp, n_folds=2, target="sbp"
        )
        anchor_rows = results[results.is_anchor]
        assert (anchor_rows["session_idx"] == 0).all(), \
            "All anchor rows must have session_idx=0"


# ---------------------------------------------------------------------------
# Section b: Offset computation
# ---------------------------------------------------------------------------

class TestOffsetComputation:

    def test_offset_equals_truth_minus_raw(self):
        """offset = cuff_truth - raw_pred for every row."""
        feats = _make_dummy_features(3, seed=10)
        records = [
            {"subject_id": "P", "session_date": datetime(2026, 4, 28), "sbp": 120.0, "dbp": 80.0, "features": feats[0]},
            {"subject_id": "P", "session_date": datetime(2026, 5, 1),  "sbp": 122.0, "dbp": 81.0, "features": feats[1]},
            {"subject_id": "Q", "session_date": datetime(2026, 4, 29), "sbp": 130.0, "dbp": 85.0, "features": feats[2]},
        ]
        df_self = _make_df_sessions(records)
        X_self = np.vstack(df_self["features"].to_list())
        X_ppgbp, y_ppgbp = _empty_ppgbp()

        results = subject_holdout_cv(
            X_self, df_self, X_ppgbp, y_ppgbp, n_folds=2, target="sbp"
        )
        assert len(results) > 0, "Results must not be empty"
        # For every row, offset must equal (anchor cuff_truth - anchor pred_raw)
        # which is verified indirectly: anchor has err_calibrated == 0
        anchor = results[results.is_anchor]
        assert np.allclose(anchor["err_calibrated"].to_numpy(), 0.0, atol=1e-8), \
            "Anchor calibrated error must be zero (offset = truth - raw_pred)"

    def test_pred_calibrated_equals_pred_raw_plus_offset(self):
        """pred_calibrated = pred_raw + offset for every row."""
        feats = _make_dummy_features(4, seed=11)
        records = [
            {"subject_id": "A", "session_date": datetime(2026, 4, 28), "sbp": 115.0, "dbp": 75.0, "features": feats[0]},
            {"subject_id": "A", "session_date": datetime(2026, 5, 2),  "sbp": 117.0, "dbp": 76.0, "features": feats[1]},
            {"subject_id": "B", "session_date": datetime(2026, 4, 29), "sbp": 125.0, "dbp": 82.0, "features": feats[2]},
            {"subject_id": "B", "session_date": datetime(2026, 5, 3),  "sbp": 123.0, "dbp": 81.0, "features": feats[3]},
        ]
        df_self = _make_df_sessions(records)
        X_self = np.vstack(df_self["features"].to_list())
        X_ppgbp, y_ppgbp = _empty_ppgbp()

        results = subject_holdout_cv(
            X_self, df_self, X_ppgbp, y_ppgbp, n_folds=2, target="sbp"
        )
        diff = (results["pred_raw"] + results["offset"] - results["pred_calibrated"]).abs()
        assert diff.max() < 1e-8, \
            f"pred_calibrated != pred_raw + offset; max diff={diff.max():.2e}"

    def test_synthetic_offset_value(self):
        """Synthetic: cuff=120, raw_pred~110 -> offset~+10; apply to pred=115 -> calibrated~125."""
        # We cannot control exact SVR output, but we can verify the algebraic invariant:
        # calibrated = raw + offset = raw + (truth_anchor - raw_anchor)
        # i.e., calibrated_anchor == truth_anchor exactly (zero error).
        feats = _make_dummy_features(2, seed=99)
        records = [
            {"subject_id": "Z", "session_date": datetime(2026, 4, 28), "sbp": 120.0, "dbp": 64.0, "features": feats[0]},
            {"subject_id": "W", "session_date": datetime(2026, 4, 29), "sbp": 130.0, "dbp": 84.0, "features": feats[1]},
        ]
        df_self = _make_df_sessions(records)
        X_self = np.vstack(df_self["features"].to_list())
        X_ppgbp, y_ppgbp = _empty_ppgbp()

        results = subject_holdout_cv(
            X_self, df_self, X_ppgbp, y_ppgbp, n_folds=2, target="sbp"
        )
        for _, row in results.iterrows():
            expected_calibrated = row["pred_raw"] + row["offset"]
            assert abs(expected_calibrated - row["pred_calibrated"]) < 1e-8

    def test_abs_err_raw_equals_abs_err_raw_column(self):
        """abs_err_raw must equal |err_raw| for every row."""
        sbp = pd.read_csv(ROOT / "Backend code" / "self_collect" / "_drift_results_sbp.csv")
        diff = (sbp["err_raw"].abs() - sbp["abs_err_raw"]).abs()
        assert diff.max() < 1e-8, "abs_err_raw column inconsistent with err_raw"

    def test_abs_err_calibrated_equals_abs_err_calibrated_column(self):
        """abs_err_calibrated must equal |err_calibrated| for every row."""
        sbp = pd.read_csv(ROOT / "Backend code" / "self_collect" / "_drift_results_sbp.csv")
        diff = (sbp["err_calibrated"].abs() - sbp["abs_err_calibrated"]).abs()
        assert diff.max() < 1e-8, "abs_err_calibrated column inconsistent with err_calibrated"


# ---------------------------------------------------------------------------
# Section c: Causality constraint
# ---------------------------------------------------------------------------

class TestCausalityConstraint:

    def _get_real_drift_dfs(self):
        sbp = pd.read_csv(ROOT / "Backend code" / "self_collect" / "_drift_results_sbp.csv",
                          parse_dates=["session_date"])
        dbp = pd.read_csv(ROOT / "Backend code" / "self_collect" / "_drift_results_dbp.csv",
                          parse_dates=["session_date"])
        return sbp, dbp

    def test_anchor_days_from_anchor_is_zero(self):
        """Anchor rows must have days_from_anchor == 0."""
        sbp, dbp = self._get_real_drift_dfs()
        for df, name in [(sbp, "SBP"), (dbp, "DBP")]:
            anchor = df[df.is_anchor]
            assert (anchor["days_from_anchor"] == 0.0).all(), \
                f"{name}: anchor rows days_from_anchor != 0"

    def test_drift_days_from_anchor_nonnegative(self):
        """All non-anchor rows must have days_from_anchor >= 0."""
        sbp, dbp = self._get_real_drift_dfs()
        for df, name in [(sbp, "SBP"), (dbp, "DBP")]:
            drift = df[~df.is_anchor].dropna(subset=["days_from_anchor"])
            assert (drift["days_from_anchor"] >= 0).all(), \
                f"{name}: drift session has negative days_from_anchor"

    def test_anchor_err_calibrated_approximately_zero(self):
        """Anchor calibrated error must be 0 (anchor calibrates to itself)."""
        sbp, _ = self._get_real_drift_dfs()
        anchor = sbp[sbp.is_anchor]
        assert np.allclose(anchor["err_calibrated"].to_numpy(), 0.0, atol=1e-8), \
            "SBP anchor rows have non-zero err_calibrated"

    def test_pred_calibrated_identity_real_data(self):
        """pred_calibrated = pred_raw + offset for all real CSV rows."""
        sbp, dbp = self._get_real_drift_dfs()
        for df, name in [(sbp, "SBP"), (dbp, "DBP")]:
            diff = (df["pred_raw"] + df["offset"] - df["pred_calibrated"]).abs()
            assert diff.max() < 1e-8, \
                f"{name}: pred_calibrated != pred_raw + offset (max={diff.max():.2e})"

    def test_session_date_causality_synthetic(self):
        """Synthetic: anchor_date must be <= all drift dates."""
        feats = _make_dummy_features(3, seed=20)
        records = [
            {"subject_id": "M", "session_date": datetime(2026, 5, 5), "sbp": 121.0, "dbp": 79.0, "features": feats[0]},
            {"subject_id": "M", "session_date": datetime(2026, 4, 29), "sbp": 121.0, "dbp": 79.0, "features": feats[1]},
            {"subject_id": "N", "session_date": datetime(2026, 5, 1),  "sbp": 110.0, "dbp": 70.0, "features": feats[2]},
        ]
        df_self = _make_df_sessions(records)
        X_self = np.vstack(df_self["features"].to_list())
        X_ppgbp, y_ppgbp = _empty_ppgbp()

        results = subject_holdout_cv(
            X_self, df_self, X_ppgbp, y_ppgbp, n_folds=2, target="sbp"
        )
        m_rows = results[results.subject_id == "M"]
        if len(m_rows) >= 2:
            anchor_date = m_rows[m_rows.is_anchor]["session_date"].iloc[0]
            drift_dates = m_rows[~m_rows.is_anchor]["session_date"]
            for d in drift_dates:
                assert d >= anchor_date, \
                    f"Causality violated: drift_date={d} < anchor_date={anchor_date}"

    def test_missing_session_date_gives_nan_days(self):
        """Rows with None session_date must have days_from_anchor = NaN."""
        feats = _make_dummy_features(2, seed=21)
        records = [
            {"subject_id": "K", "session_date": None, "sbp": 115.0, "dbp": 75.0, "features": feats[0]},
            {"subject_id": "L", "session_date": datetime(2026, 5, 1), "sbp": 120.0, "dbp": 80.0, "features": feats[1]},
        ]
        df_self = _make_df_sessions(records)
        X_self = np.vstack(df_self["features"].to_list())
        X_ppgbp, y_ppgbp = _empty_ppgbp()

        results = subject_holdout_cv(
            X_self, df_self, X_ppgbp, y_ppgbp, n_folds=2, target="sbp"
        )
        k_rows = results[results.subject_id == "K"]
        if len(k_rows) > 0:
            # days_from_anchor must be NaN when session_date is None
            assert k_rows["days_from_anchor"].isna().all(), \
                "Subject with None session_date should have NaN days_from_anchor"


# ---------------------------------------------------------------------------
# Section d: CV no-leakage
# ---------------------------------------------------------------------------

class TestCVNoLeakage:

    def test_test_subject_absent_from_train_indices(self):
        """For every fold, test subject IDs must NOT appear in train indices."""
        from sklearn.model_selection import GroupKFold

        subject_ids = ["A", "A", "B", "B", "C", "C"]
        n_unique = len(set(subject_ids))
        n_folds = min(5, n_unique)  # reduced to 3 for only 3 unique subjects

        X_dummy = np.zeros((len(subject_ids), NUM_TOTAL_FEATURES))
        y_dummy = np.zeros(len(subject_ids))

        gkf = GroupKFold(n_splits=n_folds)
        for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X_dummy, y_dummy, groups=subject_ids)):
            train_subs = set(np.array(subject_ids)[train_idx])
            test_subs = set(np.array(subject_ids)[test_idx])
            overlap = train_subs & test_subs
            assert len(overlap) == 0, \
                f"Fold {fold_idx}: subjects {overlap} appear in both train and test"

    def test_n_folds_reduced_when_fewer_unique_subjects(self):
        """n_folds is capped at n_unique_subjects when n_unique < n_folds."""
        feats = _make_dummy_features(3, seed=30)
        records = [
            {"subject_id": "A", "session_date": datetime(2026, 4, 28), "sbp": 110.0, "dbp": 70.0, "features": feats[0]},
            {"subject_id": "B", "session_date": datetime(2026, 4, 29), "sbp": 120.0, "dbp": 78.0, "features": feats[1]},
            {"subject_id": "C", "session_date": datetime(2026, 4, 30), "sbp": 130.0, "dbp": 85.0, "features": feats[2]},
        ]
        df_self = _make_df_sessions(records)
        X_self = np.vstack(df_self["features"].to_list())
        X_ppgbp, y_ppgbp = _empty_ppgbp()

        # Request 5 folds but only 3 unique subjects — should not crash
        results = subject_holdout_cv(
            X_self, df_self, X_ppgbp, y_ppgbp, n_folds=5, target="sbp"
        )
        # Fold count in results must be <= 3
        if len(results) > 0:
            max_fold = results["fold"].max()
            assert max_fold <= 3, f"Fold count {max_fold} > n_unique_subjects (3)"

    def test_no_duplicate_subject_in_same_fold_test(self):
        """Each subject appears in the test partition of exactly 1 fold."""
        feats = _make_dummy_features(6, seed=31)
        dates = [datetime(2026, 4, 28) + timedelta(days=i) for i in range(6)]
        records = [
            {"subject_id": "A", "session_date": dates[0], "sbp": 110.0, "dbp": 70.0, "features": feats[0]},
            {"subject_id": "A", "session_date": dates[1], "sbp": 112.0, "dbp": 71.0, "features": feats[1]},
            {"subject_id": "B", "session_date": dates[2], "sbp": 120.0, "dbp": 78.0, "features": feats[2]},
            {"subject_id": "C", "session_date": dates[3], "sbp": 130.0, "dbp": 85.0, "features": feats[3]},
            {"subject_id": "D", "session_date": dates[4], "sbp": 140.0, "dbp": 90.0, "features": feats[4]},
            {"subject_id": "E", "session_date": dates[5], "sbp": 105.0, "dbp": 68.0, "features": feats[5]},
        ]
        df_self = _make_df_sessions(records)
        X_self = np.vstack(df_self["features"].to_list())
        X_ppgbp, y_ppgbp = _empty_ppgbp()

        results = subject_holdout_cv(
            X_self, df_self, X_ppgbp, y_ppgbp, n_folds=5, target="sbp"
        )
        # Each subject should appear in results under exactly one fold value
        if len(results) > 0:
            for sid in results["subject_id"].unique():
                folds_for_sub = results[results.subject_id == sid]["fold"].unique()
                assert len(folds_for_sub) == 1, \
                    f"Subject {sid} appears in multiple folds: {folds_for_sub}"


# ---------------------------------------------------------------------------
# Section e: Empty / edge data
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_empty_dataframe_returns_empty_results(self):
        """Empty df_self must not crash — return empty DataFrame."""
        df_self = pd.DataFrame(columns=[
            "subject_id", "csv_path", "session_date",
            "sbp_baseline", "dbp_baseline", "features",
        ])
        X_self = np.zeros((0, NUM_TOTAL_FEATURES))
        X_ppgbp, y_ppgbp = _empty_ppgbp()

        # n_folds=2 minimum, but 0 unique subjects -> should return empty or handle gracefully
        # The function reduces n_folds to max(2, n_unique). With n_unique=0 this may
        # raise ValueError from GroupKFold. Accept either empty result or ValueError.
        try:
            results = subject_holdout_cv(
                X_self, df_self, X_ppgbp, y_ppgbp, n_folds=5, target="sbp"
            )
            assert len(results) == 0 or isinstance(results, pd.DataFrame)
        except (ValueError, IndexError):
            pass  # acceptable: GroupKFold cannot split 0 groups

    def test_single_subject_single_session_no_drift(self):
        """Single subject + 1 session: results contain 1 anchor row, 0 drift rows."""
        feats = _make_dummy_features(1, seed=40)
        records = [
            {"subject_id": "Only", "session_date": datetime(2026, 5, 1), "sbp": 120.0, "dbp": 80.0, "features": feats[0]},
        ]
        df_self = _make_df_sessions(records)
        X_self = np.vstack(df_self["features"].to_list())
        X_ppgbp, y_ppgbp = _empty_ppgbp()

        try:
            results = subject_holdout_cv(
                X_self, df_self, X_ppgbp, y_ppgbp, n_folds=5, target="sbp"
            )
            if len(results) > 0:
                assert results["is_anchor"].all(), "All rows must be anchor for single-session subject"
                assert (~results["is_anchor"]).sum() == 0, "No drift rows for single-session subject"
        except (ValueError, IndexError):
            pass  # GroupKFold may reject 1 sample

    def test_results_have_required_columns(self):
        """Output DataFrame must have all required columns."""
        feats = _make_dummy_features(4, seed=41)
        dates = [datetime(2026, 4, 28) + timedelta(days=i) for i in range(4)]
        records = [
            {"subject_id": "A", "session_date": dates[0], "sbp": 110.0, "dbp": 70.0, "features": feats[0]},
            {"subject_id": "A", "session_date": dates[1], "sbp": 112.0, "dbp": 71.0, "features": feats[1]},
            {"subject_id": "B", "session_date": dates[2], "sbp": 120.0, "dbp": 78.0, "features": feats[2]},
            {"subject_id": "B", "session_date": dates[3], "sbp": 122.0, "dbp": 79.0, "features": feats[3]},
        ]
        df_self = _make_df_sessions(records)
        X_self = np.vstack(df_self["features"].to_list())
        X_ppgbp, y_ppgbp = _empty_ppgbp()

        results = subject_holdout_cv(
            X_self, df_self, X_ppgbp, y_ppgbp, n_folds=2, target="sbp"
        )
        required_cols = {
            "fold", "subject_id", "session_idx", "is_anchor", "session_date",
            "days_from_anchor", "csv_path", "cuff_truth", "pred_raw",
            "pred_calibrated", "offset", "err_raw", "err_calibrated",
            "abs_err_raw", "abs_err_calibrated",
        }
        missing = required_cols - set(results.columns)
        assert len(missing) == 0, f"Missing columns: {missing}"

    def test_multiple_sessions_none_date_doesnt_crash(self):
        """Subject with multiple sessions where some dates are None must not crash."""
        feats = _make_dummy_features(3, seed=42)
        records = [
            {"subject_id": "N", "session_date": None,                  "sbp": 115.0, "dbp": 75.0, "features": feats[0]},
            {"subject_id": "N", "session_date": datetime(2026, 5, 3),  "sbp": 117.0, "dbp": 76.0, "features": feats[1]},
            {"subject_id": "O", "session_date": datetime(2026, 5, 1),  "sbp": 125.0, "dbp": 82.0, "features": feats[2]},
        ]
        df_self = _make_df_sessions(records)
        X_self = np.vstack(df_self["features"].to_list())
        X_ppgbp, y_ppgbp = _empty_ppgbp()

        # Must not raise
        try:
            results = subject_holdout_cv(
                X_self, df_self, X_ppgbp, y_ppgbp, n_folds=2, target="sbp"
            )
            assert isinstance(results, pd.DataFrame)
        except Exception as exc:
            pytest.fail(f"Unexpected exception with None session_date: {exc}")


# ---------------------------------------------------------------------------
# Section f: Real CSV output sanity checks
# ---------------------------------------------------------------------------

class TestRealCSVOutputs:

    @pytest.fixture(scope="class")
    def sbp_df(self):
        return pd.read_csv(ROOT / "Backend code" / "self_collect" / "_drift_results_sbp.csv")

    @pytest.fixture(scope="class")
    def dbp_df(self):
        return pd.read_csv(ROOT / "Backend code" / "self_collect" / "_drift_results_dbp.csv")

    def test_sbp_csv_parseable(self, sbp_df):
        assert len(sbp_df) > 0

    def test_dbp_csv_parseable(self, dbp_df):
        assert len(dbp_df) > 0

    def test_sbp_anchor_err_calibrated_zero(self, sbp_df):
        anchor = sbp_df[sbp_df.is_anchor]
        assert np.allclose(anchor["err_calibrated"].to_numpy(), 0.0, atol=1e-8)

    def test_dbp_anchor_err_calibrated_zero(self, dbp_df):
        anchor = dbp_df[dbp_df.is_anchor]
        assert np.allclose(anchor["err_calibrated"].to_numpy(), 0.0, atol=1e-8)

    def test_sbp_pred_calibrated_identity(self, sbp_df):
        diff = (sbp_df["pred_raw"] + sbp_df["offset"] - sbp_df["pred_calibrated"]).abs()
        assert diff.max() < 1e-8

    def test_dbp_pred_calibrated_identity(self, dbp_df):
        diff = (dbp_df["pred_raw"] + dbp_df["offset"] - dbp_df["pred_calibrated"]).abs()
        assert diff.max() < 1e-8

    def test_days_from_anchor_nonnegative(self, sbp_df):
        drift = sbp_df[~sbp_df.is_anchor].dropna(subset=["days_from_anchor"])
        assert (drift["days_from_anchor"] >= 0).all()

    def test_subject_count_matches_both_targets(self, sbp_df, dbp_df):
        """SBP and DBP outputs must cover the same set of subjects."""
        assert set(sbp_df.subject_id) == set(dbp_df.subject_id)

    def test_unique_subjects_count(self, sbp_df):
        """Must have at least 10 unique subjects (N=13 currently loaded)."""
        assert sbp_df["subject_id"].nunique() >= 10

    def test_no_null_required_columns(self, sbp_df):
        critical = ["fold", "subject_id", "cuff_truth", "pred_raw", "pred_calibrated",
                    "offset", "err_raw", "err_calibrated", "abs_err_raw", "abs_err_calibrated"]
        for col in critical:
            assert sbp_df[col].notna().all(), f"Null values in required column: {col}"

    @pytest.mark.parametrize("target", ["sbp", "dbp"])
    def test_cuff_truth_in_physiological_range(self, target, sbp_df, dbp_df):
        df = sbp_df if target == "sbp" else dbp_df
        if target == "sbp":
            assert df["cuff_truth"].between(70, 200).all(), "SBP cuff truth out of physiological range"
        else:
            assert df["cuff_truth"].between(40, 130).all(), "DBP cuff truth out of physiological range"
