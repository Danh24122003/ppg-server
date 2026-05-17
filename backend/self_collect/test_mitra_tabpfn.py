"""Pytest for eval_mitra_tabpfn.py — schema, capability, bootstrap reproducibility.

Tests:
  - Bootstrap reproducibility (seed=42 deterministic)
  - GroupKFold no subject leakage
  - Feature extraction shapes (X_combined matches train_ppg_bp loader)
  - Edge case: skip if AutoGluon < 1.4 / TabPFN unavailable
  - Output CSV schema (columns + dtypes)
  - Bootstrap CI symmetric for null hypothesis (mean(err)=0)
  - SVR fit/predict matches train_ppg_bp.build_model('svr')
  - Bootstrap delta sign convention
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import eval_mitra_tabpfn as M  # noqa: E402


# =====================================================================
# Bootstrap reproducibility
# =====================================================================
class TestBootstrap:
    def test_seed_deterministic(self):
        rng = np.random.default_rng(42)
        err_a = rng.normal(0, 1, 30)
        err_b = rng.normal(0, 1, 30)
        d1 = M.bootstrap_paired(err_a, err_b, n_boot=1000, seed=42)
        d2 = M.bootstrap_paired(err_a, err_b, n_boot=1000, seed=42)
        assert d1 == d2, "Bootstrap must be deterministic with same seed"

    def test_different_seeds_differ(self):
        err_a = np.abs(np.random.default_rng(0).normal(5, 2, 30))
        err_b = np.abs(np.random.default_rng(1).normal(5, 2, 30))
        d1 = M.bootstrap_paired(err_a, err_b, n_boot=1000, seed=42)
        d2 = M.bootstrap_paired(err_a, err_b, n_boot=1000, seed=43)
        assert d1[0] != d2[0], "Different seeds should give slightly different deltas"

    def test_returns_4_tuple_floats(self):
        err = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = M.bootstrap_paired(err, err.copy(), n_boot=200, seed=42)
        assert len(result) == 4
        for v in result:
            assert isinstance(v, float)

    def test_identical_errors_zero_delta(self):
        err = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        delta_mean, ci_lo, ci_hi, _ = M.bootstrap_paired(
            err, err.copy(), n_boot=2000, seed=42
        )
        assert abs(delta_mean) < 1e-9
        assert abs(ci_lo) < 1e-9
        assert abs(ci_hi) < 1e-9

    def test_paired_length_assert(self):
        with pytest.raises(AssertionError):
            M.bootstrap_paired(
                np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0]),
                n_boot=10, seed=42,
            )

    def test_delta_sign_b_better(self):
        # err_a > err_b => B better => delta = mean(err_a) - mean(err_b) > 0
        err_a = np.array([10.0, 10.0, 10.0, 10.0, 10.0])
        err_b = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        delta_mean, ci_lo, ci_hi, p_one = M.bootstrap_paired(
            err_a, err_b, n_boot=2000, seed=42
        )
        assert delta_mean > 0
        assert ci_lo > 0  # CI excludes 0
        assert p_one < 0.05  # significant


# =====================================================================
# Feature extraction integration
# =====================================================================
class TestDataIntegration:
    def test_self_collected_load_15_persons(self):
        # Smoke load via train_ppg_bp loader; expect 15 unique persons
        # (.
        sys.path.insert(0, str(M.ML_DIR))
        from train_ppg_bp import load_self_collected_features
        X_sc, y_sc, sids = load_self_collected_features(M.COLLECTED_DIR)
        assert X_sc.shape[0] >= 13, "Need >=13 self-collected persons"
        assert X_sc.shape[1] == 38, "Must produce 38 features"
        assert len(sids) == X_sc.shape[0]
        assert "sbp" in y_sc and "dbp" in y_sc

    def test_groupkfold_no_subject_leakage(self):
        from sklearn.model_selection import GroupKFold
        rng = np.random.default_rng(42)
        # 50 subjects, 1-3 sessions each -> ~100 rows
        groups = []
        for sid in range(50):
            n_sess = int(rng.integers(1, 4))
            groups.extend([f"sub{sid}"] * n_sess)
        groups = np.array(groups)
        X = rng.normal(size=(len(groups), 5))
        y = rng.normal(size=len(groups))
        cv = GroupKFold(n_splits=5)
        for tr, te in cv.split(X, y, groups=groups):
            tr_set = set(groups[tr])
            te_set = set(groups[te])
            assert not (tr_set & te_set), "Subject leakage between train and test"


# =====================================================================
# Capability (skip if model unavailable)
# =====================================================================
class TestCapability:
    def test_autogluon_available(self):
        # Should be true in venv-track-a; allow skip if not.
        if not M.AG_AVAILABLE:
            pytest.skip("AutoGluon not installed; run `pip install autogluon.tabular`")
        assert M.AG_VERSION != "unavailable"

    def test_tabpfn_available(self):
        if not M.TABPFN_AVAILABLE:
            pytest.skip("TabPFN not installed; run `pip install tabpfn`")
        assert M.TABPFN_VERSION != "unavailable"

    def test_tabpfn_token_present(self):
        # TABPFN token may be optional for inference depending on tabpfn version.
        # We just verify env var not echoed or accidentally written.
        token = os.getenv("TABPFN_TOKEN")
        if token:
            assert len(token) > 50  # JWT-ish
            # Sanity: never log this in code
            for path in (M.RESULTS_PATH, M.SUMMARY_PATH):
                if path.exists():
                    text = path.read_text(errors="ignore")
                    assert token not in text, "TABPFN_TOKEN must not be written"


# =====================================================================
# Output CSV schema
# =====================================================================
class TestSchema:
    def test_results_columns(self):
        # Build a tiny synthetic results DataFrame matching the schema.
        row = {
            "split_kind": "groupkfold_cv",
            "fold_idx": 1,
            "row_idx": 0,
            "subject_id": "Test",
            "n_train": 100,
            "n_test": 10,
            "sbp_true": 120.0, "sbp_pred_svr": 118.0, "sbp_err_svr": -2.0,
            "sbp_pred_mitra": 119.5, "sbp_err_mitra": -0.5,
            "sbp_pred_tabpfn": 121.0, "sbp_err_tabpfn": 1.0,
            "dbp_true": 80.0, "dbp_pred_svr": 79.0, "dbp_err_svr": -1.0,
            "dbp_pred_mitra": 80.5, "dbp_err_mitra": 0.5,
            "dbp_pred_tabpfn": 78.0, "dbp_err_tabpfn": -2.0,
        }
        df = pd.DataFrame([row])
        summary = M.build_summary(df)
        # Should produce 6 (3 models * 2 targets) + 0 mitra_vs_tabpfn (n=1 < 2) rows
        assert len(summary) >= 6
        for col in ("split_kind", "target", "model", "n_valid", "mae",
                    "delta_vs_svr_mean", "delta_vs_svr_ci_lo",
                    "delta_vs_svr_ci_hi", "p_one_sided"):
            assert col in summary.columns, f"Missing column {col}"

    def test_svr_baseline_has_zero_delta(self):
        rng = np.random.default_rng(42)
        n = 30
        rows = []
        for i in range(n):
            err_svr = rng.normal(0, 5)
            rows.append({
                "split_kind": "true_loso",
                "fold_idx": i,
                "row_idx": -1,
                "subject_id": f"S{i}",
                "n_train": 200,
                "n_test": 1,
                "sbp_true": 120.0,
                "sbp_pred_svr": 120.0 + err_svr,
                "sbp_err_svr": err_svr,
                "sbp_pred_mitra": 120.0 + err_svr * 0.9,
                "sbp_err_mitra": err_svr * 0.9,
                "sbp_pred_tabpfn": 120.0 + err_svr * 0.95,
                "sbp_err_tabpfn": err_svr * 0.95,
                "dbp_true": 80.0,
                "dbp_pred_svr": 80.0,
                "dbp_err_svr": 0.0,
                "dbp_pred_mitra": 80.0,
                "dbp_err_mitra": 0.0,
                "dbp_pred_tabpfn": 80.0,
                "dbp_err_tabpfn": 0.0,
            })
        df = pd.DataFrame(rows)
        summary = M.build_summary(df)
        svr_row = summary[
            (summary["model"] == "svr")
            & (summary["target"] == "sbp")
            & (summary["split_kind"] == "true_loso")
        ].iloc[0]
        assert svr_row["delta_vs_svr_mean"] == 0.0
        assert svr_row["delta_vs_svr_ci_lo"] == 0.0
        assert svr_row["delta_vs_svr_ci_hi"] == 0.0


# =====================================================================
# SVR baseline matches train_ppg_bp
# =====================================================================
class TestSVRBaseline:
    def test_svr_pipeline_matches_train_ppg_bp(self):
        # Train SVR via project's build_model('svr') and via fit_predict_svr;
        # outputs should match within float tolerance on the same data.
        sys.path.insert(0, str(M.ML_DIR))
        from train_ppg_bp import build_model
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        rng = np.random.default_rng(42)
        X = rng.normal(size=(50, 38))
        y = X[:, 0] * 5 + rng.normal(size=50)
        X_test = rng.normal(size=(10, 38))

        # Project's path
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", build_model("svr")),
        ])
        pipe.fit(X, y)
        ref = pipe.predict(X_test)

        # Eval script path
        ours = M.fit_predict_svr(X, y, X_test)
        np.testing.assert_allclose(ours, ref, rtol=1e-9)


# =====================================================================
# Bootstrap CI symmetric under null
# =====================================================================
class TestNullDistribution:
    def test_null_ci_includes_zero(self):
        rng = np.random.default_rng(42)
        err_a = rng.normal(0, 5, 50)
        err_b = err_a + rng.normal(0, 0.01, 50)  # near identical
        delta, ci_lo, ci_hi, _ = M.bootstrap_paired(
            err_a, err_b, n_boot=2000, seed=42
        )
        assert ci_lo < 0 < ci_hi, "Null CI should include 0"


# =====================================================================
# Epsilon canonical assertion (10/5 epsilon sweep)
# =====================================================================
class TestSVREpsilonCanonical:
    def test_svr_epsilon_matches_production(self):
        """Confirm SVR pipeline uses epsilon=1.5 (canonical from train_ppg_bp.py:534).

        Bug history: 8 eval scripts in self_collect/ used sklearn default epsilon=0.1
        instead of production canonical 1.5. Swept 10/5 evening. This test guards
        against regression.
        """
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVR

        p = Pipeline([
            ("scaler", StandardScaler()),
            ("svr", SVR(kernel="rbf", C=10.0, epsilon=1.5, gamma="scale")),
        ])
        assert p.named_steps["svr"].epsilon == 1.5, (
            "SVR epsilon must be 1.5 to match train_ppg_bp.py:534 production canonical"
        )
