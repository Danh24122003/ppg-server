"""Tests for eval_sqa_classifier_spo2_aware.py — SpO2-aware 9-feature LR.

Groups:
  1. New feature correctness (red_ir_correlation, r_ratio_std)
  2. Class count verification (P0: new LR 2-class vs 3-class old LR)
  3. Pseudo-label distribution for BP+S09 ( finding: 0% acceptable)
  4. Headline reproducibility (93.28% unfit pct_in_94_100 root cause)
  5. Feature correlation / redundancy check (r_ratio_std vs red_ir_correlation)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BACKEND_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND_DIR))

from eval_sqa_classifier_spo2_aware import (  # noqa: E402
    compute_spo2_aware_features,
)

MODEL_NEW_PKL = BACKEND_DIR / "sqa_lr_spo2_aware_model.pkl"
MODEL_OLD_PKL = BACKEND_DIR / "sqa_lr_model.pkl"
PSEUDO_LABELS_CSV = BACKEND_DIR / "pseudo_labels.csv"
RESULTS_CSV = BACKEND_DIR / "eval_sqa_spo2_aware_results.csv"
SUMMARY_CSV = BACKEND_DIR / "eval_sqa_spo2_aware_summary.csv"


# ============================================================
# Group 1 — New feature correctness
# ============================================================

class TestRedIrCorrelation:

    def test_clean_in_phase_high_correlation(self):
        """Synthetic clean Red+IR sine in phase -> correlation > 0.95."""
        fs = 100
        t = np.arange(1000) / fs
        ir = 80000 + 2000 * np.sin(2 * np.pi * 1.2 * t)
        red = 70000 + 1500 * np.sin(2 * np.pi * 1.2 * t)
        feats = compute_spo2_aware_features(ir, red, fs)
        assert feats["red_ir_correlation"] > 0.95, (
            f"In-phase signals should correlate >0.95; got {feats['red_ir_correlation']:.4f}"
        )

    def test_anti_phase_negative_correlation(self):
        """Red and IR 180-degree anti-phase -> correlation should be negative."""
        fs = 100
        t = np.arange(1000) / fs
        ir = 80000 + 2000 * np.sin(2 * np.pi * 1.2 * t)
        red = 70000 + 1500 * np.sin(2 * np.pi * 1.2 * t + np.pi)
        feats = compute_spo2_aware_features(ir, red, fs)
        assert feats["red_ir_correlation"] < -0.5, (
            f"Anti-phase signals should give negative corr; got {feats['red_ir_correlation']:.4f}"
        )

    def test_random_uncorrelated_near_zero(self):
        """Random uncorrelated signals -> |correlation| < 0.3."""
        rng = np.random.RandomState(42)
        ir = 80000 + rng.randn(1000) * 1000
        red = 70000 + rng.randn(1000) * 800
        feats = compute_spo2_aware_features(ir, red, fs=100)
        assert abs(feats["red_ir_correlation"]) < 0.3, (
            f"Uncorrelated signals should give |corr| <0.3; got {feats['red_ir_correlation']:.4f}"
        )

    def test_flat_signal_safe_default(self):
        """Bug #2 fix verified: flat signal -> red_ir_correlation = 0.0.

        Previously the guard `np.std(ir_f) < 1e-9` (filtered std) was bypassed
        because filtfilt on a constant array produces ~5e-7 numerical noise.
        The fix tests the RAW signal std (>= 1.0) before filtering, so flat
        inputs are short-circuited to (red_ir_correlation=0.0, r_ratio_std=NaN).
        """
        fs = 100
        ir = np.full(1000, 80000.0)
        red = np.full(1000, 70000.0)
        feats = compute_spo2_aware_features(ir, red, fs)
        actual = feats["red_ir_correlation"]
        assert actual == 0.0, (
            f"Flat signal red_ir_correlation must be 0.0 after Bug #2 fix; got {actual}"
        )

    def test_correlation_range_in_minus1_to_1(self):
        """red_ir_correlation must always lie in [-1, 1]."""
        fs = 100
        t = np.arange(500) / fs
        for seed in range(5):
            rng = np.random.RandomState(seed)
            ir = 80000 + rng.randn(500) * 2000
            red = 70000 + rng.randn(500) * 1500
            feats = compute_spo2_aware_features(ir, red, fs)
            r = feats["red_ir_correlation"]
            assert -1.0 <= r <= 1.0, f"seed={seed}: correlation {r:.4f} out of [-1,1]"


class TestRRatioStd:

    def test_constant_ratio_near_zero_std(self):
        """Beats with consistent AC/DC ratio -> r_ratio_std near 0."""
        fs = 100
        t = np.arange(1000) / fs
        # IR and Red share same phase and same AC/DC ratio across all beats
        ir = 80000 + 2000 * np.sin(2 * np.pi * 1.2 * t)
        red = 70000 + 1750 * np.sin(2 * np.pi * 1.2 * t)
        feats = compute_spo2_aware_features(ir, red, fs)
        assert feats["r_ratio_std"] is not None, "r_ratio_std should not be None"
        if not np.isnan(feats["r_ratio_std"]):
            assert feats["r_ratio_std"] < 0.05, (
                f"Constant-ratio signal should give r_ratio_std near 0; got {feats['r_ratio_std']:.4f}"
            )

    def test_motion_artifact_higher_std(self):
        """Beat-to-beat amplitude variation -> r_ratio_std > 0.05."""
        fs = 100
        t = np.arange(1000) / fs
        rng0 = np.random.RandomState(0)
        rng1 = np.random.RandomState(1)
        amp_ir = rng0.uniform(1500, 3000, len(t))
        amp_red = rng1.uniform(1200, 2400, len(t))
        ir = 80000 + amp_ir * np.abs(np.sin(2 * np.pi * 1.2 * t))
        red = 70000 + amp_red * np.abs(np.sin(2 * np.pi * 1.2 * t))
        feats = compute_spo2_aware_features(ir, red, fs)
        if feats["r_ratio_std"] is not None and not np.isnan(feats["r_ratio_std"]):
            assert feats["r_ratio_std"] > 0.05, (
                f"Motion artifact signal should give r_ratio_std > 0.05; got {feats['r_ratio_std']:.4f}"
            )

    def test_flat_signal_returns_nan(self):
        """Bug #2 fix verified: flat signal -> r_ratio_std = NaN.

        After Bug #2 fix, the raw-std guard (`np.std(ir) < 1.0`) short-circuits
        before filtering / peak detection, so spurious peaks in filtfilt noise
        cannot produce a finite r_ratio_std.
        """
        fs = 100
        ir = np.full(1000, 80000.0)
        red = np.full(1000, 70000.0)
        feats = compute_spo2_aware_features(ir, red, fs)
        actual = feats["r_ratio_std"]
        assert np.isnan(actual), (
            f"Flat signal r_ratio_std must be NaN after Bug #2 fix; got {actual}"
        )

    def test_r_ratio_std_nonnegative(self):
        """r_ratio_std (when finite) must be >= 0 (it is np.std, never negative)."""
        fs = 100
        t = np.arange(1000) / fs
        ir = 80000 + 2000 * np.sin(2 * np.pi * 1.2 * t)
        red = 70000 + 1500 * np.sin(2 * np.pi * 1.2 * t)
        feats = compute_spo2_aware_features(ir, red, fs)
        if not np.isnan(feats["r_ratio_std"]):
            assert feats["r_ratio_std"] >= 0.0


# ============================================================
# Group 2 — Class count verification (P0: 2-class vs 3-class)
# ============================================================

class TestLRClassCount:

    @pytest.mark.skipif(not MODEL_NEW_PKL.exists(), reason="sqa_lr_spo2_aware_model.pkl not found")
    def test_new_lr_is_binary_intentionally(self):
        """Bug #1 fix: LR-9 is INTENTIONALLY binary.

        BP+S09 has 0 'acceptable' windows ( finding). After the apples-to-
        apples refactor, LR-9 is trained as a binary excellent-vs-unfit classifier
        on the same 1033 windows as LR-7. Bundle should declare is_binary=True
        and label_order=['excellent', 'unfit'].
        """
        import joblib
        bundle = joblib.load(MODEL_NEW_PKL)
        assert isinstance(bundle, dict), "bundle must be dict (not bare pipeline)"
        pipe = bundle["pipeline"]
        lr = pipe.named_steps["lr"]
        n_classes = len(lr.classes_)
        print(f"\nLR-9 classes: {lr.classes_}, count={n_classes}")
        assert n_classes == 2, f"LR-9 must be binary; got {n_classes} classes"
        assert set(lr.classes_) == {"excellent", "unfit"}
        assert bundle.get("is_binary") is True
        assert bundle.get("label_order") == ["excellent", "unfit"]

    @pytest.mark.skipif(not MODEL_OLD_PKL.exists(), reason="sqa_lr_model.pkl not found")
    def test_old_lr_class_count_is_three(self):
        """ LR must be 3-class (excellent, acceptable, unfit)."""
        import joblib
        bundle = joblib.load(MODEL_OLD_PKL)
        pipe = bundle["pipeline"] if isinstance(bundle, dict) else bundle
        lr = pipe.named_steps["lr"] if hasattr(pipe, "named_steps") else pipe
        print(f"\nOld LR classes: {lr.classes_}, count={len(lr.classes_)}")
        assert len(lr.classes_) == 3, (
            f"Old LR should be 3-class; got {lr.classes_}"
        )
        assert set(lr.classes_) == {"excellent", "acceptable", "unfit"}


# ============================================================
# Group 3 — Pseudo-label distribution for BP+S09
# ============================================================

class TestPseudoLabelDistribution:

    @pytest.mark.skipif(not PSEUDO_LABELS_CSV.exists(), reason="pseudo_labels.csv not found")
    def test_bp_quoc_have_zero_acceptable_windows(self):
        """ finding: BP-protocol has bimodal distribution (~0% acceptable).
        S09-SQA also ~0% acceptable.
        This is the ROOT CAUSE of the 2-class LR (no acceptable to train on).

        Updated 2026-05-13: post  cohort grew (BP-protocol 937 -> 1440 after
        re-cleaning). Bimodal finding still holds — acceptable count remains <1%
        of the BP-protocol + S09-SQA union. Currently 2/1536 = 0.130%
        (BP-protocol contributes both acceptable windows: 2/1440 = 0.139%;
        S09-SQA contributes 0/96). Companion check on BP-protocol alone lives in
        test_pseudo_labels.py::test_zero_acceptable_is_bimodal_finding.
        """
        df = pd.read_csv(PSEUDO_LABELS_CSV)
        bp_quoc = df[df["dataset"].isin(["BP-protocol", "S09-SQA"])]
        counts = bp_quoc["label"].value_counts()
        print(f"\nBP+S09 label distribution: {counts.to_dict()}")
        acceptable_count = int(counts.get("acceptable", 0))
        n_total = len(bp_quoc)
        acceptable_ratio = acceptable_count / n_total if n_total else 0.0
        assert acceptable_ratio < 0.01, (
            f"Expected <1% 'acceptable' windows in BP+S09; got "
            f"{acceptable_count}/{n_total} ({acceptable_ratio*100:.2f}%). "
            "If significantly higher, the 2-class LR P0 bug would not exist."
        )

    @pytest.mark.skipif(not PSEUDO_LABELS_CSV.exists(), reason="pseudo_labels.csv not found")
    def test_bp_quoc_have_both_excellent_and_unfit(self):
        """BP+S09 must have both excellent and unfit classes for binary LR to train."""
        df = pd.read_csv(PSEUDO_LABELS_CSV)
        bp_quoc = df[df["dataset"].isin(["BP-protocol", "S09-SQA"])]
        counts = bp_quoc["label"].value_counts()
        assert counts.get("excellent", 0) > 0, "No 'excellent' windows in BP+S09"
        assert counts.get("unfit", 0) > 0, "No 'unfit' windows in BP+S09"


# ============================================================
# Group 4 — Headline reproducibility (93.28% unfit pct_in_94_100)
# ============================================================

class TestHeadlineReproducibility:

    @pytest.mark.skipif(not SUMMARY_CSV.exists(), reason="eval_sqa_spo2_aware_summary.csv not found")
    def test_summary_has_three_label_sources(self):
        """Summary CSV must contain rows from LR-7 binary, LR-9 binary, and  LR (ref)."""
        summary = pd.read_csv(SUMMARY_CSV)
        sources = set(summary["label_source"].unique())
        assert {"lr7_binary", "lr9_binary", "day5_lr_3class"}.issubset(sources), (
            f"Summary CSV missing one of the 3 label_sources; got {sources}"
        )

    @pytest.mark.skipif(not SUMMARY_CSV.exists(), reason="eval_sqa_spo2_aware_summary.csv not found")
    def test_lr7_lr9_excellent_high_pct_in_94_100(self):
        """Both LR-7 and LR-9 BP-protocol excellent tier should show high pct_in_94_100
        (clean windows -> healthy SpO2 in [94, 100]).
        """
        summary = pd.read_csv(SUMMARY_CSV)
        for ls in ["lr7_binary", "lr9_binary"]:
            row = summary[
                (summary["label_source"] == ls)
                & (summary["dataset"] == "BP-protocol")
                & (summary["tier"] == "excellent")
            ]
            assert not row.empty, f"Missing {ls}/BP-protocol/excellent row"
            pct = float(row.iloc[0]["pct_in_94_100"])
            print(f"\n{ls} BP-protocol excellent pct_in_94_100: {pct:.2f}%")
            assert pct > 90.0, (
                f"{ls} BP-protocol excellent pct_in_94_100 unexpectedly low: {pct:.2f}%"
            )

    @pytest.mark.skipif(not RESULTS_CSV.exists(), reason="eval_sqa_spo2_aware_results.csv not found")
    def test_binary_lrs_never_predict_acceptable(self):
        """Both LR-7 and LR-9 are binary -> 'acceptable' must never appear in predictions."""
        res = pd.read_csv(RESULTS_CSV)
        for col in ["label_pred_lr7", "label_pred_lr9"]:
            n_acc = int((res[col] == "acceptable").sum())
            print(f"\n{col} 'acceptable' predictions: {n_acc}")
            assert n_acc == 0, (
                f"{col} (binary LR) should never predict 'acceptable'; got {n_acc}"
            )


# ============================================================
# Group 5 — Feature correlation / redundancy check
# ============================================================

class TestFeatureCorrelation:

    @pytest.mark.skipif(not RESULTS_CSV.exists(), reason="eval_sqa_spo2_aware_results.csv not found")
    def test_r_ratio_std_vs_red_ir_correlation_high_anticorrelation(self):
        """DIAGNOSTIC: r_ratio_std and red_ir_correlation show high anti-correlation
        (r ~ -0.85 in the data). This means the two 'new' features are nearly
        redundant with each other. Adding both provides minimal independent information.
        Print the value; assert it is negatively correlated (not independently informative).
        """
        res = pd.read_csv(RESULTS_CSV)
        df_clean = res[["r_ratio_std", "red_ir_correlation"]].dropna()
        corr = float(df_clean["r_ratio_std"].corr(df_clean["red_ir_correlation"]))
        print(f"\nr_ratio_std vs red_ir_correlation: r = {corr:.4f}")
        # These two new features are highly anti-correlated with each other
        assert corr < -0.7, (
            f"Expected high anti-correlation between new features; got r={corr:.4f}. "
            "If not anti-correlated, the feature set may have changed."
        )

    @pytest.mark.skipif(not RESULTS_CSV.exists(), reason="eval_sqa_spo2_aware_results.csv not found")
    def test_existing_features_no_extreme_collinearity_with_new(self):
        """No existing feature should have |correlation| > 0.90 with the new features.
        (r_ratio_std is correlated with entropy_amp_dist at ~-0.40, acceptable.)
        Prints full correlation row for audit.
        """
        res = pd.read_csv(RESULTS_CSV)
        feats = [
            "spectral_purity", "hr_bpm", "amplitude_ratio", "ssqi", "ksqi",
            "entropy_amp_dist", "tsqi_median", "r_ratio_std", "red_ir_correlation",
        ]
        df_clean = res[feats].dropna()
        corr = df_clean.corr()
        print("\nr_ratio_std vs existing 7 features:")
        for f in feats:
            if f != "r_ratio_std":
                print(f"  {f:<25}: {corr.loc['r_ratio_std', f]:.4f}")
        print("\nred_ir_correlation vs existing 7 features:")
        for f in feats:
            if f != "red_ir_correlation":
                print(f"  {f:<25}: {corr.loc['red_ir_correlation', f]:.4f}")

        # No existing-7 feature should be >0.90 correlated with new features
        for new_f in ["r_ratio_std", "red_ir_correlation"]:
            for old_f in ["spectral_purity", "hr_bpm", "amplitude_ratio",
                          "ssqi", "ksqi", "entropy_amp_dist", "tsqi_median"]:
                c = abs(corr.loc[new_f, old_f])
                assert c < 0.90, (
                    f"|corr({new_f}, {old_f})| = {c:.4f} >= 0.90 (near-collinear with existing)"
                )

    @pytest.mark.skipif(not RESULTS_CSV.exists(), reason="eval_sqa_spo2_aware_results.csv not found")
    def test_feature_value_ranges_plausible(self):
        """red_ir_correlation in [-1,1]; r_ratio_std typically [0, 0.3] for clean signals."""
        res = pd.read_csv(RESULTS_CSV)
        r_corr = res["red_ir_correlation"].dropna()
        r_std = res["r_ratio_std"].dropna()

        # red_ir_correlation must be in [-1, 1]
        assert r_corr.min() >= -1.0, f"red_ir_correlation below -1: {r_corr.min():.4f}"
        assert r_corr.max() <= 1.0, f"red_ir_correlation above  1: {r_corr.max():.4f}"

        # r_ratio_std must be nonneg; most BP-protocol (clean) values should be < 0.3
        assert r_std.min() >= 0.0, f"r_ratio_std is negative: {r_std.min():.4f}"
        pct_above_point3 = (r_std > 0.3).mean() * 100
        print(f"\nr_ratio_std: min={r_std.min():.4f}, max={r_std.max():.4f}, "
              f"mean={r_std.mean():.4f}, pct>0.3 = {pct_above_point3:.1f}%")
        # Not a hard assertion on pct_above_point3 — just print for review
