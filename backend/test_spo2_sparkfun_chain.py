"""
Tests for eval_spo2_sparkfun_chain.py (this work diagnostic, eval-only).

Groups:
  1. compute_spo2_sparkfun() correctness (P0)
  2. Verbatim USPO2_TABLE accuracy (P0) — flags 176 vs 184 entries issue
  3. Production parity — main.py untouched (P0)
  4. End-to-end pipeline via analyze_file() (P1)
  5. Honest framing — docstring caveats (P2)
  6. S10_02 path correctness (P1, quality tracker flag)
"""
from __future__ import annotations

import hashlib
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
BACKEND_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND_DIR))

from eval_spo2_sparkfun_chain import (  # noqa: E402
    USPO2_TABLE,
    analyze_file,
    compute_spo2_sparkfun,
    make_summary,
)
from main import calculate_spo2_v23  # noqa: E402

RESULTS_CSV = BACKEND_DIR / "eval_spo2_sparkfun_chain_results.csv"
SUMMARY_CSV = BACKEND_DIR / "eval_spo2_sparkfun_chain_summary.csv"

# Capture main.py SHA256 once at import time so tests can verify it stays
# unchanged across the test run.
_MAIN_PY_PATH = BACKEND_DIR / "main.py"
_ML_MAIN_PY_PATH = BACKEND_DIR.parent / "ml" / "main.py"
_MAIN_SHA256_AT_IMPORT = hashlib.sha256(_MAIN_PY_PATH.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean_sine(
    fs: int = 100,
    n: int = 500,
    freq: float = 1.2,
    ir_baseline: float = 80_000,
    ir_amp: float = 2_000,
    red_baseline: float = 80_000,
    red_amp: float = 2_000,
) -> tuple[np.ndarray, np.ndarray]:
    t = np.arange(n) / fs
    ir = ir_baseline + ir_amp * np.sin(2 * np.pi * freq * t)
    red = red_baseline + red_amp * np.sin(2 * np.pi * freq * t)
    return ir, red


def _write_temp_csv(ir: np.ndarray, red: np.ndarray, fs: int = 100) -> Path:
    """Write a minimal valid CSV to a temp file and return its Path."""
    t_ms = (np.arange(len(ir)) * 1000.0 / fs).astype(int)
    df = pd.DataFrame({"timestamp_ms": t_ms, "ir": ir.astype(int), "red": red.astype(int)})
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, dir=str(BACKEND_DIR)
    )
    tmp.write("# synthetic test header\n")
    df.to_csv(tmp, index=False)
    tmp.close()
    return Path(tmp.name)


# ===========================================================================
# Group 1 — compute_spo2_sparkfun() correctness (P0)
# ===========================================================================

class TestComputeSpo2SparkfunCorrectness:
    """Verify each step of compute_spo2_sparkfun using synthetic signals."""

    def test_identical_sine_r_equals_one_spo2_near_80(self):
        """IR == Red sine -> R = 1.0 -> lookup index 100 -> SpO2 == 81 (table[100])."""
        ir, red = _clean_sine(ir_amp=2_000, red_amp=2_000)
        spo2, R, ac_ir, ac_red, dc_ir, dc_red, n_pulses, status = compute_spo2_sparkfun(
            ir, red, fs=100
        )
        assert status == "ok", f"Expected 'ok', got '{status}'"
        assert abs(R - 1.0) < 0.01, f"Expected R ~1.0, got {R:.4f}"
        assert spo2 is not None
        # At R=1.0, table index = round(1.0*100) = 100 -> USPO2_TABLE[100]
        assert spo2 == float(USPO2_TABLE[100]), (
            f"SpO2={spo2} != USPO2_TABLE[100]={USPO2_TABLE[100]}"
        )

    def test_identical_sine_peak_detection_finds_enough_peaks(self):
        """5-second 1.2 Hz sine should yield >= 5 pulses detected."""
        ir, red = _clean_sine(n=500, freq=1.2)
        spo2, R, ac_ir, ac_red, dc_ir, dc_red, n_pulses, status = compute_spo2_sparkfun(
            ir, red, fs=100
        )
        assert n_pulses >= 5, f"Expected >= 5 pulses for 1.2 Hz × 5s, got {n_pulses}"
        assert status == "ok"

    def test_identical_sine_ac_approximately_twice_amplitude(self):
        """peak-valley AC per pulse should be approximately 2 * amplitude = 4000."""
        ir, red = _clean_sine(ir_amp=2_000, red_amp=2_000)
        spo2, R, ac_ir, ac_red, dc_ir, dc_red, n_pulses, status = compute_spo2_sparkfun(
            ir, red, fs=100
        )
        assert status == "ok"
        # Mean per-pulse AC should be close to 2*amp but peak-valley of a
        # segment spanning multiple samples approximates 2*amplitude.
        assert 3_000 < ac_ir < 5_000, f"ac_ir={ac_ir:.1f} not in (3000, 5000)"
        assert 3_000 < ac_red < 5_000, f"ac_red={ac_red:.1f} not in (3000, 5000)"

    def test_identical_sine_dc_approximately_baseline(self):
        """DC = (max+min)/2 per pulse should be ~baseline (80000)."""
        ir, red = _clean_sine(ir_baseline=80_000, red_baseline=80_000)
        spo2, R, ac_ir, ac_red, dc_ir, dc_red, n_pulses, status = compute_spo2_sparkfun(
            ir, red, fs=100
        )
        assert status == "ok"
        assert abs(dc_ir - 80_000) < 1_000, f"dc_ir={dc_ir:.0f} too far from 80000"
        assert abs(dc_red - 80_000) < 1_000, f"dc_red={dc_red:.0f} too far from 80000"

    def test_different_ratio_r_approximately_0_45(self):
        """
        IR amp=2000, Red amp=900, same DC=80000:
        R = (900/80000) / (2000/80000) = 0.45.
        Lookup index = round(0.45*100) = 45 -> SpO2 = USPO2_TABLE[45].
        """
        ir, red = _clean_sine(ir_amp=2_000, red_amp=900)
        spo2, R, ac_ir, ac_red, dc_ir, dc_red, n_pulses, status = compute_spo2_sparkfun(
            ir, red, fs=100
        )
        assert status == "ok", f"Expected 'ok', got '{status}'"
        assert abs(R - 0.45) < 0.02, f"Expected R ~0.45, got {R:.4f}"
        expected_idx = int(round(R * 100))
        expected_spo2 = float(USPO2_TABLE[expected_idx])
        assert spo2 == pytest.approx(expected_spo2, abs=0.5), (
            f"SpO2={spo2} != USPO2_TABLE[{expected_idx}]={expected_spo2}"
        )

    def test_too_few_peaks_returns_too_few_pulses_status(self):
        """Signal with fewer than 3 detectable peaks -> status='too_few_pulses', spo2=None."""
        # 100 samples at 100 Hz = 1 second; 1.2 Hz sine -> ~1 peak, too few
        ir, red = _clean_sine(n=100, freq=1.2)
        spo2, R, ac_ir, ac_red, dc_ir, dc_red, n_pulses, status = compute_spo2_sparkfun(
            ir, red, fs=100
        )
        assert status == "too_few_pulses", (
            f"Expected 'too_few_pulses' for short signal, got '{status}'"
        )
        assert spo2 is None, f"Expected None SpO2, got {spo2}"

    def test_zero_dc_signal_returns_too_few_pulses_or_invalid(self):
        """All-zero signal has std=0 on centered signal -> guard fires -> status != 'ok'."""
        ir = np.zeros(500)
        red = np.zeros(500)
        spo2, R, ac_ir, ac_red, dc_ir, dc_red, n_pulses, status = compute_spo2_sparkfun(
            ir, red, fs=100
        )
        assert status in ("too_few_pulses", "invalid_R"), (
            f"Expected rejection for zero signal, got '{status}'"
        )
        assert spo2 is None

    def test_high_r_triggers_lookup_oor(self):
        """
        Very large Red amplitude relative to IR -> R >> 1.75 ->
        table_idx >= len(USPO2_TABLE) -> status='lookup_oor'.
        """
        fs = 100
        t = np.arange(500) / fs
        # IR tiny AC, Red huge AC -> R >> 1
        ir = 80_000 + 100 * np.sin(2 * np.pi * 1.2 * t)
        red = 80_000 + 5_000 * np.sin(2 * np.pi * 1.2 * t)
        spo2, R, ac_ir, ac_red, dc_ir, dc_red, n_pulses, status = compute_spo2_sparkfun(
            ir, red, fs=100
        )
        assert status == "lookup_oor", (
            f"Expected 'lookup_oor' for R={R:.2f}, got '{status}'"
        )
        assert spo2 is None

    def test_determinism_same_input_same_output(self):
        """Calling compute_spo2_sparkfun twice with same input yields identical output."""
        ir, red = _clean_sine(ir_amp=2_000, red_amp=900)
        r1 = compute_spo2_sparkfun(ir, red, fs=100)
        r2 = compute_spo2_sparkfun(ir, red, fs=100)
        assert r1 == r2, f"Non-deterministic output:\n  r1={r1}\n  r2={r2}"


# ===========================================================================
# Group 2 — Verbatim USPO2_TABLE accuracy (P0)
# ===========================================================================

class TestUSPO2TableAccuracy:
    """
    Verify USPO2_TABLE contents against SparkFun source claim of 184 entries.
    The quality tracker flagged 176 entries vs SparkFun source claim of 184.
    Tests in this group DOCUMENT the actual state and flag the discrepancy.
    """

    def test_table_length_documented_176_with_oor_caveat(self):
        """
        USPO2_TABLE has 176 meaningful entries (last value = 1 at index 175).
        SparkFun C declares array size 184 with 8 trailing zero-padding entries
        (indices 176-183 would map to non-physiological SpO2=0).
        Python implementation explicitly documents this in the source comment
        and uses lookup_oor guard at R >= 1.76 instead of R >= 1.84.
        Boundary differs only in (1.76, 1.83] R range — non-physiological,
        never observed in cohort (R range 0.33-0.94).
        """
        assert len(USPO2_TABLE) == 176, (
            f"USPO2_TABLE expected 176 meaningful entries, got {len(USPO2_TABLE)}. "
            "If changed, update source comment + this test docstring accordingly."
        )

    def test_table_plateau_indices_24_to_42_all_100(self):
        """Indices 24 through 42 (inclusive) should all equal 100."""
        plateau = USPO2_TABLE[24:43]
        assert all(v == 100 for v in plateau), (
            f"Some plateau values != 100: {plateau}"
        )

    def test_table_index_43_equals_99(self):
        """Index 43 marks descent from plateau: must be 99."""
        assert int(USPO2_TABLE[43]) == 99, (
            f"USPO2_TABLE[43]={USPO2_TABLE[43]}, expected 99"
        )

    def test_table_index_100_is_correct(self):
        """
        SparkFun source index 100 = 81 (verified by counting source rows).
        Corresponds to R = 1.00.
        """
        assert int(USPO2_TABLE[100]) == 81, (
            f"USPO2_TABLE[100]={USPO2_TABLE[100]}, expected 81 per SparkFun source"
        )

    def test_table_last_valid_entry(self):
        """
        Last entry of the CURRENT table (index 175) should equal 1.
        If table were full 184, index 183 = 1.
        """
        last_val = int(USPO2_TABLE[-1])
        assert last_val == 1, (
            f"Last table entry = {last_val}, expected 1"
        )

    def test_table_dtype_float64(self):
        """Table must be float64 (ensures downstream arithmetic is correct)."""
        assert USPO2_TABLE.dtype == np.float64, (
            f"USPO2_TABLE.dtype={USPO2_TABLE.dtype}, expected float64"
        )

    def test_table_values_monotonically_decrease_after_peak(self):
        """Values from index 43 onward must be non-increasing (monotone descent)."""
        descent = USPO2_TABLE[43:]
        diffs = np.diff(descent)
        assert np.all(diffs <= 0), (
            f"Non-monotone descent at indices: {np.where(diffs > 0)[0] + 43}"
        )


# ===========================================================================
# Group 3 — Production parity (P0)
# ===========================================================================

class TestProductionParity:
    """
    Verify that eval_spo2_sparkfun_chain.py does NOT modify production files.
    The eval script must be read-only relative to main.py and ml/main.py.
    """

    def test_backend_main_py_sha256_unchanged(self):
        """Backend/main.py SHA256 must match the hash captured at test-import time."""
        current_sha = hashlib.sha256(_MAIN_PY_PATH.read_bytes()).hexdigest()
        assert current_sha == _MAIN_SHA256_AT_IMPORT, (
            f"Backend/main.py has been modified during test run.\n"
            f"  Expected: {_MAIN_SHA256_AT_IMPORT}\n"
            f"  Got:      {current_sha}"
        )

    def test_eval_script_imports_calculate_spo2_v23_from_main(self):
        """
        eval_spo2_sparkfun_chain imports calculate_spo2_v23 directly from main.
        Verify the function object is identical (not a re-implementation).
        """
        import eval_spo2_sparkfun_chain as evl  # noqa: F401
        # The module calls calculate_spo2_v23 within analyze_file.
        # Verify it uses the production object by checking the source import line.
        src = (BACKEND_DIR / "eval_spo2_sparkfun_chain.py").read_text()
        assert "from main import calculate_spo2_v23" in src, (
            "eval_spo2_sparkfun_chain.py does not import calculate_spo2_v23 from main"
        )

    def test_import_does_not_raise(self):
        """Importing eval_spo2_sparkfun_chain must not raise any exception."""
        try:
            import importlib
            import eval_spo2_sparkfun_chain
            importlib.reload(eval_spo2_sparkfun_chain)
        except Exception as exc:
            pytest.fail(f"Import of eval_spo2_sparkfun_chain raised: {exc}")

    def test_ml_main_py_not_imported_or_modified_by_eval_script(self):
        """eval_spo2_sparkfun_chain must not import from ml/main.py (cross-server)."""
        src = (BACKEND_DIR / "eval_spo2_sparkfun_chain.py").read_text()
        assert "ml.main" not in src and "ml/main" not in src, (
            "eval_spo2_sparkfun_chain.py unexpectedly references ML server code"
        )


# ===========================================================================
# Group 4 — End-to-end pipeline (P1)
# ===========================================================================

class TestEndToPipeline:
    """Verify analyze_file() and make_summary() with synthetic CSV."""

    @pytest.fixture
    def synthetic_csv(self):
        """Create a 15-second synthetic CSV, yield path, clean up after."""
        fs = 100
        n = 1_500
        t = np.arange(n) / fs
        ir = (80_000 + 2_000 * np.sin(2 * np.pi * 1.2 * t)).astype(int)
        red = (70_000 + 1_200 * np.sin(2 * np.pi * 1.2 * t)).astype(int)
        path = _write_temp_csv(ir, red, fs=fs)
        yield path
        os.unlink(path)

    def test_analyze_file_returns_3_windows_for_15s_at_100hz(self, synthetic_csv):
        """15s / 5s window = 3 windows expected."""
        result = analyze_file(synthetic_csv, fs=100, win_sec=5)
        assert len(result) == 3, f"Expected 3 windows, got {len(result)}"

    def test_analyze_file_output_schema(self, synthetic_csv):
        """Output DataFrame must have the 9 required columns."""
        required = {
            "file", "window_idx", "project_spo2", "project_R", "project_status",
            "sparkfun_spo2", "sparkfun_R", "sparkfun_status", "sparkfun_n_pulses",
        }
        result = analyze_file(synthetic_csv, fs=100, win_sec=5)
        missing = required - set(result.columns)
        assert not missing, f"Missing columns: {missing}"

    def test_analyze_file_both_chains_produce_valid_spo2(self, synthetic_csv):
        """Both project_spo2 and sparkfun_spo2 must not be all-None on clean sine."""
        result = analyze_file(synthetic_csv, fs=100, win_sec=5)
        assert result["project_spo2"].notna().any(), (
            "project_spo2 is all None on clean synthetic signal"
        )
        assert result["sparkfun_spo2"].notna().any(), (
            "sparkfun_spo2 is all None on clean synthetic signal"
        )

    def test_results_csv_exists_and_has_required_columns(self):
        """eval_spo2_sparkfun_chain_results.csv must exist with 9 required columns."""
        if not RESULTS_CSV.exists():
            pytest.skip(f"Results CSV not generated yet: {RESULTS_CSV}")
        df = pd.read_csv(RESULTS_CSV)
        required = {
            "file", "window_idx", "project_spo2", "project_R", "project_status",
            "sparkfun_spo2", "sparkfun_R", "sparkfun_status", "sparkfun_n_pulses",
        }
        missing = required - set(df.columns)
        assert not missing, f"Results CSV missing columns: {missing}"

    def test_summary_csv_exists_and_has_required_columns(self):
        """eval_spo2_sparkfun_chain_summary.csv must exist with 9 required columns."""
        if not SUMMARY_CSV.exists():
            pytest.skip(f"Summary CSV not generated yet: {SUMMARY_CSV}")
        ds = pd.read_csv(SUMMARY_CSV)
        required = {
            "file", "n_windows", "project_n_valid", "sparkfun_n_valid",
            "project_spo2_median", "sparkfun_spo2_median", "spo2_diff_median",
            "project_R_median", "sparkfun_R_median",
        }
        missing = required - set(ds.columns)
        assert not missing, f"Summary CSV missing columns: {missing}"

    def test_summary_csv_cohort_aggregate_project_median(self):
        """Cohort aggregate project SpO2 median must be near 99.50 (claimed in docstring)."""
        if not RESULTS_CSV.exists():
            pytest.skip(f"Results CSV not generated: {RESULTS_CSV}")
        df = pd.read_csv(RESULTS_CSV)
        cohort_median = float(df["project_spo2"].dropna().median())
        assert abs(cohort_median - 99.50) < 1.0, (
            f"Cohort project SpO2 median={cohort_median:.2f}, expected ~99.50"
        )

    def test_summary_csv_cohort_aggregate_sparkfun_median(self):
        """Cohort aggregate SparkFun SpO2 median must be near 99.00 (claimed in docstring)."""
        if not RESULTS_CSV.exists():
            pytest.skip(f"Results CSV not generated: {RESULTS_CSV}")
        df = pd.read_csv(RESULTS_CSV)
        cohort_median = float(df["sparkfun_spo2"].dropna().median())
        assert abs(cohort_median - 99.00) < 1.0, (
            f"Cohort SparkFun SpO2 median={cohort_median:.2f}, expected ~99.00"
        )

    def test_make_summary_groups_by_file_correctly(self, synthetic_csv):
        """make_summary produces one row per file."""
        result = analyze_file(synthetic_csv, fs=100, win_sec=5)
        summary = make_summary(result)
        assert len(summary) == 1, f"Expected 1 summary row, got {len(summary)}"
        row = summary.iloc[0]
        assert row["n_windows"] == 3
        assert row["project_n_valid"] >= 1
        assert row["sparkfun_n_valid"] >= 1


# ===========================================================================
# Group 5 — Honest framing: docstring caveats (P2)
# ===========================================================================

class TestHonestFramingDocstring:
    """
    Verify the module docstring contains required caveats .
    """

    @pytest.fixture(scope="class")
    def docstring(self):
        src = (BACKEND_DIR / "eval_spo2_sparkfun_chain.py").read_text()
        # Extract the module-level docstring (between first pair of triple-quotes)
        start = src.find('"""')
        end = src.find('"""', start + 3)
        return src[start:end + 3].lower()

    def test_caveat_differs_slightly_present(self, docstring):
        """Docstring must mention implementation may differ from actual ESP32 firmware."""
        assert "differ slightly" in docstring, (
            "Docstring missing caveat 'differ slightly' about C++ implementation differences"
        )

    def test_caveat_extrapolating_present(self, docstring):
        """Docstring must mention both chains are extrapolating in the cohort R range."""
        assert "extrapolat" in docstring, (
            "Docstring missing caveat about extrapolation at low R"
        )

    def test_caveat_no_co_oximetry_gt_present(self, docstring):
        """Docstring must acknowledge absence of co-oximetry ground truth."""
        assert "no co-oximetry gt" in docstring or "co-oximetry" in docstring, (
            "Docstring missing caveat about no co-oximetry ground truth"
        )

    def test_caveat_approximations_not_clinically_validated(self, docstring):
        """Docstring must use Maxim's own disclaimer about approximations."""
        assert "approximation" in docstring, (
            "Docstring missing Maxim disclaimer 'approximations, not clinically validated'"
        )


# ===========================================================================
# Group 6 — S10_02 path correctness (P1, quality tracker flag)
# ===========================================================================

class TestSub3_02PathCorrectness:
    """
    Quality-tracker P1: check whether S10_02 typo path exists vs correct path.
    The concern was a silent skip due to path typo.
    """

    # The quality tracker originally claimed the code had a typo
    # 'S10_02_20260503_135502.csv' (wrong timestamp).
    # Inspect the actual source to determine ground truth.
    TYPO_FILENAME = "S10_02_20260503_135502.csv"
    CORRECT_FILENAME = "S10_02_20260503_155516.csv"

    @pytest.fixture(scope="class")
    def collected_data_dir(self):
        # Navigate from backend/ up one level to project root, then collected_data/
        return BACKEND_DIR.parent / "collected_data"

    def test_correct_path_exists_on_disk(self, collected_data_dir):
        """The correct S10_02 file (155516) must exist on disk."""
        correct = collected_data_dir / self.CORRECT_FILENAME
        assert correct.exists(), (
            f"Correct S10_02 file not found: {correct}. "
            "Data file may be missing from collected_data/."
        )

    def test_typo_path_does_not_exist_on_disk(self, collected_data_dir):
        """The typo path (135502) must NOT exist — confirms it would silently skip."""
        typo = collected_data_dir / self.TYPO_FILENAME
        assert not typo.exists(), (
            f"Typo path unexpectedly exists: {typo}. "
            "If this passes, the typo filename corresponds to a real file."
        )

    def test_source_code_uses_correct_filename(self):
        """
        eval_spo2_sparkfun_chain.py source must reference the correct filename
        (155516), NOT the typo (135502).
        """
        src = (BACKEND_DIR / "eval_spo2_sparkfun_chain.py").read_text()
        assert self.CORRECT_FILENAME in src, (
            f"Source does not contain correct filename '{self.CORRECT_FILENAME}'. "
            "File may be silently skipped."
        )
        # Quality-tracker concern: if typo is also present, flag it
        if self.TYPO_FILENAME in src:
            pytest.fail(
                f"Source contains typo filename '{self.TYPO_FILENAME}'. "
                f"This causes a silent skip — correct to '{self.CORRECT_FILENAME}'."
            )

    def test_results_csv_includes_sub3_02_if_csv_exists(self, collected_data_dir):
        """
        If S10_02 is processed, results CSV must contain a row for it.
        This confirms the file was NOT silently skipped.
        """
        if not RESULTS_CSV.exists():
            pytest.skip("Results CSV not found — run main() first")
        df = pd.read_csv(RESULTS_CSV)
        files_in_results = df["file"].unique()
        sub3_present = any(self.CORRECT_FILENAME in str(f) for f in files_in_results)
        assert sub3_present, (
            f"'{self.CORRECT_FILENAME}' not found in results CSV file column. "
            f"Files present: {list(files_in_results)}"
        )
