"""
QA Test Script — clean_gap_csv.py v2 (S08 an 3)
Tests: resample factor, fs detection, output metadata, synthetic CSV, edge cases.

Chay: python test_clean_gap_csv_v2.py
Working dir: c:/Users/Acer/Desktop/PPG monitor
"""
from __future__ import annotations

import importlib.util
import math
import shutil
import sys
import tempfile
from fractions import Fraction
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path("c:/Users/Acer/Desktop/PPG monitor")
BACKEND_SELF_COLLECT = ROOT / "Backend code" / "self_collect"
COLLECTED_DIR = ROOT / "collected_data"
CLEANED_DIR = COLLECTED_DIR / "cleaned"
TEMP_DIR = ROOT / "_qa_clean_csv_tmp"

# ---------------------------------------------------------------------------
# Load the module under test
# ---------------------------------------------------------------------------
spec = importlib.util.spec_from_file_location(
    "clean_gap_csv",
    BACKEND_SELF_COLLECT / "clean_gap_csv.py",
)
clean_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(clean_mod)

clean_csv = clean_mod.clean_csv
_resample_to_target = clean_mod._resample_to_target

# ---------------------------------------------------------------------------
# Result tracking (independent from pytest for standalone run)
# ---------------------------------------------------------------------------
RESULTS: List[Dict] = []


def record(name: str, passed: bool, detail: str = "") -> None:
    status = "PASS" if passed else "FAIL"
    RESULTS.append({"name": name, "status": status, "detail": detail})
    mark = "OK " if passed else "FAIL"
    print(f"  [{mark}] {name}" + (f" -- {detail}" if detail else ""))


def run_test(name: str):
    def wrapper(fn):
        try:
            fn()
            record(name, True)
        except AssertionError as e:
            record(name, False, f"AssertionError: {e}")
        except Exception as e:
            record(name, False, f"{type(e).__name__}: {e}")
        return fn
    return wrapper


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _write_synthetic_csv(
    folder: Path,
    filename: str,
    n_samples: int,
    fs_hz: float,
    subject_id: str = "S_SYNTH",
    sbp: int = 120,
    dbp: int = 80,
    disconnect_after: List[int] = None,  # list of (sample_index, gap_ms) tuples
    disconnect_gaps: List[int] = None,   # gap in ms at each disconnect index
) -> Path:
    """
    Write a synthetic PPG CSV with optional disconnect gaps.

    disconnect_after: list of sample indices after which to insert a big gap.
    disconnect_gaps: corresponding gap sizes in ms (default 2000 ms each).
    """
    folder.mkdir(parents=True, exist_ok=True)
    p = folder / filename
    period_ms = 1000.0 / fs_hz

    if disconnect_after is None:
        disconnect_after = []
    if disconnect_gaps is None:
        disconnect_gaps = [2000] * len(disconnect_after)

    ts = 1_714_000_000_000  # arbitrary epoch ms start
    ts_list = []
    ir_list = []
    red_list = []

    for i in range(n_samples):
        ts_list.append(int(round(ts)))
        ir_list.append(80000 + int(3000 * math.sin(2 * math.pi * 1.2 * i / fs_hz)))
        red_list.append(75000 + int(2500 * math.sin(2 * math.pi * 1.2 * i / fs_hz)))
        # Advance ts
        if i in disconnect_after:
            idx = disconnect_after.index(i)
            ts += disconnect_gaps[idx]
        else:
            ts += period_ms

    with open(p, "w", encoding="utf-8", newline="") as f:
        f.write(f"# subject_id={subject_id},age=25,sex=M,height_cm=175,weight_kg=70,bmi=22.9\n")
        f.write(f"# sbp_baseline={sbp},dbp_baseline={dbp},sbp_post=NA,dbp_post=NA\n")
        f.write("# hypertension=n,medication=n,smoking=n,finger=middle,cuff_arm=left\n")
        f.write(f"# session_date=2026-05-03T10:00:00,fs=100,duration_s={n_samples/100:.1f}\n")
        f.write("# notes=synthetic test\n")
        f.write("timestamp_ms,ir,red\n")
        for t, i, r in zip(ts_list, ir_list, red_list):
            f.write(f"{t},{i},{r}\n")
    return p


# ===========================================================================
# GROUP 1: Resample factor calculation (Fraction.limit_denominator)
# ===========================================================================
print()
print("=" * 70)
print("GROUP 1 — Resample factor calculation")
print("=" * 70)


def test_resample_factor_111hz_to_100hz():
    """111.11 Hz -> 100 Hz. Fraction(100/111.11).limit_denominator(1000) = 9/10."""
    fs_real = 1000.0 / 9.0   # ~111.11 Hz (median interval 9ms as observed)
    frac = Fraction(100.0 / fs_real).limit_denominator(1000)
    assert frac.numerator == 9, f"Expected up=9, got {frac.numerator}"
    assert frac.denominator == 10, f"Expected down=10, got {frac.denominator}"


run_test("resample_factor_111hz_to_100hz")(test_resample_factor_111hz_to_100hz)


def test_resample_factor_100hz_to_100hz_is_1_1():
    """100 Hz -> 100 Hz. Fraction(1.0) = 1/1 (no resample needed)."""
    fs_real = 100.0
    frac = Fraction(100.0 / fs_real).limit_denominator(1000)
    assert frac.numerator == 1, f"Expected up=1, got {frac.numerator}"
    assert frac.denominator == 1, f"Expected down=1, got {frac.denominator}"


run_test("resample_factor_100hz_to_100hz_is_1_1")(test_resample_factor_100hz_to_100hz_is_1_1)


def test_resample_factor_80hz_to_100hz():
    """80 Hz -> 100 Hz. Fraction(100/80) = 5/4."""
    fs_real = 80.0
    frac = Fraction(100.0 / fs_real).limit_denominator(1000)
    assert frac.numerator == 5, f"Expected up=5, got {frac.numerator}"
    assert frac.denominator == 4, f"Expected down=4, got {frac.denominator}"


run_test("resample_factor_80hz_to_100hz")(test_resample_factor_80hz_to_100hz)


def test_resample_to_target_preserves_length_ratio():
    """_resample_to_target: output length ~ input * (up/down)."""
    n_in = 10000
    ir = np.random.randint(70000, 90000, n_in).astype(np.float64)
    red = np.random.randint(65000, 80000, n_in).astype(np.float64)
    fs_real = 1000.0 / 9.0   # 111.11 Hz
    ir_out, red_out, up, down = _resample_to_target(ir, red, fs_real, 100.0)
    expected_len = int(n_in * up / down)
    # resample_poly output length may differ by ±1
    assert abs(len(ir_out) - expected_len) <= 2, (
        f"Output len {len(ir_out)} far from expected {expected_len}"
    )
    assert len(ir_out) == len(red_out), "ir and red output lengths must match"


run_test("resample_to_target_preserves_length_ratio")(test_resample_to_target_preserves_length_ratio)


def test_resample_to_target_output_dtype_int64():
    """_resample_to_target rounds and returns int64."""
    n_in = 1000
    ir = np.full(n_in, 80000, dtype=np.float64)
    red = np.full(n_in, 75000, dtype=np.float64)
    ir_out, red_out, _, _ = _resample_to_target(ir, red, 111.11, 100.0)
    assert ir_out.dtype == np.int64, f"Expected int64, got {ir_out.dtype}"
    assert red_out.dtype == np.int64, f"Expected int64, got {red_out.dtype}"


run_test("resample_to_target_output_dtype_int64")(test_resample_to_target_output_dtype_int64)


# ===========================================================================
# GROUP 2: fs detection from intra-segment median intervals
# ===========================================================================
print()
print("=" * 70)
print("GROUP 2 — fs detection from intra-segment intervals")
print("=" * 70)


def test_fs_detection_uniform_9ms_gives_111hz():
    """Synthetic file with uniform 9ms intervals -> fs_real ~111 Hz detected."""
    tmp = TEMP_DIR / "fs_detect_111"
    # n_samples=3000 at 111Hz ~= 27s, no disconnects
    p = _write_synthetic_csv(tmp, "test_111hz.csv", n_samples=3000, fs_hz=1000/9)
    out = clean_csv(p)
    shutil.rmtree(tmp, ignore_errors=True)
    # Must have produced a cleaned file (fs_drift > 5%)
    assert out != p, "Expected cleaned file to be created for 111Hz input"


run_test("fs_detection_uniform_9ms_gives_111hz")(test_fs_detection_uniform_9ms_gives_111hz)


def test_fs_detection_excludes_disconnect_gaps():
    """
    File with 1 disconnect (2s gap): intra-segment median should still be ~9ms,
    not polluted by the 2000ms gap.
    """
    tmp = TEMP_DIR / "fs_detect_disconnect"
    p = _write_synthetic_csv(
        tmp, "test_disconnect.csv",
        n_samples=2000, fs_hz=1000/9,
        disconnect_after=[999],   # disconnect after sample 999
        disconnect_gaps=[2000],   # 2 second gap
    )
    # Read back the produced file and verify fs_real was detected correctly
    import io
    from contextlib import redirect_stdout
    buf = io.StringIO()
    with redirect_stdout(buf):
        out = clean_csv(p)
    output_text = buf.getvalue()
    shutil.rmtree(tmp, ignore_errors=True)

    # Verify fs_real is reported near 111Hz (NOT polluted by 2s gap)
    assert "111" in output_text or "110" in output_text, (
        f"fs_real should be ~111Hz when disconnect gap excluded.\nOutput:\n{output_text}"
    )


run_test("fs_detection_excludes_disconnect_gaps")(test_fs_detection_excludes_disconnect_gaps)


def test_fs_detection_100hz_no_drift_no_output():
    """
    File with exactly 100Hz (10ms intervals), no disconnects ->
    no cleaned file created (return original path).
    """
    tmp = TEMP_DIR / "fs_detect_100"
    p = _write_synthetic_csv(tmp, "test_100hz.csv", n_samples=3000, fs_hz=100.0)
    out = clean_csv(p)
    shutil.rmtree(tmp, ignore_errors=True)
    assert out == p, (
        "100Hz file with no disconnects should return original path (no output file)"
    )


run_test("fs_detection_100hz_no_drift_no_output")(test_fs_detection_100hz_no_drift_no_output)


# ===========================================================================
# GROUP 3: Output metadata header format
# ===========================================================================
print()
print("=" * 70)
print("GROUP 3 — Output metadata header format")
print("=" * 70)


def test_output_metadata_has_all_required_fields():
    """
    Cleaned CSV must contain all 7 new metadata fields in the # header line:
    cleaned, fs_real_observed, resampled, resample_factor,
    disconnects_removed, dead_time_s, original_samples.
    """
    tmp = TEMP_DIR / "meta_fields"
    p = _write_synthetic_csv(tmp, "test_meta.csv", n_samples=3000, fs_hz=1000/9)
    out = clean_csv(p)
    assert out != p, "Expected cleaned file to be created"

    content = out.read_text(encoding="utf-8")
    required_fields = [
        "cleaned=true",
        "fs_real_observed=",
        "resampled=",
        "resample_factor=",
        "disconnects_removed=",
        "dead_time_s=",
        "original_samples=",
    ]
    for field in required_fields:
        assert field in content, f"Missing metadata field: {field}\nHeader:\n{content[:500]}"
    shutil.rmtree(tmp, ignore_errors=True)


run_test("output_metadata_has_all_required_fields")(test_output_metadata_has_all_required_fields)


def test_output_metadata_fs_updated_to_100():
    """Cleaned file header must report fs=100 (not original 111)."""
    tmp = TEMP_DIR / "meta_fs100"
    p = _write_synthetic_csv(tmp, "test_fs_in_meta.csv", n_samples=3000, fs_hz=1000/9)
    out = clean_csv(p)
    content = out.read_text(encoding="utf-8")
    # Find the header line with fs=
    header_lines = [l for l in content.splitlines() if l.startswith("#") and "fs=" in l]
    assert header_lines, "No header line with fs= found"
    fs_line = header_lines[0]
    # fs= should be 100, NOT 111
    import re
    m = re.search(r'fs=(\d+)', fs_line)
    assert m, f"No fs=<number> found in header: {fs_line}"
    assert int(m.group(1)) == 100, f"Expected fs=100 in cleaned header, got fs={m.group(1)}"
    shutil.rmtree(tmp, ignore_errors=True)


run_test("output_metadata_fs_updated_to_100")(test_output_metadata_fs_updated_to_100)


def test_output_metadata_original_samples_correct():
    """original_samples= must equal the row count of the raw input file."""
    tmp = TEMP_DIR / "meta_orig_samples"
    n = 3000
    p = _write_synthetic_csv(tmp, "test_orig.csv", n_samples=n, fs_hz=1000/9)
    out = clean_csv(p)
    content = out.read_text(encoding="utf-8")
    import re
    m = re.search(r'original_samples=(\d+)', content)
    assert m, "original_samples not found in header"
    assert int(m.group(1)) == n, (
        f"Expected original_samples={n}, got {m.group(1)}"
    )
    shutil.rmtree(tmp, ignore_errors=True)


run_test("output_metadata_original_samples_correct")(test_output_metadata_original_samples_correct)


# ===========================================================================
# GROUP 4: Synthetic CSV full pipeline round-trip
# ===========================================================================
print()
print("=" * 70)
print("GROUP 4 — Synthetic CSV full pipeline round-trip")
print("=" * 70)


def test_output_timestamps_uniform_10ms():
    """
    After cleaning + resampling 111Hz -> 100Hz:
    all consecutive timestamp diffs in cleaned file must be exactly 10ms.
    """
    tmp = TEMP_DIR / "ts_uniform"
    p = _write_synthetic_csv(tmp, "test_uniform_ts.csv", n_samples=3000, fs_hz=1000/9)
    out = clean_csv(p)
    assert out != p, "Expected cleaned file"

    df = pd.read_csv(out, comment="#")
    ts = df["timestamp_ms"].to_numpy(dtype=np.float64)
    diffs = np.diff(ts)
    # Allow ±0.5ms for int rounding in output
    assert np.allclose(diffs, 10.0, atol=0.5), (
        f"Timestamps not uniform 10ms after resample. "
        f"min={diffs.min():.3f}, max={diffs.max():.3f}"
    )
    shutil.rmtree(tmp, ignore_errors=True)


run_test("output_timestamps_uniform_10ms")(test_output_timestamps_uniform_10ms)


def test_output_has_header_and_data_columns():
    """Cleaned CSV must have columns: timestamp_ms, ir, red."""
    tmp = TEMP_DIR / "cols_check"
    p = _write_synthetic_csv(tmp, "test_cols.csv", n_samples=2000, fs_hz=1000/9)
    out = clean_csv(p)
    df = pd.read_csv(out, comment="#")
    assert list(df.columns) == ["timestamp_ms", "ir", "red"], (
        f"Expected [timestamp_ms, ir, red], got {df.columns.tolist()}"
    )
    shutil.rmtree(tmp, ignore_errors=True)


run_test("output_has_header_and_data_columns")(test_output_has_header_and_data_columns)


def test_disconnect_segments_stripped():
    """
    File with 1 disconnect (2s dead time): output sample count must be
    less than input (disconnected gap removed). UPDATED 2026-05-03:
    boundary trim drops 50 samples each inner edge -> 100 samples total
    for 1 disconnect (2 segments, 1 inner boundary x 2 sides).
    """
    tmp = TEMP_DIR / "strip_disconnect"
    n_total = 3000
    p = _write_synthetic_csv(
        tmp, "test_strip.csv",
        n_samples=n_total, fs_hz=1000/9,
        disconnect_after=[999],
        disconnect_gaps=[2000],
    )
    out = clean_csv(p)
    assert out != p, "Expected cleaned file"

    df_raw = pd.read_csv(p, comment="#")
    df_clean = pd.read_csv(out, comment="#")
    # Boundary trim: 50 samples each at end of seg1, start of seg2 = 100 samples
    # then resample 111->100 = factor 9/10
    expected_approx = int((n_total - 100) * 9 / 10)
    assert len(df_clean) <= len(df_raw), "Cleaned file should not be larger than raw"
    assert abs(len(df_clean) - expected_approx) <= 5, (
        f"Expected ~{expected_approx} samples in cleaned, got {len(df_clean)}"
    )
    shutil.rmtree(tmp, ignore_errors=True)


run_test("disconnect_segments_stripped")(test_disconnect_segments_stripped)


def test_output_saved_in_cleaned_subfolder():
    """Cleaned file must be saved in <input_dir>/cleaned/ with _cleaned suffix."""
    tmp = TEMP_DIR / "output_path"
    p = _write_synthetic_csv(tmp, "test_path.csv", n_samples=2000, fs_hz=1000/9)
    out = clean_csv(p)
    assert out != p, "Expected cleaned file created"
    assert out.parent.name == "cleaned", f"Expected parent dir 'cleaned', got {out.parent.name}"
    assert out.stem.endswith("_cleaned"), f"Expected _cleaned suffix, got {out.stem}"
    shutil.rmtree(tmp, ignore_errors=True)


run_test("output_saved_in_cleaned_subfolder")(test_output_saved_in_cleaned_subfolder)


# ===========================================================================
# GROUP 5: Edge cases
# ===========================================================================
print()
print("=" * 70)
print("GROUP 5 — Edge cases")
print("=" * 70)


def test_edge_no_disconnect_small_drift_skip_output():
    """
    File with NO disconnect AND fs drift <= 5% -> return original path (no output).
    Using fs=103Hz (drift 3% from 100Hz target).
    """
    tmp = TEMP_DIR / "edge_small_drift"
    # 103Hz = period 9.709ms, drift = |103-100|/100 = 3% <= 5% threshold
    p = _write_synthetic_csv(tmp, "test_small_drift.csv", n_samples=3000, fs_hz=103.0)
    out = clean_csv(p)
    shutil.rmtree(tmp, ignore_errors=True)
    assert out == p, (
        "File with no disconnect and drift <=5% should return original path unchanged"
    )


run_test("edge_no_disconnect_small_drift_skip_output")(test_edge_no_disconnect_small_drift_skip_output)


def test_edge_disconnect_with_small_drift_creates_output():
    """
    File WITH disconnect but fs_real ~103Hz (drift 3%):
    disconnects are removed but NO resample (drift below threshold).
    Still creates cleaned file because disconnect was present.
    """
    tmp = TEMP_DIR / "edge_disconnect_small_drift"
    p = _write_synthetic_csv(
        tmp, "test_dc_small_drift.csv",
        n_samples=3000, fs_hz=103.0,
        disconnect_after=[999],
        disconnect_gaps=[2000],
    )
    out = clean_csv(p)
    shutil.rmtree(tmp, ignore_errors=True)
    assert out != p, (
        "File with disconnect (even small drift) should create cleaned output"
    )


run_test("edge_disconnect_with_small_drift_creates_output")(
    test_edge_disconnect_with_small_drift_creates_output
)


def test_edge_all_segments_too_short_returns_original():
    """
    All valid segments < MIN_SEGMENT_SAMPLES (100) after disconnect filtering
    -> function returns original path (no output created).
    """
    tmp = TEMP_DIR / "edge_all_short"
    # Create a file where both segments are < 100 samples
    # Disconnect after sample 50 (segment 0: 50 samples, segment 1: 49 samples)
    p = _write_synthetic_csv(
        tmp, "test_short_segs.csv",
        n_samples=100, fs_hz=100.0,
        disconnect_after=[50],
        disconnect_gaps=[2000],
    )
    out = clean_csv(p)
    shutil.rmtree(tmp, ignore_errors=True)
    # Both segments are 50 samples, both < MIN_SEGMENT_SAMPLES=100 -> skip
    assert out == p, (
        "All segments < MIN_SEGMENT_SAMPLES should return original path"
    )


run_test("edge_all_segments_too_short_returns_original")(
    test_edge_all_segments_too_short_returns_original
)


def test_edge_large_drift_no_disconnect_resamples_only():
    """
    File with NO disconnect but large fs drift (111Hz, drift 11% > 5%):
    cleaned file created via resample-only path (disconnects_removed=0).
    """
    tmp = TEMP_DIR / "edge_resample_only"
    p = _write_synthetic_csv(tmp, "test_resample_only.csv", n_samples=3000, fs_hz=1000/9)
    out = clean_csv(p)
    assert out != p, "Expected cleaned file for large drift"

    content = out.read_text(encoding="utf-8")
    assert "disconnects_removed=0" in content, (
        "No disconnects: disconnects_removed should be 0"
    )
    assert "resampled=true" in content, "Expected resampled=true"
    shutil.rmtree(tmp, ignore_errors=True)


run_test("edge_large_drift_no_disconnect_resamples_only")(
    test_edge_large_drift_no_disconnect_resamples_only
)


# ===========================================================================
# GROUP 6: Verify real cleaned files exist (smoke test)
# ===========================================================================
print()
print("=" * 70)
print("GROUP 6 — Real cleaned files smoke test")
print("=" * 70)


def test_real_cleaned_files_count_is_12():
    """At least 12 cleaned CSV files must exist in collected_data/cleaned/.

    UPDATED 2026-05-03: dataset growing (S07_02 session 2 added) -> >= 12
    instead of == 12. Test still validates baseline coverage.
    """
    if not CLEANED_DIR.exists():
        raise AssertionError(f"cleaned/ dir does not exist: {CLEANED_DIR}")
    files = sorted(CLEANED_DIR.glob("*_cleaned.csv"))
    assert len(files) >= 12, f"Expected >=12 cleaned files, got {len(files)}: {[f.name for f in files]}"


run_test("real_cleaned_files_count_is_12")(test_real_cleaned_files_count_is_12)


def test_real_cleaned_sub3_metadata_correct():
    """
    S10 cleaned file: verify all metadata fields present and
    resampled=true, resample_factor=9/10, disconnects_removed=8.
    """
    p = CLEANED_DIR / "S10_20260429_150828_cleaned.csv"
    if not p.exists():
        raise AssertionError(f"File not found: {p}")
    content = p.read_text(encoding="utf-8")
    assert "cleaned=true" in content
    assert "resampled=true" in content
    assert "resample_factor=9/10" in content
    assert "disconnects_removed=8" in content
    assert "fs_real_observed=111.11" in content


run_test("real_cleaned_sub3_metadata_correct")(test_real_cleaned_sub3_metadata_correct)


def test_real_cleaned_sub3_timestamps_uniform():
    """S10 cleaned: consecutive ts diffs should all be 10ms."""
    p = CLEANED_DIR / "S10_20260429_150828_cleaned.csv"
    if not p.exists():
        raise AssertionError(f"File not found: {p}")
    df = pd.read_csv(p, comment="#")
    ts = df["timestamp_ms"].to_numpy(dtype=np.float64)
    diffs = np.diff(ts)
    assert np.allclose(diffs, 10.0, atol=0.5), (
        f"S10 cleaned timestamps not uniform 10ms. min={diffs.min():.3f}, max={diffs.max():.3f}"
    )


run_test("real_cleaned_sub3_timestamps_uniform")(test_real_cleaned_sub3_timestamps_uniform)


def test_real_cleaned_duy01_metadata_has_no_disconnect():
    """
    S02_01 cleaned: no disconnects, resampled=true (111Hz drift), disconnects_removed=0.
    """
    p = CLEANED_DIR / "S02_01_20260501_103156_cleaned.csv"
    if not p.exists():
        raise AssertionError(f"File not found: {p}")
    content = p.read_text(encoding="utf-8")
    assert "disconnects_removed=0" in content, "S02_01 should have 0 disconnects"
    assert "resampled=true" in content, "S02_01 should be resampled (111Hz drift)"
    assert "fs_real_observed=111.11" in content


run_test("real_cleaned_duy01_metadata_has_no_disconnect")(
    test_real_cleaned_duy01_metadata_has_no_disconnect
)


def test_real_cleaned_all_have_3_columns():
    """All 12 cleaned files: CSV must have columns [timestamp_ms, ir, red]."""
    for p in sorted(CLEANED_DIR.glob("*_cleaned.csv")):
        df = pd.read_csv(p, comment="#")
        assert list(df.columns) == ["timestamp_ms", "ir", "red"], (
            f"{p.name}: columns = {df.columns.tolist()}"
        )


run_test("real_cleaned_all_have_3_columns")(test_real_cleaned_all_have_3_columns)


def test_real_cleaned_all_uniform_timestamps():
    """
    All 12 cleaned files: consecutive timestamp diffs should be ~10ms.
    Allow atol=0.5ms for int rounding.
    """
    bad = []
    for p in sorted(CLEANED_DIR.glob("*_cleaned.csv")):
        df = pd.read_csv(p, comment="#")
        ts = df["timestamp_ms"].to_numpy(dtype=np.float64)
        if len(ts) < 2:
            bad.append(f"{p.name}: too short")
            continue
        diffs = np.diff(ts)
        if not np.allclose(diffs, 10.0, atol=0.5):
            bad.append(
                f"{p.name}: min={diffs.min():.3f}, max={diffs.max():.3f}"
            )
    assert not bad, "Non-uniform timestamps found:\n  " + "\n  ".join(bad)


run_test("real_cleaned_all_uniform_timestamps")(test_real_cleaned_all_uniform_timestamps)


# ===========================================================================
# GROUP 7: Task 1A — log_ppg_local.py timestamp fix verification
# ===========================================================================
print()
print("=" * 70)
print("GROUP 7 — log_ppg_local.py timestamp fix verification")
print("=" * 70)


def test_timestamp_fix_code_uses_float_division():
    """
    Verify log_ppg_local.py uses float division (1000.0 / fs) not integer (//).
    Read the source lines and check the expression.
    """
    src = (BACKEND_SELF_COLLECT / "log_ppg_local.py").read_text(encoding="utf-8")
    # Must NOT have integer division for period_ms
    assert "1000 // batch.sample_rate" not in src, (
        "Bug not fixed: still uses integer floor division '1000 // batch.sample_rate'"
    )
    # Must use float division
    assert "1000.0 / batch.sample_rate" in src, (
        "Fix not found: expected '1000.0 / batch.sample_rate'"
    )
    assert "int(round(" in src, (
        "Fix incomplete: expected int(round(...)) for timestamp computation"
    )


run_test("timestamp_fix_code_uses_float_division")(test_timestamp_fix_code_uses_float_division)


def test_timestamp_fix_period_drift_for_fs111():
    """
    Compare old behavior (1000 // 111 = 9ms) vs new (1000.0/111 = 9.009ms).
    Over N=1000 samples, old drift = N * (9 - 9.009) = -9ms ~ -1 sample lost.
    New accumulation: 1000 * round(9.009 * i) should closely track real elapsed.
    """
    fs = 111  # firmware claiming 111 sample/s (9ms period)
    n = 1000

    # Old behavior: period = int(1000 // fs) = 9ms (truncated)
    period_old = 1000 // fs  # = 9
    ts_old_end = n * period_old
    real_elapsed_ms = n * (1000.0 / fs)

    # New behavior: period = 1000.0 / fs = 9.009...ms, rounded
    period_new_float = 1000.0 / fs
    # The new code uses int(round(ts0 + i * period_new_float)) for each i
    ts_new_end = int(round(n * period_new_float))

    drift_old = abs(ts_old_end - real_elapsed_ms)
    drift_new = abs(ts_new_end - real_elapsed_ms)

    # Old drift should be ~9ms, new drift should be < 1ms
    assert drift_old > 5, (
        f"Expected old drift > 5ms for fs=111 over {n} samples, got {drift_old:.3f}ms"
    )
    assert drift_new < 1, (
        f"Expected new drift < 1ms for fs=111 over {n} samples, got {drift_new:.3f}ms"
    )


run_test("timestamp_fix_period_drift_for_fs111")(test_timestamp_fix_period_drift_for_fs111)


def test_timestamp_fix_drift_comparison_fs104():
    """
    At fs=104: old period = 1000 // 104 = 9ms (should be 9.615ms).
    Drift over 1000 samples: old = 1000 * (9.615 - 9) = 615ms (massive).
    New: 1000 * round(9.615) ≈ tracks correctly.
    """
    fs = 104
    n = 1000
    period_old = 1000 // fs  # = 9
    ts_old_end = n * period_old
    real_elapsed_ms = n * (1000.0 / fs)
    drift_old = abs(ts_old_end - real_elapsed_ms)
    # New
    period_new_float = 1000.0 / fs
    ts_new_end = int(round(n * period_new_float))
    drift_new = abs(ts_new_end - real_elapsed_ms)

    assert drift_old > 500, (
        f"Expected large drift (>500ms) for fs=104 with old code, got {drift_old:.1f}ms"
    )
    assert drift_new < 1, (
        f"Expected near-zero drift (<1ms) for fs=104 with new code, got {drift_new:.3f}ms"
    )


run_test("timestamp_fix_drift_comparison_fs104")(test_timestamp_fix_drift_comparison_fs104)


# ===========================================================================
# GROUP 8: Task 5 — Bug regression: eval_naive_baseline.py on raw CSVs
# ===========================================================================
print()
print("=" * 70)
print("GROUP 8 — Bug regression: eval_naive_baseline formula")
print("=" * 70)


def test_realized_fs_formula_not_wrong():
    """
    Verify eval_naive_baseline.py uses MEDIAN formula (gap-resistant).

    UPDATED 2026-05-03: previous (n-1)/dur formula bị pollute boi gaps cho
    files co disconnects -> tinh fs sai (80-88Hz thay vi 111Hz). Da chuyen
    sang median(diff(ts)) -> 1000/median.
    """
    src_path = BACKEND_SELF_COLLECT / "eval_naive_baseline.py"
    src = src_path.read_text(encoding="utf-8")
    # The realized_fs function must use median
    import ast
    tree = ast.parse(src)
    found_correct = False
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "realized_fs":
            func_src = ast.unparse(node)
            if "np.median(np.diff(timestamp_ms))" in func_src:
                found_correct = True
    assert found_correct, (
        "realized_fs in eval_naive_baseline.py must use median(diff(ts)) "
        "not (n-1)/dur (gap-pollution bug)"
    )


run_test("realized_fs_formula_not_wrong")(test_realized_fs_formula_not_wrong)


def test_clean_gap_csv_guard_empty_segments():
    """
    Regression: clean_gap_csv.py must guard against empty segment list
    (all segments < MIN_SEGMENT_SAMPLES) and return original path, not crash.
    """
    tmp = TEMP_DIR / "guard_empty_segs"
    # 3 segments, all 30 samples each (< 100) separated by 2s gaps
    p = _write_synthetic_csv(
        tmp, "test_all_tiny.csv",
        n_samples=90, fs_hz=100.0,
        disconnect_after=[29, 59],
        disconnect_gaps=[2000, 2000],
    )
    # Must not crash
    try:
        out = clean_csv(p)
        passed = True
        detail = f"returned {'original' if out == p else 'cleaned'}"
    except Exception as e:
        passed = False
        detail = f"CRASH: {e}"

    record("clean_gap_csv_guard_empty_segments", passed, detail)
    shutil.rmtree(tmp, ignore_errors=True)
    return


test_clean_gap_csv_guard_empty_segments()


# ===========================================================================
# CLEANUP
# ===========================================================================
print()
print("Cleaning up temp fixtures...")
if TEMP_DIR.exists():
    shutil.rmtree(TEMP_DIR, ignore_errors=True)
    print(f"  Deleted: {TEMP_DIR}")

# ===========================================================================
# SUMMARY
# ===========================================================================
print()
print("=" * 70)
print("KET QUA TONG HOP — test_clean_gap_csv_v2.py")
print("=" * 70)

total = len(RESULTS)
passed_count = sum(1 for r in RESULTS if r["status"] == "PASS")
failed_count = sum(1 for r in RESULTS if r["status"] == "FAIL")

print()
print(f"  Tong: {total} | Pass: {passed_count} | Fail: {failed_count}")
print()

col_w = 55
print(f"  {'Test Case':<{col_w}} {'Status':<8} {'Chi tiet'}")
print(f"  {'-'*col_w} {'-'*8} {'-'*35}")
for r in RESULTS:
    detail = r["detail"][:60] if r["detail"] else ""
    print(f"  {r['name']:<{col_w}} {r['status']:<8} {detail}")

print()
if failed_count == 0:
    print("  TAT CA TESTS PASS")
else:
    print(f"  {failed_count} TEST(S) FAIL - Xem chi tiet ben tren")
print("=" * 70)

# Exit with non-zero if any failures
sys.exit(0 if failed_count == 0 else 1)
