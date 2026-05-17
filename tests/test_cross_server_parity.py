"""Assert byte-identical signal-quality-assessment functions between the
Backend lite service and the ML full service.

The two FastAPI services share a substantial common subset (the SQA
pipeline, the HR / HRV / SpO2 estimators, and the supporting peak-detection
and RR-rejection utilities). The cross-server parity invariant is:

    For every function name that appears in BOTH `backend/main.py` and
    `ml/main.py`, the function bodies are byte-identical, except
    for the four endpoints listed in EXPECTED_DIVERGED.

Any drift between the two implementations would be a silent correctness
hazard (one service reports HR computed under one convention while the
other reports it under a different convention); this test fails fast on
any such drift.

Run from the repository root:

    pytest tests/test_cross_server_parity.py -v
"""
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND_MAIN = REPO_ROOT / "backend" / "main.py"
ML_MAIN = REPO_ROOT / "ml" / "main.py"

# Endpoint-level functions that deliberately diverge because the ML
# service adds BP prediction and the four calibration endpoints. The
# remaining common functions must be byte-identical.
EXPECTED_DIVERGED = {
    "_check_device_id",
    "_require_token",
    "root",
    "upload_ppg_data",
}


def _extract_functions(path: Path) -> dict[str, str]:
    """Return {function_name: function_body} for every top-level def."""
    text = path.read_text(encoding="utf-8")
    pattern = re.compile(r"^(?:async\s+)?def\s+(\w+)\s*\(", re.MULTILINE)
    matches = list(pattern.finditer(text))
    funcs: dict[str, str] = {}
    for i, m in enumerate(matches):
        name = m.group(1)
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        funcs[name] = text[start:end].rstrip()
    return funcs


def test_backend_main_exists():
    assert BACKEND_MAIN.exists(), f"missing {BACKEND_MAIN}"


def test_ml_main_exists():
    assert ML_MAIN.exists(), f"missing {ML_MAIN}"


def test_common_sqa_functions_byte_identical():
    """Every function present in both main.py files must have the same body,
    unless it is in EXPECTED_DIVERGED."""
    backend = _extract_functions(BACKEND_MAIN)
    ml = _extract_functions(ML_MAIN)
    common = set(backend) & set(ml)

    assert common, "no common functions found — file parse likely broken"

    drift = []
    for name in sorted(common):
        if name in EXPECTED_DIVERGED:
            continue
        if backend[name] != ml[name]:
            drift.append(name)

    assert not drift, (
        f"cross-server SQA function drift detected — these functions diverge "
        f"between Backend and ML services but are not in EXPECTED_DIVERGED: "
        f"{drift}. Either reconcile the two implementations or add the name to "
        f"EXPECTED_DIVERGED with a comment explaining why divergence is "
        f"intentional."
    )


def test_expected_diverged_actually_diverge():
    """Sanity check: every name in EXPECTED_DIVERGED really differs between
    the two files. If they happen to match, the entry is stale and should be
    removed."""
    backend = _extract_functions(BACKEND_MAIN)
    ml = _extract_functions(ML_MAIN)
    stale = []
    for name in EXPECTED_DIVERGED:
        if name in backend and name in ml and backend[name] == ml[name]:
            stale.append(name)
    assert not stale, (
        f"these EXPECTED_DIVERGED entries are now byte-identical and can be "
        f"removed from the allow-list: {stale}"
    )


def test_minimum_function_coverage():
    """Guard against a parse regression that drops the function count to
    near-zero (e.g. someone refactors to a different syntax)."""
    backend = _extract_functions(BACKEND_MAIN)
    ml = _extract_functions(ML_MAIN)
    assert len(backend) >= 20, f"Backend function count suspiciously low: {len(backend)}"
    assert len(ml) >= 25, f"ML function count suspiciously low: {len(ml)}"
