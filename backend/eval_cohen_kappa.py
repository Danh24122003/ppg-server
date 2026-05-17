"""
Compute Cohen's kappa between pseudo_label vs manual_label after the user
finishes labeling spot-check plots via label_spot_check_cli.py.

Outputs:
  - 3x3 confusion matrix (rows = pseudo, cols = manual)
  - Cohen's kappa (target >= 0.6 = substantial agreement)
  - Per-class precision / recall / F1 (treats manual_label as ground truth)
  - Overall accuracy

Usage:
  python "backend/eval_cohen_kappa.py"
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "spot_check" / "manifest.csv"
LABELS = ["excellent", "acceptable", "unfit"]


def _interpret_kappa(k: float) -> str:
    if k >= 0.81:
        return "almost perfect (>=0.81)"
    if k >= 0.61:
        return "substantial (0.61-0.80)  <- target for SQA"
    if k >= 0.41:
        return "moderate (0.41-0.60)"
    if k >= 0.21:
        return "fair (0.21-0.40)"
    if k >= 0.0:
        return "slight (0.00-0.20)"
    return "worse than chance (<0)"


def cohen_kappa(y1: list[str], y2: list[str], labels: list[str],
                weights: str | None = None) -> float:
    """Cohen's kappa with explicit label set + optional weighted variant.

    For 3-class ORDINAL labels (excellent/acceptable/unfit), linear-weighted
    kappa is methodologically standard (Fleiss & Cohen 1973, Landis & Koch
    1977) — penalizes "excellent vs unfit" (sai 2 bậc) MORE than "excellent
    vs acceptable" (sai 1 bậc).

    Args:
        weights: None (unweighted), "linear" (recommended for ordinal),
                 "quadratic" (more aggressive penalty for distant errors).

    Returns nan for degenerate cases (empty input, single-class marginals).
    """
    if len(y1) != len(y2) or len(y1) == 0:
        return float("nan")
    cm = np.zeros((len(labels), len(labels)), dtype=float)
    idx = {lab: i for i, lab in enumerate(labels)}
    for a, b in zip(y1, y2):
        if a not in idx or b not in idx:
            continue
        cm[idx[a], idx[b]] += 1
    total = cm.sum()
    if total == 0:
        return float("nan")

    n = len(labels)
    if weights is None:
        # Unweighted: standard Cohen's κ
        po = np.trace(cm) / total
        row_marg = cm.sum(axis=1) / total
        col_marg = cm.sum(axis=0) / total
        pe = float(np.sum(row_marg * col_marg))
    else:
        # Weighted: κ_w = 1 - Σ(w_ij × o_ij) / Σ(w_ij × e_ij)
        w = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(n):
                if weights == "linear":
                    w[i, j] = abs(i - j) / (n - 1)
                elif weights == "quadratic":
                    w[i, j] = ((i - j) ** 2) / ((n - 1) ** 2)
                else:
                    raise ValueError(f"Unknown weights={weights}")
        observed = cm / total
        row_marg = cm.sum(axis=1) / total
        col_marg = cm.sum(axis=0) / total
        expected = np.outer(row_marg, col_marg)
        num = float(np.sum(w * observed))
        den = float(np.sum(w * expected))
        if den < 1e-12:
            return float("nan")
        return float(1.0 - num / den)

    # Degenerate guard for unweighted: pe == 1 OR (pe == 0 AND po == 0)
    if abs(1.0 - pe) < 1e-12:
        return float("nan")
    if pe < 1e-12 and po < 1e-12:
        # Single-class marginals on both sides — Cohen κ undefined
        return float("nan")
    return float((po - pe) / (1.0 - pe))


def confusion_matrix(y1: list[str], y2: list[str], labels: list[str]) -> np.ndarray:
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    idx = {lab: i for i, lab in enumerate(labels)}
    for a, b in zip(y1, y2):
        if a in idx and b in idx:
            cm[idx[a], idx[b]] += 1
    return cm


def per_class_metrics(cm: np.ndarray, labels: list[str]) -> pd.DataFrame:
    """Treats columns as ground truth (manual), rows as prediction (pseudo)."""
    rows = []
    n_total = cm.sum()
    for i, lab in enumerate(labels):
        tp = cm[i, i]
        fp = cm[i, :].sum() - tp  # pseudo says i but manual says other
        fn = cm[:, i].sum() - tp  # manual says i but pseudo says other
        precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        if not np.isnan(precision) and not np.isnan(recall) and (precision + recall) > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = float("nan")
        support = int(cm[:, i].sum())
        rows.append({
            "label": lab,
            "support_manual": support,
            "precision": round(precision, 3) if not np.isnan(precision) else "n/a",
            "recall": round(recall, 3) if not np.isnan(recall) else "n/a",
            "f1": round(f1, 3) if not np.isnan(f1) else "n/a",
        })
    rows.append({
        "label": "TOTAL",
        "support_manual": int(n_total),
        "precision": "",
        "recall": "",
        "f1": "",
    })
    return pd.DataFrame(rows)


def main() -> int:
    if not MANIFEST.exists():
        print(f"[error] {MANIFEST} not found. Run sample_pseudo_label_batches.py first.")
        return 1

    df = pd.read_csv(MANIFEST)
    df["manual_label"] = df["manual_label"].fillna("").astype(str)
    df["pseudo_label"] = df["pseudo_label"].astype(str)

    labeled = df[df["manual_label"].str.len() > 0].copy()
    n_labeled = len(labeled)
    n_total = len(df)
    print(f"Manifest rows: {n_total}")
    print(f"Manually labeled: {n_labeled}")
    if n_labeled == 0:
        print("[error] no manual_label entries found. Run label_spot_check_cli.py first.")
        return 1

    # Filter unknown labels (defensive)
    valid_mask = labeled["manual_label"].isin(LABELS) & labeled["pseudo_label"].isin(LABELS)
    n_dropped = int((~valid_mask).sum())
    if n_dropped > 0:
        print(f"[warn] dropping {n_dropped} rows with non-standard labels")
    labeled = labeled[valid_mask]
    if labeled.empty:
        print("[error] no rows with valid labels after filtering")
        return 1

    y_pseudo = labeled["pseudo_label"].tolist()
    y_manual = labeled["manual_label"].tolist()

    kappa_uw = cohen_kappa(y_pseudo, y_manual, LABELS)
    kappa_lw = cohen_kappa(y_pseudo, y_manual, LABELS, weights="linear")
    kappa_qw = cohen_kappa(y_pseudo, y_manual, LABELS, weights="quadratic")
    cm = confusion_matrix(y_pseudo, y_manual, LABELS)
    accuracy = float(np.trace(cm)) / float(cm.sum()) if cm.sum() > 0 else float("nan")

    print()
    print("=" * 70)
    print("COHEN'S KAPPA — pseudo_label (Orphanidou rules) vs manual_label")
    print("=" * 70)
    print(f"N pairs              : {len(y_pseudo)}")
    print(f"Overall accuracy     : {accuracy:.3f}")
    print(f"Cohen's kappa (unwt) : {kappa_uw:.3f}   [{_interpret_kappa(kappa_uw)}]")
    print(f"Cohen's kappa (linw) : {kappa_lw:.3f}   <- cite this for ordinal labels")
    print(f"Cohen's kappa (quadw): {kappa_qw:.3f}")
    print()
    print("Confusion matrix (rows=pseudo, cols=manual):")
    header = "                " + "".join(f"{lab:>12s}" for lab in LABELS)
    print(header)
    for i, lab in enumerate(LABELS):
        row = f"  pseudo={lab:<10s}" + "".join(f"{cm[i, j]:>12d}" for j in range(len(LABELS)))
        print(row)

    print()
    print("Per-class metrics (manual = ground truth):")
    metrics = per_class_metrics(cm, LABELS)
    print(metrics.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
