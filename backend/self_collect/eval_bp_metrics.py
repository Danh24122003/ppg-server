"""
Đánh giá BP estimation theo các chuẩn AAMI/BHS/IEEE — sinh ra full report
gồm metrics + Bland-Altman + scatter + Markdown table cho thesis.

INPUT: CSV chứa paired prediction vs ground truth (BP cuff Omron).
       Schema bắt buộc:
         subject_id, sbp_true, sbp_pred, dbp_true, dbp_pred
       Optional columns: session_id, fold (cho LOSO CV)

OUTPUT: folder `report/` (default) chứa:
  - metrics.json           — đầy đủ metrics dạng JSON cho thesis cite
  - metrics_table.md        — Markdown table copy-paste vào thesis
  - bland_altman_sbp.png   — Bland-Altman plot SBP
  - bland_altman_dbp.png   — Bland-Altman plot DBP
  - scatter_sbp.png         — Scatter prediction vs truth SBP
  - scatter_dbp.png         — Scatter prediction vs truth DBP
  - error_histogram.png    — Histogram phân bố errors

USAGE:
  # Sau khi train + predict, tạo CSV results.csv với 4 cột bắt buộc
  python eval_bp_metrics.py --predictions results.csv --output report/

  # Compare 2 model (uncalibrated vs calibrated):
  python eval_bp_metrics.py --predictions raw.csv --output report_raw/
  python eval_bp_metrics.py --predictions calibrated.csv --output report_cal/

REQUIREMENTS: pandas, numpy, scipy, matplotlib (đã có sẵn từ project)

CHUẨN ÁP DỤNG:
  - AAMI/ANSI/ISO 81060-2:2018: ME ≤ 5 mmHg, SD ≤ 8 mmHg
  - BHS Protocol (O'Brien 1993): Grade A/B/C/D theo % errors within 5/10/15 mmHg
  - IEEE 1708-2014: ±5 mmHg cho cuffless BP
  - Bland-Altman: bias + 95% LoA (mean ± 1.96*SD)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

# Force UTF-8 output trên Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import matplotlib

matplotlib.use("Agg")  # non-interactive backend, không hiện figure GUI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


# ============================================================
# Metrics computation
# ============================================================
def compute_basic_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Tính các metric cơ bản: MAE, RMSE, ME, SD, R, R²."""
    errors = y_pred - y_true  # signed (cho Bland-Altman bias)
    abs_errors = np.abs(errors)

    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    me = float(np.mean(errors))  # mean error / bias
    # SD ddof=1 cần N>1; với N=1 SD undefined, default 0.0
    sd = float(np.std(errors, ddof=1)) if len(errors) > 1 else 0.0

    # Pearson r crash nếu N<3 hoặc một trong 2 array là constant (std=0).
    # Với BP estimation hiện tại R² âm, scenario predictions=constant rất có thể.
    if len(y_true) >= 3 and np.std(y_true) > 1e-9 and np.std(y_pred) > 1e-9:
        try:
            pearson_r, p_value = stats.pearsonr(y_true, y_pred)
            if not np.isfinite(pearson_r):
                pearson_r, p_value = 0.0, 1.0
        except (ValueError, FloatingPointError):
            pearson_r, p_value = 0.0, 1.0
    else:
        pearson_r, p_value = 0.0, 1.0

    # R² của linear fit (không phải coefficient of determination cho residual)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {
        "n_samples": int(len(y_true)),
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "me": round(me, 2),
        "sd": round(sd, 2),
        "pearson_r": round(float(pearson_r), 3),
        "pearson_p": round(float(p_value), 4),
        "r2": round(r2, 3),
        "max_error": round(float(np.max(abs_errors)), 2),
        "min_error": round(float(np.min(abs_errors)), 2),
    }


def compute_aami_pass(metrics: Dict[str, float]) -> Dict[str, object]:
    """
    AAMI/ISO 81060-2:2018 criteria: ME ≤ 5 mmHg, SD ≤ 8 mmHg.
    Trả về dict với pass/fail và lý do.
    """
    me_pass = abs(metrics["me"]) <= 5.0
    sd_pass = metrics["sd"] <= 8.0
    overall_pass = me_pass and sd_pass

    reasons = []
    if not me_pass:
        reasons.append(f"|ME|={abs(metrics['me']):.2f} > 5 mmHg")
    if not sd_pass:
        reasons.append(f"SD={metrics['sd']:.2f} > 8 mmHg")

    return {
        "pass": overall_pass,
        "me_pass": me_pass,
        "sd_pass": sd_pass,
        "reasons": reasons if reasons else ["Pass cả ME và SD"],
    }


def compute_bhs_grade(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, object]:
    """
    BHS Protocol grading (O'Brien 1993, validated trong AHA Hypertension):
    Tính % errors within 5/10/15 mmHg, gán Grade A/B/C/D.

    Grade  | ≤5 mmHg | ≤10 mmHg | ≤15 mmHg
    A      | ≥60%    | ≥85%     | ≥95%
    B      | ≥50%    | ≥75%     | ≥90%
    C      | ≥40%    | ≥65%     | ≥85%
    D      | dưới C  |          |
    """
    abs_errors = np.abs(y_pred - y_true)
    n = len(abs_errors)
    pct_within_5 = float((abs_errors <= 5).sum() / n * 100)
    pct_within_10 = float((abs_errors <= 10).sum() / n * 100)
    pct_within_15 = float((abs_errors <= 15).sum() / n * 100)

    # Grade decision: cần đạt CẢ 3 ngưỡng cùng lúc
    if pct_within_5 >= 60 and pct_within_10 >= 85 and pct_within_15 >= 95:
        grade = "A"
    elif pct_within_5 >= 50 and pct_within_10 >= 75 and pct_within_15 >= 90:
        grade = "B"
    elif pct_within_5 >= 40 and pct_within_10 >= 65 and pct_within_15 >= 85:
        grade = "C"
    else:
        grade = "D"

    bhs_pass = grade in ("A", "B")

    return {
        "grade": grade,
        "pass": bhs_pass,
        "pct_within_5": round(pct_within_5, 1),
        "pct_within_10": round(pct_within_10, 1),
        "pct_within_15": round(pct_within_15, 1),
    }


def compute_bland_altman(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Bland-Altman analysis: bias (ME) + Limits of Agreement (LoA).
    LoA = mean_diff ± 1.96 × SD_diff (95% CI cho 1 cá nhân mới).
    """
    diff = y_pred - y_true
    bias = float(np.mean(diff))
    sd_diff = float(np.std(diff, ddof=1)) if len(diff) > 1 else 0.0
    loa_upper = float(bias + 1.96 * sd_diff)
    loa_lower = float(bias - 1.96 * sd_diff)

    # Test bias significantly different from 0 (paired t-test against 0).
    # Crash nếu sd_diff=0 (all errors equal). Với perfect bias (vd model luôn predict
    # +5 mmHg), bias != 0 nhưng sd=0 → systematic bias hoàn toàn.
    if sd_diff > 1e-9 and len(diff) > 1:
        try:
            t_stat, p_value = stats.ttest_1samp(diff, 0)
            if not np.isfinite(p_value):
                t_stat, p_value = 0.0, 1.0
        except (ValueError, FloatingPointError):
            t_stat, p_value = 0.0, 1.0
    else:
        # sd=0: nếu bias != 0 → systematic bias mark significant
        t_stat = float("inf") if abs(bias) > 1e-9 else 0.0
        p_value = 0.0 if abs(bias) > 1e-9 else 1.0

    return {
        "bias": round(bias, 2),
        "sd_diff": round(sd_diff, 2),
        "loa_upper": round(loa_upper, 2),
        "loa_lower": round(loa_lower, 2),
        "loa_width": round(loa_upper - loa_lower, 2),
        "bias_t_stat": round(float(t_stat), 3),
        "bias_p_value": round(float(p_value), 4),
        "bias_significant": bool(p_value < 0.05),
    }


# ============================================================
# Plots
# ============================================================
def plot_bland_altman(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_name: str,
    out_path: Path,
    title_suffix: str = "",
) -> None:
    """Bland-Altman plot với bias + 95% LoA."""
    diff = y_pred - y_true
    mean_bp = (y_pred + y_true) / 2.0
    bias = np.mean(diff)
    sd_diff = np.std(diff, ddof=1)
    loa_upper = bias + 1.96 * sd_diff
    loa_lower = bias - 1.96 * sd_diff

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(mean_bp, diff, alpha=0.5, s=30, color="#2E86AB", edgecolor="white", linewidth=0.5)

    # Bias line
    ax.axhline(bias, color="#06A77D", linestyle="-", linewidth=2,
               label=f"Mean bias: {bias:+.2f} mmHg")
    # LoA lines
    ax.axhline(loa_upper, color="#D62246", linestyle="--", linewidth=1.5,
               label=f"+1.96 SD: {loa_upper:+.2f}")
    ax.axhline(loa_lower, color="#D62246", linestyle="--", linewidth=1.5,
               label=f"-1.96 SD: {loa_lower:+.2f}")
    # Zero line
    ax.axhline(0, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)

    ax.set_xlabel(f"Mean of measured and predicted {target_name} (mmHg)", fontsize=11)
    ax.set_ylabel(f"Difference: Predicted - True {target_name} (mmHg)", fontsize=11)
    title = f"Bland-Altman Plot — {target_name}"
    if title_suffix:
        title += f" ({title_suffix})"
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_name: str,
    out_path: Path,
    title_suffix: str = "",
) -> None:
    """Scatter prediction vs truth với line of identity + linear fit."""
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(y_true, y_pred, alpha=0.5, s=30, color="#2E86AB",
               edgecolor="white", linewidth=0.5, label="Data points")

    # Line of identity (perfect prediction)
    lo = min(y_true.min(), y_pred.min()) - 5
    hi = max(y_true.max(), y_pred.max()) + 5
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.5, alpha=0.7,
            label="Identity (y=x)")

    # Linear regression fit — skip nếu input degenerate (constant) để tránh
    # ValueError "Cannot calculate a linear regression if all x values are identical"
    if len(y_true) >= 3 and np.std(y_true) > 1e-9 and np.std(y_pred) > 1e-9:
        slope, intercept, r_value, _, _ = stats.linregress(y_true, y_pred)
        x_fit = np.array([lo, hi])
        y_fit = slope * x_fit + intercept
        ax.plot(x_fit, y_fit, color="#D62246", linewidth=1.8,
                label=f"Linear fit: y={slope:.2f}x{intercept:+.1f}, r={r_value:.3f}")

    ax.set_xlabel(f"True {target_name} from cuff (mmHg)", fontsize=11)
    ax.set_ylabel(f"Predicted {target_name} (mmHg)", fontsize=11)
    title = f"Prediction vs Ground Truth — {target_name}"
    if title_suffix:
        title += f" ({title_suffix})"
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_error_histogram(
    sbp_errors: np.ndarray, dbp_errors: np.ndarray, out_path: Path
) -> None:
    """Histogram phân bố errors cho cả SBP và DBP với reference lines BHS."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, errors, name, color in [
        (axes[0], sbp_errors, "SBP", "#2E86AB"),
        (axes[1], dbp_errors, "DBP", "#06A77D"),
    ]:
        ax.hist(np.abs(errors), bins=20, color=color, alpha=0.7, edgecolor="white")
        # BHS reference lines
        ax.axvline(5, color="green", linestyle="--", linewidth=1.5, label="5 mmHg (BHS)")
        ax.axvline(10, color="orange", linestyle="--", linewidth=1.5, label="10 mmHg (BHS)")
        ax.axvline(15, color="red", linestyle="--", linewidth=1.5, label="15 mmHg (BHS)")

        ax.set_xlabel(f"|Error| {name} (mmHg)", fontsize=11)
        ax.set_ylabel("Frequency", fontsize=11)
        ax.set_title(f"Error Distribution — {name}", fontsize=12, fontweight="bold")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Markdown table generation
# ============================================================
def generate_markdown_report(
    sbp_metrics: dict, dbp_metrics: dict,
    sbp_aami: dict, dbp_aami: dict,
    sbp_bhs: dict, dbp_bhs: dict,
    sbp_ba: dict, dbp_ba: dict,
    n_subjects: int,
    n_recordings: int,
    note: str = "",
) -> str:
    """Sinh Markdown report đầy đủ — copy paste vào thesis."""

    md = []
    md.append("# BP Estimation Evaluation Report\n")
    if note:
        md.append(f"**Note:** {note}\n")
    md.append(f"- Subjects: **{n_subjects}**")
    md.append(f"- Total recordings: **{n_recordings}**\n")

    # Table 1: Metrics
    md.append("## Bảng 1 — Metrics tổng hợp\n")
    md.append("| Metric | SBP | DBP | Chuẩn |")
    md.append("|---|---|---|---|")
    md.append(f"| **MAE** (mmHg) | {sbp_metrics['mae']} | {dbp_metrics['mae']} | IEEE 1708: ≤5 |")
    md.append(f"| RMSE (mmHg) | {sbp_metrics['rmse']} | {dbp_metrics['rmse']} | — |")
    md.append(f"| ME ± SD (mmHg) | {sbp_metrics['me']} ± {sbp_metrics['sd']} | {dbp_metrics['me']} ± {dbp_metrics['sd']} | AAMI: \\|ME\\|≤5, SD≤8 |")
    md.append(f"| Pearson r | {sbp_metrics['pearson_r']} | {dbp_metrics['pearson_r']} | — |")
    md.append(f"| R² | {sbp_metrics['r2']} | {dbp_metrics['r2']} | — |")
    md.append(f"| Max abs error | {sbp_metrics['max_error']} | {dbp_metrics['max_error']} | — |\n")

    # Table 2: AAMI Pass/Fail
    md.append("## Bảng 2 — AAMI/ISO 81060-2:2018\n")
    md.append("| Tiêu chí | SBP | DBP |")
    md.append("|---|---|---|")
    sbp_me_str = "✅" if sbp_aami["me_pass"] else "❌"
    dbp_me_str = "✅" if dbp_aami["me_pass"] else "❌"
    sbp_sd_str = "✅" if sbp_aami["sd_pass"] else "❌"
    dbp_sd_str = "✅" if dbp_aami["sd_pass"] else "❌"
    sbp_pass_str = "✅ PASS" if sbp_aami["pass"] else "❌ FAIL"
    dbp_pass_str = "✅ PASS" if dbp_aami["pass"] else "❌ FAIL"
    md.append(f"| \\|ME\\| ≤ 5 mmHg | {sbp_me_str} | {dbp_me_str} |")
    md.append(f"| SD ≤ 8 mmHg | {sbp_sd_str} | {dbp_sd_str} |")
    md.append(f"| **Overall** | {sbp_pass_str} | {dbp_pass_str} |\n")

    # Table 3: BHS Grade
    md.append("## Bảng 3 — BHS Protocol Grading\n")
    md.append("| Ngưỡng | SBP % | DBP % | Grade A | Grade B | Grade C |")
    md.append("|---|---|---|---|---|---|")
    md.append(f"| ≤ 5 mmHg | {sbp_bhs['pct_within_5']}% | {dbp_bhs['pct_within_5']}% | ≥60% | ≥50% | ≥40% |")
    md.append(f"| ≤ 10 mmHg | {sbp_bhs['pct_within_10']}% | {dbp_bhs['pct_within_10']}% | ≥85% | ≥75% | ≥65% |")
    md.append(f"| ≤ 15 mmHg | {sbp_bhs['pct_within_15']}% | {dbp_bhs['pct_within_15']}% | ≥95% | ≥90% | ≥85% |")
    md.append(f"| **Grade** | **{sbp_bhs['grade']}** | **{dbp_bhs['grade']}** | | | |\n")

    # Table 4: Bland-Altman
    md.append("## Bảng 4 — Bland-Altman Analysis\n")
    md.append("| | SBP | DBP |")
    md.append("|---|---|---|")
    md.append(f"| Mean bias (mmHg) | {sbp_ba['bias']:+.2f} | {dbp_ba['bias']:+.2f} |")
    md.append(f"| SD of differences | {sbp_ba['sd_diff']} | {dbp_ba['sd_diff']} |")
    md.append(f"| 95% LoA upper | {sbp_ba['loa_upper']:+.2f} | {dbp_ba['loa_upper']:+.2f} |")
    md.append(f"| 95% LoA lower | {sbp_ba['loa_lower']:+.2f} | {dbp_ba['loa_lower']:+.2f} |")
    md.append(f"| LoA width | {sbp_ba['loa_width']} | {dbp_ba['loa_width']} |")
    bias_sig_sbp = "p<0.05 (có bias)" if sbp_ba["bias_significant"] else "p≥0.05 (không bias)"
    bias_sig_dbp = "p<0.05 (có bias)" if dbp_ba["bias_significant"] else "p≥0.05 (không bias)"
    md.append(f"| Bias significance | {bias_sig_sbp} | {bias_sig_dbp} |\n")

    # Conclusion
    md.append("## Kết luận\n")
    aami_summary = []
    if sbp_aami["pass"] and dbp_aami["pass"]:
        aami_summary.append("✅ AAMI: PASS cho cả SBP và DBP")
    elif sbp_aami["pass"]:
        aami_summary.append("⚠️ AAMI: chỉ SBP pass")
    elif dbp_aami["pass"]:
        aami_summary.append("⚠️ AAMI: chỉ DBP pass")
    else:
        aami_summary.append("❌ AAMI: FAIL cả SBP và DBP")
    md.append(f"- {aami_summary[0]}")
    md.append(f"- BHS Grade: SBP={sbp_bhs['grade']}, DBP={dbp_bhs['grade']}")

    if sbp_metrics["mae"] <= 5 and dbp_metrics["mae"] <= 5:
        md.append("- ✅ MAE đạt chuẩn IEEE 1708 (≤5 mmHg)")
    else:
        md.append(f"- ❌ MAE vượt chuẩn IEEE 1708 (SBP={sbp_metrics['mae']}, DBP={dbp_metrics['mae']}). " +
                  "Đây là known limitation toàn ngành — Moulaeifard, Charlton & Strodthoff 2025 (arXiv:2502.19167) " +
                  "xác nhận 5 DL architectures (LeNet1D, XResNet1d50/101, Inception1D, S4) train PulseDB " +
                  "đều có cross-dataset calibration-free SBP MAE ~15–25 mmHg.")
    md.append("")

    # Citations
    md.append("## References\n")
    md.append("- AAMI/ANSI/ISO 81060-2:2018: Non-invasive sphygmomanometers — Clinical investigation")
    md.append("- O'Brien, E. et al. (1993). British Hypertension Society Protocol")
    md.append("- IEEE Std 1708-2014: Wearable Cuffless Blood Pressure Measuring Devices")
    md.append("- Bland, J.M. & Altman, D.G. (1986). Statistical methods for assessing agreement")
    md.append("- Moulaeifard, M., Charlton, P. H., & Strodthoff, N. (2025). Generalizable deep learning "
              "for photoplethysmography-based blood pressure estimation — a benchmarking study. "
              "arXiv:2502.19167 (PubMed PMID: 40959521)")
    md.append("")

    return "\n".join(md)


# ============================================================
# Main pipeline
# ============================================================
def evaluate(
    df: pd.DataFrame,
    output_dir: Path,
    note: str = "",
) -> Dict[str, object]:
    """Pipeline đánh giá đầy đủ. Trả về dict tất cả metrics + lưu files."""

    # Validate columns
    required = {"sbp_true", "sbp_pred", "dbp_true", "dbp_pred"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV thiếu cột bắt buộc: {missing}")

    # Drop NaN
    df = df.dropna(subset=list(required)).copy()
    if len(df) < 5:
        raise ValueError(f"Quá ít data sau drop NaN: N={len(df)}, cần ≥5")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Stats
    n_subjects = df["subject_id"].nunique() if "subject_id" in df.columns else len(df)
    n_recordings = len(df)
    print(f"[INFO] Loaded {n_recordings} recordings từ {n_subjects} subjects")

    # SBP metrics
    sbp_t = df["sbp_true"].to_numpy(dtype=float)
    sbp_p = df["sbp_pred"].to_numpy(dtype=float)
    sbp_metrics = compute_basic_metrics(sbp_t, sbp_p)
    sbp_aami = compute_aami_pass(sbp_metrics)
    sbp_bhs = compute_bhs_grade(sbp_t, sbp_p)
    sbp_ba = compute_bland_altman(sbp_t, sbp_p)

    # DBP metrics
    dbp_t = df["dbp_true"].to_numpy(dtype=float)
    dbp_p = df["dbp_pred"].to_numpy(dtype=float)
    dbp_metrics = compute_basic_metrics(dbp_t, dbp_p)
    dbp_aami = compute_aami_pass(dbp_metrics)
    dbp_bhs = compute_bhs_grade(dbp_t, dbp_p)
    dbp_ba = compute_bland_altman(dbp_t, dbp_p)

    # Plots
    plot_bland_altman(sbp_t, sbp_p, "SBP", output_dir / "bland_altman_sbp.png", note)
    plot_bland_altman(dbp_t, dbp_p, "DBP", output_dir / "bland_altman_dbp.png", note)
    plot_scatter(sbp_t, sbp_p, "SBP", output_dir / "scatter_sbp.png", note)
    plot_scatter(dbp_t, dbp_p, "DBP", output_dir / "scatter_dbp.png", note)
    plot_error_histogram(sbp_p - sbp_t, dbp_p - dbp_t, output_dir / "error_histogram.png")

    # Aggregate result
    result = {
        "n_subjects": n_subjects,
        "n_recordings": n_recordings,
        "sbp": {
            "metrics": sbp_metrics,
            "aami": sbp_aami,
            "bhs": sbp_bhs,
            "bland_altman": sbp_ba,
        },
        "dbp": {
            "metrics": dbp_metrics,
            "aami": dbp_aami,
            "bhs": dbp_bhs,
            "bland_altman": dbp_ba,
        },
        "note": note,
    }

    # Save JSON
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # Save Markdown
    md_report = generate_markdown_report(
        sbp_metrics, dbp_metrics,
        sbp_aami, dbp_aami,
        sbp_bhs, dbp_bhs,
        sbp_ba, dbp_ba,
        n_subjects, n_recordings, note,
    )
    with open(output_dir / "metrics_table.md", "w", encoding="utf-8") as f:
        f.write(md_report)

    # Console summary
    print()
    print("=" * 60)
    print("KẾT QUẢ ĐÁNH GIÁ")
    print("=" * 60)
    print(f"  N subjects = {n_subjects}, recordings = {n_recordings}")
    print()
    print("SBP:")
    print(f"  MAE = {sbp_metrics['mae']:.2f} mmHg")
    print(f"  ME ± SD = {sbp_metrics['me']:+.2f} ± {sbp_metrics['sd']:.2f}")
    print(f"  AAMI: {'PASS' if sbp_aami['pass'] else 'FAIL'} ({', '.join(sbp_aami['reasons'])})")
    print(f"  BHS Grade: {sbp_bhs['grade']} ({sbp_bhs['pct_within_5']}% ≤5, {sbp_bhs['pct_within_10']}% ≤10)")
    print()
    print("DBP:")
    print(f"  MAE = {dbp_metrics['mae']:.2f} mmHg")
    print(f"  ME ± SD = {dbp_metrics['me']:+.2f} ± {dbp_metrics['sd']:.2f}")
    print(f"  AAMI: {'PASS' if dbp_aami['pass'] else 'FAIL'} ({', '.join(dbp_aami['reasons'])})")
    print(f"  BHS Grade: {dbp_bhs['grade']} ({dbp_bhs['pct_within_5']}% ≤5, {dbp_bhs['pct_within_10']}% ≤10)")
    print()
    print(f"  Output: {output_dir}/")
    print(f"    - metrics.json")
    print(f"    - metrics_table.md")
    print(f"    - bland_altman_sbp.png + bland_altman_dbp.png")
    print(f"    - scatter_sbp.png + scatter_dbp.png")
    print(f"    - error_histogram.png")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Đánh giá BP estimation theo AAMI / BHS / Bland-Altman."
    )
    parser.add_argument(
        "--predictions", "-p", required=True, type=str,
        help="CSV file với cột: subject_id, sbp_true, sbp_pred, dbp_true, dbp_pred",
    )
    parser.add_argument(
        "--output", "-o", default="report", type=str,
        help="Folder output (default: report/)",
    )
    parser.add_argument(
        "--note", "-n", default="", type=str,
        help="Note để in vào report (vd: 'uncalibrated', 'calibrated', 'LOSO fold 3')",
    )
    args = parser.parse_args()

    csv_path = Path(args.predictions)
    if not csv_path.exists():
        print(f"[ERROR] File không tồn tại: {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    out_dir = Path(args.output)
    evaluate(df, out_dir, note=args.note)


if __name__ == "__main__":
    main()
