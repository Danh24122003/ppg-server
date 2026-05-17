"""
Sample 50 random pseudo-labeled batches stratified by (dataset, label).
Generate matplotlib PNG plots for user manual labeling (blind — pseudo_label
hidden in image text, only stored in manifest.csv).

Output:
  backend/spot_check/
    batch_001.png ... batch_050.png
    manifest.csv  (batch_idx, file, dataset, subject_id, window_idx,
                   t_start_s, pseudo_label, manual_label)

Usage:
  python "backend/sample_pseudo_label_batches.py"
"""
from __future__ import annotations

import gc
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend (no display required)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent

PSEUDO_CSV = ROOT / "pseudo_labels.csv"
SPOT_DIR = ROOT / "spot_check"

BP_CLEANED_DIR = PROJECT_ROOT / "collected_data" / "cleaned"
SQA_DIR = PROJECT_ROOT / "collected_data" / "sqa_experiment"
BIDMC_DIR = PROJECT_ROOT / "external_data" / "bidmc"
DALIA_DIR = PROJECT_ROOT / "external_data" / "ppg_dalia" / "PPG_FieldStudy"

SEED = 42
N_SAMPLES = 50
DATASETS = ["BP-protocol", "S09-SQA", "BIDMC", "PPG-DaLiA"]
LABELS = ["excellent", "acceptable", "unfit"]


# ============================================================
# Signal loaders (subset of generate_pseudo_labels loaders, scoped to this tool)
# ============================================================
def _load_bp_or_quoc(path: Path) -> np.ndarray:
    df = pd.read_csv(
        path,
        comment="#",
        usecols=lambda c: c in {"timestamp_ms", "ir", "red"},
    )
    if "ir" not in df.columns or df.empty:
        raise ValueError(f"missing ir column or empty: {path.name}")
    return df["ir"].to_numpy(dtype=float)


def _load_bidmc(path: Path) -> np.ndarray:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    if "PLETH" not in df.columns:
        raise ValueError(f"PLETH not found in {path.name}")
    return df["PLETH"].to_numpy(dtype=float)


def _load_dalia(pkl_path: Path) -> np.ndarray:
    with open(pkl_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    bvp = np.asarray(data["signal"]["wrist"]["BVP"], dtype=float).reshape(-1).copy()
    del data
    gc.collect()
    return bvp


def _resolve_path(dataset: str, file_name: str) -> Path:
    if dataset == "BP-protocol":
        return BP_CLEANED_DIR / file_name
    if dataset == "S09-SQA":
        return SQA_DIR / file_name
    if dataset == "BIDMC":
        return BIDMC_DIR / file_name
    if dataset == "PPG-DaLiA":
        # DaLiA file name in pseudo_labels.csv is "S1.pkl" -> located at .../PPG_FieldStudy/S1/S1.pkl
        stem = Path(file_name).stem
        return DALIA_DIR / stem / file_name
    raise ValueError(f"unknown dataset: {dataset}")


def _load_signal(dataset: str, file_name: str) -> np.ndarray:
    p = _resolve_path(dataset, file_name)
    if dataset in ("BP-protocol", "S09-SQA"):
        return _load_bp_or_quoc(p)
    if dataset == "BIDMC":
        return _load_bidmc(p)
    if dataset == "PPG-DaLiA":
        return _load_dalia(p)
    raise ValueError(f"unknown dataset: {dataset}")


# ============================================================
# Stratified sampler
# ============================================================
def stratified_sample(df: pd.DataFrame, n_total: int, seed: int) -> pd.DataFrame:
    """Sample n_total rows stratified by (dataset, label).

    Allocate ~n_total/4 to each of 4 datasets, then split among labels
    proportional to availability. Falls back to dataset-level random sampling
    if a label is empty in that dataset.
    """
    rng = np.random.default_rng(seed)
    per_dataset = max(1, n_total // len(DATASETS))

    chunks: list[pd.DataFrame] = []
    for dataset in DATASETS:
        sub = df[df["dataset"] == dataset]
        if sub.empty:
            continue
        if len(sub) <= per_dataset:
            chunks.append(sub)
            continue

        # Within dataset: try to sample equally from each available label
        avail_labels = [lab for lab in LABELS if (sub["label"] == lab).any()]
        per_label = max(1, per_dataset // max(1, len(avail_labels)))
        for lab in avail_labels:
            lab_sub = sub[sub["label"] == lab]
            k = min(per_label, len(lab_sub))
            if k == 0:
                continue
            picks = lab_sub.sample(n=k, random_state=int(rng.integers(0, 2**31 - 1)))
            chunks.append(picks)

    sampled = pd.concat(chunks, ignore_index=True)
    if len(sampled) > n_total:
        sampled = sampled.sample(n=n_total, random_state=seed).reset_index(drop=True)
    elif len(sampled) < n_total:
        # Top up with random rows from full df not yet picked
        already = set(zip(sampled["file"], sampled["window_idx"]))
        pool = df[~df.set_index(["file", "window_idx"]).index.isin(already)]
        deficit = n_total - len(sampled)
        if len(pool) >= deficit:
            extra = pool.sample(n=deficit, random_state=seed)
            sampled = pd.concat([sampled, extra], ignore_index=True)

    # Shuffle so blind labeling order is randomized
    sampled = sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return sampled


# ============================================================
# Plotting
# ============================================================
def plot_window(
    signal: np.ndarray,
    fs: int,
    t_start_s: float,
    t_end_s: float,
    out_path: Path,
    batch_idx: int,
    dataset: str,
) -> None:
    """Plot raw + filtered + peaks for one window. Pseudo_label HIDDEN."""
    start = int(t_start_s * fs)
    end = int(t_end_s * fs)
    end = min(end, len(signal))
    if end <= start:
        # Edge case: window beyond signal length
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f"empty window\nstart={start} end={end} len={len(signal)}",
                ha="center", va="center", transform=ax.transAxes)
        fig.savefig(out_path, dpi=80)
        plt.close(fig)
        return

    raw = signal[start:end].astype(float)
    n = len(raw)
    t = np.arange(n) / fs

    # Bandpass [0.5, 4] Hz, order 4 (mirrors generate_pseudo_labels)
    nyq = 0.5 * fs
    if nyq <= 4.0:
        # fs too low for a 4 Hz upper cutoff (shouldn't happen here)
        filtered = raw - np.mean(raw)
    else:
        b, a = butter(4, [0.5, 4.0], btype="band", fs=fs)
        filtered = filtfilt(b, a, raw)

    peaks_idx, _ = find_peaks(filtered, distance=int(fs * 0.3))

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # P1 fix: blind dataset hint via 2-letter code → reduce rater bias
    # (BIDMC ICU vs MAX30102 reflectance vs Empatica wrist could bias rater)
    # Mapping stored in manifest.csv (dataset col), NOT in plot.
    dataset_token = f"D{abs(hash(dataset)) % 100:02d}"

    axes[0].plot(t, raw, color="gray", linewidth=0.8)
    axes[0].set_title(f"Batch {batch_idx:03d}  |  source {dataset_token}  |  raw PPG")
    axes[0].set_ylabel("raw amplitude")
    axes[0].grid(alpha=0.3)

    axes[1].plot(t, filtered, color="steelblue", linewidth=0.9)
    if len(peaks_idx) > 0:
        axes[1].plot(t[peaks_idx], filtered[peaks_idx], "rx", markersize=8, label=f"{len(peaks_idx)} peaks")
        axes[1].legend(loc="upper right")
    axes[1].set_title("Filtered [0.5-4 Hz] + detected peaks")
    axes[1].set_xlabel(f"time within window (s)  |  fs={fs} Hz  |  duration {n/fs:.1f}s")
    axes[1].set_ylabel("filtered amplitude")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=80)
    plt.close(fig)


# ============================================================
# Main
# ============================================================
def main() -> int:
    if not PSEUDO_CSV.exists():
        print(f"[error] {PSEUDO_CSV} not found")
        return 1

    df = pd.read_csv(PSEUDO_CSV)
    print(f"Loaded {len(df)} pseudo-labeled rows from {PSEUDO_CSV.name}")

    sampled = stratified_sample(df, N_SAMPLES, SEED)
    print(f"Stratified sample: {len(sampled)} rows")
    print(sampled.groupby(["dataset", "label"]).size().to_string())

    SPOT_DIR.mkdir(exist_ok=True)

    # Cache loaded signals (file -> array) to avoid re-loading the same file
    signal_cache: dict[tuple[str, str], np.ndarray] = {}

    manifest_rows: list[dict] = []
    n_ok = 0
    n_fail = 0
    for i, row in sampled.iterrows():
        batch_idx = int(i) + 1
        dataset = str(row["dataset"])
        file_name = str(row["file"])
        fs = int(row["fs"])
        t_start = float(row["t_start_s"])
        t_end = float(row["t_end_s"])

        out_png = SPOT_DIR / f"batch_{batch_idx:03d}.png"

        cache_key = (dataset, file_name)
        try:
            if cache_key not in signal_cache:
                signal_cache[cache_key] = _load_signal(dataset, file_name)
            sig = signal_cache[cache_key]
            plot_window(sig, fs, t_start, t_end, out_png, batch_idx, dataset)
            n_ok += 1
            # P0 fix: only append manifest row when plot succeeded (no phantom rows)
            manifest_rows.append({
                "batch_idx": batch_idx,
                "file": file_name,
                "dataset": dataset,
                "subject_id": row["subject_id"],
                "window_idx": int(row["window_idx"]),
                "t_start_s": t_start,
                "t_end_s": t_end,
                "fs": fs,
                "pseudo_label": str(row["label"]),
                "manual_label": "",
            })
        except Exception as e:
            print(f"  [fail] batch {batch_idx} {dataset} {file_name}: {type(e).__name__}: {e}")
            n_fail += 1
            # NO append → manifest stays consistent với PNG files actually written

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_path = SPOT_DIR / "manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)

    print()
    print(f"[ok] {n_ok} plots written, {n_fail} failed")
    print(f"[ok] manifest -> {manifest_path}")
    print(f"[ok] output dir: {SPOT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
