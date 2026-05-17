"""
Interactive CLI to manually label the 50 spot-check PNG plots.

Reads spot_check/manifest.csv, opens each unlabeled PNG in the OS default
viewer, prompts user for a label, persists progress after every entry.

Usage:
  python "backend/label_spot_check_cli.py"

Keys:
  e = excellent
  a = acceptable
  u = unfit
  s = skip (leave blank, ask again next run)
  q = quit (saves progress so far)

Manual labels are blind: pseudo_label column is loaded but NOT shown in the
prompt, to keep Cohen's kappa unbiased.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
SPOT_DIR = ROOT / "spot_check"
MANIFEST = SPOT_DIR / "manifest.csv"

LABEL_MAP = {
    "e": "excellent",
    "a": "acceptable",
    "u": "unfit",
}


def _open_image(png_path: Path) -> None:
    """Open image in the OS default viewer (Windows / macOS / Linux)."""
    try:
        if sys.platform == "win32":
            subprocess.Popen(["cmd", "/c", "start", "", str(png_path)], shell=False)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(png_path)])
        else:
            subprocess.Popen(["xdg-open", str(png_path)])
    except Exception as e:
        print(f"  [warn] could not open viewer: {type(e).__name__}: {e}")


def main() -> int:
    if not MANIFEST.exists():
        print(f"[error] {MANIFEST} not found. Run sample_pseudo_label_batches.py first.")
        return 1

    df = pd.read_csv(MANIFEST)
    if "manual_label" not in df.columns:
        df["manual_label"] = ""
    df["manual_label"] = df["manual_label"].fillna("")

    n_total = len(df)
    n_done = int((df["manual_label"].astype(str).str.len() > 0).sum())
    n_remain = n_total - n_done
    print(f"Total batches: {n_total}")
    print(f"Already labeled: {n_done}")
    print(f"Remaining: {n_remain}")
    print()
    print("Keys: e=excellent  a=acceptable  u=unfit  s=skip  q=quit")
    print()

    quit_requested = False
    for idx, row in df.iterrows():
        if quit_requested:
            break
        existing = str(row["manual_label"]).strip()
        if existing:
            continue

        batch_idx = int(row["batch_idx"])
        png_path = SPOT_DIR / f"batch_{batch_idx:03d}.png"
        if not png_path.exists():
            print(f"  [skip] batch {batch_idx}: {png_path.name} missing")
            continue

        _open_image(png_path)

        prompt = f"[{batch_idx}/{n_total}] {row['dataset']} | {row['file']}  label > "
        try:
            choice = input(prompt).strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n[interrupt] saving progress...")
            break

        if choice == "q":
            print("[quit] saving progress...")
            quit_requested = True
            break
        if choice == "s" or choice == "":
            continue
        if choice not in LABEL_MAP:
            print(f"  [warn] unknown key '{choice}', skipping")
            continue

        df.at[idx, "manual_label"] = LABEL_MAP[choice]
        df.to_csv(MANIFEST, index=False)
        print(f"  saved: {LABEL_MAP[choice]}")

    n_done_final = int((df["manual_label"].astype(str).str.len() > 0).sum())
    print()
    print(f"[done] manifest saved -> {MANIFEST}")
    print(f"[done] labeled {n_done_final}/{n_total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
