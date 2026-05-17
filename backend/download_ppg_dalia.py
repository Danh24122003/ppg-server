"""Download PPG-DaLiA UCI dataset 495 for cross-dataset SQA validation.

Source: https://archive.ics.uci.edu/dataset/495/ppg+dalia
License: CC BY 4.0
Reference: Reiss et al. 2019, Sensors 19(14):3079.

Saves to: external_data/ppg_dalia/
After extract:
    external_data/ppg_dalia/PPG_FieldStudy/S{1..15}/S{i}.pkl
"""
from __future__ import annotations

import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path


ZIP_URL = "https://archive.ics.uci.edu/static/public/495/ppg+dalia.zip"
# Path resolved relative to PROJECT ROOT (parent of "Backend code" folder)
# matching eval_sqa_ppg_dalia.py convention. Allows running from any cwd.
ROOT = Path(__file__).resolve().parent.parent
TARGET_DIR = ROOT / "external_data" / "ppg_dalia"
TIMEOUT = 600  # seconds — large file


def _report_hook(start_time: float):
    last = [0.0]

    def hook(block_num: int, block_size: int, total_size: int):
        downloaded = block_num * block_size
        now = time.time()
        if now - last[0] < 2.0 and downloaded < total_size:
            return
        last[0] = now
        if total_size > 0:
            pct = downloaded * 100.0 / total_size
            mb_done = downloaded / 1024 / 1024
            mb_total = total_size / 1024 / 1024
            elapsed = now - start_time
            speed = mb_done / max(elapsed, 0.1)
            print(
                f"  {pct:5.1f}%  {mb_done:7.1f} / {mb_total:7.1f} MB"
                f"  {speed:5.2f} MB/s  ({elapsed:.0f}s)",
                flush=True,
            )
        else:
            mb_done = downloaded / 1024 / 1024
            print(f"  {mb_done:7.1f} MB", flush=True)

    return hook


def download_zip(zip_path: Path) -> bool:
    if zip_path.exists() and zip_path.stat().st_size > 100_000_000:
        print(f"[exists] {zip_path} ({zip_path.stat().st_size/1024/1024:.1f} MB)")
        return True
    print(f"[download] {ZIP_URL}")
    print(f"[target]   {zip_path}")
    t0 = time.time()
    try:
        urllib.request.urlretrieve(ZIP_URL, zip_path, _report_hook(t0))
    except urllib.error.HTTPError as e:
        print(f"[error] HTTP {e.code}: {e.reason}")
        return False
    except urllib.error.URLError as e:
        print(f"[error] URLError: {e.reason}")
        return False
    except Exception as e:
        print(f"[error] {type(e).__name__}: {e}")
        return False
    elapsed = time.time() - t0
    size_mb = zip_path.stat().st_size / 1024 / 1024
    print(f"[ok] downloaded {size_mb:.1f} MB in {elapsed:.0f}s")
    return True


def extract_zip(zip_path: Path, target_dir: Path, subjects: list[int] | None) -> int:
    """Extract pickle files. If subjects given, only extract S{i}/S{i}.pkl for those.
    Returns count of pickle files extracted.
    """
    print(f"[extract] {zip_path} -> {target_dir}")
    n_pkl = 0
    with zipfile.ZipFile(zip_path) as zf:
        members = zf.namelist()
        if subjects is None:
            wanted = members
        else:
            wanted_subjs = {f"S{i}" for i in subjects}
            wanted = []
            for m in members:
                # match patterns: PPG_FieldStudy/S{i}/... and any readme/txt
                low = m.lower()
                if low.endswith(".pdf") or low.endswith("readme.txt") or low.endswith(".md"):
                    wanted.append(m)
                    continue
                parts = m.split("/")
                # find any path component that exactly matches "S{i}"
                if any(p in wanted_subjs for p in parts):
                    wanted.append(m)
        for m in wanted:
            try:
                zf.extract(m, target_dir)
                if m.endswith(".pkl"):
                    n_pkl += 1
            except Exception as e:
                print(f"  [warn] cannot extract {m}: {e}")
    print(f"[ok] extracted {len(wanted)} entries ({n_pkl} pickle files)")
    return n_pkl


def main() -> int:
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = TARGET_DIR / "ppg+dalia.zip"

    # Subjects subset: --subjects 1,2,3,4,5  (default: all 15)
    subjects: list[int] | None = None
    args = sys.argv[1:]
    for i, a in enumerate(args):
        if a == "--subjects" and i + 1 < len(args):
            subjects = [int(x) for x in args[i + 1].split(",") if x.strip().isdigit()]
        if a == "--subset":
            subjects = [1, 2, 3, 4, 5]

    if not download_zip(zip_path):
        return 1

    if subjects is not None:
        print(f"[subset] extracting only subjects: {subjects}")
    n_pkl = extract_zip(zip_path, TARGET_DIR, subjects)
    if n_pkl == 0:
        print("[warn] no pickle files extracted")
        return 2

    # Resolve PPG_FieldStudy directory
    found = list(TARGET_DIR.rglob("S*.pkl"))
    print(f"[summary] {len(found)} pickle files in {TARGET_DIR}")
    for p in sorted(found)[:10]:
        size_mb = p.stat().st_size / 1024 / 1024
        print(f"  {p.relative_to(TARGET_DIR)}  ({size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
