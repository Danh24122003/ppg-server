"""Download BIDMC PhysioNet CSV files for cross-dataset SQA validation.

Source: https://physionet.org/content/bidmc/1.0.0/
License: PhysioNet modified, free for research.

Saves to: external_data/bidmc/bidmc_NN_Signals.csv (NN=01..53)
"""
import sys
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


URL_TEMPLATE = "https://physionet.org/files/bidmc/1.0.0/bidmc_csv/bidmc_{:02d}_Signals.csv"
OUT_DIR = Path("external_data/bidmc")
TIMEOUT = 90  # seconds per file
MAX_WORKERS = 6  # parallel downloads


def download_one(idx: int) -> tuple[bool, int, str]:
    """Download bidmc_NN_Signals.csv. Returns (ok, bytes, msg)."""
    url = URL_TEMPLATE.format(idx)
    out_path = OUT_DIR / f"bidmc_{idx:02d}_Signals.csv"

    if out_path.exists() and out_path.stat().st_size > 1_000_000:
        return True, out_path.stat().st_size, "exists"

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "PPGMonitor-Thesis/1.0"})
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            data = resp.read()
        with open(out_path, "wb") as f:
            f.write(data)
        return True, len(data), "downloaded"
    except urllib.error.HTTPError as e:
        return False, 0, f"HTTP {e.code}"
    except urllib.error.URLError as e:
        return False, 0, f"URLError: {e.reason}"
    except Exception as e:
        return False, 0, f"Exception: {type(e).__name__}: {e}"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    n_total = 53
    if len(sys.argv) > 1 and sys.argv[1] == "--subset":
        n_total = 10
        print(f"SUBSET MODE: downloading first {n_total} recordings only")

    print(f"Downloading {n_total} BIDMC recordings to {OUT_DIR.resolve()}")
    print(f"Parallel workers: {MAX_WORKERS}, timeout per file: {TIMEOUT}s")
    print("-" * 70, flush=True)

    ok_count = 0
    fail_count = 0
    total_bytes = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        future_to_idx = {ex.submit(download_one, i): i for i in range(1, n_total + 1)}
        for fut in as_completed(future_to_idx):
            i = future_to_idx[fut]
            try:
                ok, nbytes, msg = fut.result()
            except Exception as e:
                ok, nbytes, msg = False, 0, f"future-exception: {e}"
            if ok:
                ok_count += 1
                total_bytes += nbytes
                print(f"  [{i:02d}/{n_total}] OK ({nbytes/1024/1024:.2f} MB) [{msg}]", flush=True)
            else:
                fail_count += 1
                print(f"  [{i:02d}/{n_total}] FAIL: {msg}", flush=True)

    elapsed = time.time() - t0
    print("-" * 70)
    print(f"Summary: {ok_count} OK, {fail_count} FAIL")
    print(f"Total: {total_bytes/1024/1024:.1f} MB in {elapsed:.1f}s")
    return 0 if ok_count > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
