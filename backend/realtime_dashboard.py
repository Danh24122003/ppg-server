"""Realtime 4-panel PPG dashboard.

Standalone observer — polls /api/ppg/latest/{device_id} at 1 Hz and tails a CSV
written by log_ppg_local.py. Does NOT modify firmware, backend, ML server, or
log_ppg_local.py. Run alongside an existing pipeline.

Usage:
    python "backend/realtime_dashboard.py" \\
        --device-id esp32_01 \\
        --server http://localhost:8000 \\
        [--csv "collected_data/<file>.csv"] \\
        [--token <X-Device-Token>]
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from typing import Any, Optional

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import pandas as pd
import requests

POLL_INTERVAL_MS = 1000
SPO2_HISTORY_LEN = 60
WAVEFORM_SAMPLES = 1000
HTTP_TIMEOUT_S = 2.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Realtime 4-panel PPG dashboard (HR/SpO2/HRV/BP)."
    )
    p.add_argument("--device-id", required=True, help="ESP32 device_id to poll")
    p.add_argument(
        "--server",
        default="http://localhost:8000",
        help="Backend or ML server base URL (default: http://localhost:8000)",
    )
    p.add_argument("--csv", default=None, help="Path to log_ppg_local.py CSV (optional)")
    p.add_argument("--token", default=None, help="X-Device-Token header value (optional)")
    return p.parse_args()


class DashboardState:
    def __init__(self, server: str, device_id: str, csv_path: Optional[str], token: Optional[str]):
        self.server = server.rstrip("/")
        self.device_id = device_id
        self.csv_path = csv_path
        self.headers = {"X-Device-Token": token} if token else {}
        self.spo2_history: deque[float] = deque(maxlen=SPO2_HISTORY_LEN)
        self.spo2_times: deque[float] = deque(maxlen=SPO2_HISTORY_LEN)
        self.latest: Optional[dict[str, Any]] = None
        self.connected = False
        self.last_error = ""
        self.poll_count = 0
        self.start_time = time.time()

    def poll_latest(self) -> None:
        url = f"{self.server}/api/ppg/latest/{self.device_id}"
        try:
            r = requests.get(url, headers=self.headers, timeout=HTTP_TIMEOUT_S)
            if r.status_code == 200:
                self.latest = r.json()
                self.connected = True
                self.last_error = ""
                self.poll_count += 1
                spo2 = self.latest.get("spo2")
                if isinstance(spo2, (int, float)) and 70 <= spo2 <= 100:
                    self.spo2_history.append(float(spo2))
                    self.spo2_times.append(time.time() - self.start_time)
            else:
                self.connected = False
                self.last_error = f"HTTP {r.status_code}"
        except requests.RequestException as e:
            self.connected = False
            self.last_error = type(e).__name__

    def read_csv_tail(self) -> Optional[pd.DataFrame]:
        if not self.csv_path:
            return None
        try:
            df = pd.read_csv(self.csv_path, comment="#")
            if "ir" not in df.columns or "timestamp_ms" not in df.columns:
                return None
            return df.tail(WAVEFORM_SAMPLES)
        except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
            return None


def draw_waveform(
    ax: plt.Axes,
    df: Optional[pd.DataFrame],
    csv_path: Optional[str],
    state_latest: Optional[dict] = None,
) -> None:
    ax.clear()
    ax.set_title("PPG Waveform (filtered IR, last batch)")
    sig = (state_latest or {}).get("filtered_signal") if state_latest else None
    if isinstance(sig, list) and len(sig) > 10:
        fs = 100
        t_sec = [i / fs for i in range(len(sig))]
        ax.plot(t_sec, sig, color="tab:blue", linewidth=0.9)
        peaks = (state_latest or {}).get("peaks") or []
        if isinstance(peaks, list) and peaks:
            peak_t = [p / fs for p in peaks if 0 <= p < len(sig)]
            peak_y = [sig[p] for p in peaks if 0 <= p < len(sig)]
            ax.plot(peak_t, peak_y, "ro", markersize=5, label=f"{len(peak_t)} peaks")
            ax.legend(loc="upper right", fontsize=8)
        ax.set_xlabel("Time (s, last batch)")
        ax.set_ylabel("Filtered IR (AC)")
        ax.grid(alpha=0.3)
        return
    if df is not None and not df.empty:
        ts = df["timestamp_ms"].to_numpy()
        t_sec = (ts - ts[0]) / 1000.0
        ax.plot(t_sec, df["ir"].to_numpy(), color="tab:blue", linewidth=0.8)
        ax.set_xlabel("Time (s, last 10s)")
        ax.set_ylabel("IR ADC (raw)")
        ax.grid(alpha=0.3)
        return
    msg = (
        "Waiting for waveform data\n(server response needs filtered_signal field)"
        if state_latest is not None
        else (
            "CSV not provided\nstart log_ppg_local.py first"
            if not csv_path
            else f"CSV unreadable:\n{csv_path}"
        )
    )
    ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])


def draw_spo2(ax: plt.Axes, state: DashboardState) -> None:
    ax.clear()
    ax.set_title("SpO2 Timeline (last 60s)")
    ax.set_ylim(85, 100)
    ax.set_ylabel("SpO2 (%)")
    ax.set_xlabel("Time since start (s)")
    ax.grid(alpha=0.3)
    if not state.connected:
        ax.text(0.5, 0.5, f"DISCONNECTED\n{state.last_error}", ha="center", va="center",
                transform=ax.transAxes, fontsize=14, color="red")
        return
    if state.spo2_history:
        ax.plot(list(state.spo2_times), list(state.spo2_history), "o-", color="tab:green",
                markersize=3, linewidth=1)
        latest_spo2 = state.spo2_history[-1]
        sq = state.latest.get("signal_quality", "n/a") if state.latest else "n/a"
        ax.annotate(f"{latest_spo2:.1f}% [{sq}]",
                    xy=(state.spo2_times[-1], latest_spo2),
                    xytext=(5, 5), textcoords="offset points", fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    else:
        ax.text(0.5, 0.5, "Waiting for first SpO2 reading...", ha="center", va="center",
                transform=ax.transAxes, fontsize=11)


def draw_hrv(ax: plt.Axes, state: DashboardState) -> None:
    ax.clear()
    ax.set_title("HRV (latest)")
    if not state.connected or not state.latest:
        ax.text(0.5, 0.5, "DISCONNECTED" if not state.connected else "No data",
                ha="center", va="center", transform=ax.transAxes, fontsize=14,
                color="red" if not state.connected else "gray")
        ax.set_xticks([])
        ax.set_yticks([])
        return
    hrv = state.latest.get("hrv") or {}
    metrics = {
        "SDNN (ms)": hrv.get("sdnn_ms"),
        "RMSSD (ms)": hrv.get("rmssd_ms"),
        "pNN50 (%)": hrv.get("pnn50_pct"),
        "pNN20 (%)": hrv.get("pnn20_pct"),
    }
    names = list(metrics.keys())
    values = [v if isinstance(v, (int, float)) else 0.0 for v in metrics.values()]
    reliability = hrv.get("reliability", "low")
    color_map = {"high": "tab:green", "medium": "tab:orange", "low": "tab:red"}
    bar_color = color_map.get(reliability, "gray")
    ax.bar(names, values, color=bar_color)
    ax.set_ylabel("Value")
    ax.set_title(f"HRV (reliability: {reliability}, rr_count: {hrv.get('rr_count', 0)})")
    for i, v in enumerate(values):
        ax.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=9)


def draw_bp_status(ax: plt.Axes, state: DashboardState) -> None:
    ax.clear()
    ax.set_title("BP + Status")
    ax.set_xticks([])
    ax.set_yticks([])
    uptime = int(time.time() - state.start_time)
    poll_rate = state.poll_count / max(uptime, 1)
    lines = [
        f"device:    {state.device_id}",
        f"server:    {state.server}",
        f"polls/sec: {poll_rate:.2f}  (target 1.0)",
        f"uptime:    {uptime}s",
        f"connected: {state.connected}",
        "",
    ]
    if state.connected and state.latest:
        ml = state.latest.get("ml_predictions") or {}
        bp = ml.get("blood_pressure") or {}
        sbp = bp.get("systolic")
        dbp = bp.get("diastolic")
        if sbp is not None and dbp is not None:
            lines.append(f"BP:        {sbp}/{dbp} mmHg")
        else:
            lines.append("BP:        n/a (Backend server, no ML)")
        lines.append(f"calibrated: {state.latest.get('calibrated', 'n/a')}")
        hr = state.latest.get("heart_rate")
        if isinstance(hr, (int, float)):
            lines.append(f"HR:        {hr:.1f} BPM")
    else:
        lines.append("BP:        DISCONNECTED")
        lines.append(f"error:     {state.last_error}")
    ax.text(0.05, 0.95, "\n".join(lines), ha="left", va="top",
            transform=ax.transAxes, fontsize=11, family="monospace")


def main() -> int:
    args = parse_args()
    state = DashboardState(args.server, args.device_id, args.csv, args.token)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(f"PPG Realtime Monitor - {args.device_id}", fontsize=14, fontweight="bold")
    ax_wave, ax_spo2 = axes[0, 0], axes[0, 1]
    ax_hrv, ax_bp = axes[1, 0], axes[1, 1]

    def update(_frame: int) -> None:
        state.poll_latest()
        df = state.read_csv_tail()
        draw_waveform(ax_wave, df, state.csv_path, state.latest)
        draw_spo2(ax_spo2, state)
        draw_hrv(ax_hrv, state)
        draw_bp_status(ax_bp, state)
        fig.tight_layout(rect=[0, 0, 1, 0.96])

    anim = animation.FuncAnimation(fig, update, interval=POLL_INTERVAL_MS, cache_frame_data=False)
    # Keep reference to prevent GC of animation object.
    fig._anim_ref = anim  # type: ignore[attr-defined]

    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
