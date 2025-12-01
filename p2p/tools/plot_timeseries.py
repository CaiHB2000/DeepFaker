# -*- coding: utf-8 -*-
"""
Quickly visualise Reddit propagation time series stored in propagation_timeseries.csv.

Usage:
    python -m p2p.tools.plot_timeseries \
        --timeseries tmp/reddit_seed_run_mass/propagation_timeseries.csv \
        --posting-id reddit:1oc4bds \
        --out-dir tmp/reddit_seed_run_mass/analysis/plots
"""
from __future__ import annotations

import argparse
import csv
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Plot score/comment trajectories for tracked Reddit posts.")
    ap.add_argument("--timeseries", required=True, help="propagation_timeseries.csv produced by watch_propagation.")
    ap.add_argument(
        "--posting-id",
        action="append",
        help="Posting ID(s) to plot (e.g., reddit:abc123). Specify multiple times or comma-separated.",
    )
    ap.add_argument(
        "--top-n",
        type=int,
        default=0,
        help="Fallback: choose top-N posts by final score if --posting-id not supplied.",
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Directory to save plots (defaults to same folder as CSV).",
    )
    ap.add_argument(
        "--dpi",
        type=int,
        default=120,
        help="Figure DPI for saved PNGs (default 120).",
    )
    return ap.parse_args()


def load_timeseries(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return [dict(row) for row in csv.DictReader(f)]


def select_post_ids(rows: List[Dict[str, str]], args: argparse.Namespace) -> List[str]:
    explicit_ids: List[str] = []
    if args.posting_id:
        for item in args.posting_id:
            explicit_ids.extend([token.strip() for token in item.split(",") if token.strip()])
    if explicit_ids:
        return explicit_ids
    if args.top_n > 0:
        latest: Dict[str, Dict[str, str]] = {}
        for row in rows:
            pid = row["posting_id"]
            snapshot = float(row.get("snapshot_utc") or 0.0)
            if pid not in latest or snapshot > float(latest[pid].get("snapshot_utc") or 0.0):
                latest[pid] = row
        ordered = sorted(
            latest.items(),
            key=lambda kv: float(kv[1].get("score") or 0.0),
            reverse=True,
        )
        return [pid for pid, _ in ordered[: args.top_n]]
    unique_ids = sorted({row["posting_id"] for row in rows})
    if not unique_ids:
        raise RuntimeError("No posting IDs found in time series; supply --posting-id explicitly.")
    return [unique_ids[0]]


def convert_timestamp(ts: float) -> datetime:
    return datetime.utcfromtimestamp(ts)


def plot_post(
    rows: Sequence[Dict[str, str]],
    posting_id: str,
    out_dir: Path,
    dpi: int,
) -> Optional[Path]:
    series = [row for row in rows if row["posting_id"] == posting_id]
    if not series:
        print(f"[plot] posting_id {posting_id} not found in time series.")
        return None
    series_sorted = sorted(series, key=lambda r: float(r.get("snapshot_utc") or 0.0))
    times = [convert_timestamp(float(row.get("snapshot_utc") or 0.0)) for row in series_sorted]
    scores = [float(row.get("score") or 0.0) for row in series_sorted]
    comments = [float(row.get("num_comments") or 0.0) for row in series_sorted]

    fig, ax1 = plt.subplots(figsize=(9, 4.5), dpi=dpi)
    ax1.plot(times, scores, color="tab:blue", marker="o", label="Score")
    ax1.set_ylabel("Score", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.set_xlabel("Snapshot UTC")

    ax2 = ax1.twinx()
    ax2.plot(times, comments, color="tab:orange", marker="s", label="Comments")
    ax2.set_ylabel("Comments", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    title = f"{posting_id} | snapshots={len(times)}"
    ax1.set_title(title)
    fig.autofmt_xdate()

    # combine legends
    lines = ax1.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="upper left")

    out_dir.mkdir(parents=True, exist_ok=True)
    safe_name = posting_id.replace(":", "_")
    out_path = out_dir / f"{safe_name}_timeseries.png"
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[plot] saved {out_path}")
    return out_path


def main() -> None:
    args = parse_args()
    csv_path = Path(args.timeseries).resolve()
    rows = load_timeseries(csv_path)
    out_dir = Path(args.out_dir or os.path.join(csv_path.parent, "plots")).resolve()
    targets = select_post_ids(rows, args)
    for pid in targets:
        plot_post(rows, pid, out_dir, dpi=args.dpi)


if __name__ == "__main__":
    main()
