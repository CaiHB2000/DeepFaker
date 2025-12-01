# -*- coding: utf-8 -*-
"""
Run a long-lived propagation tracker that continuously polls Reddit for engagement updates.

Example:
    python -m p2p.tools.watch_propagation \
        --posts-jsonl tmp/reddit_seed_run_mass/reddit_posts.jsonl \
        --out-csv tmp/reddit_seed_run_mass/propagation_timeseries.csv \
        --interval 600 \
        --refresh-targets
"""
from __future__ import annotations

import argparse
import json
import random
import signal
import sys
import time
from pathlib import Path
from typing import List

from p2p.tools import track_propagation as tp


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Continuously poll Reddit posts and append engagement snapshots.")
    ap.add_argument(
        "--posts-jsonl",
        type=str,
        help="reddit_posts.jsonl emitted by run_reddit_seed (provides posting_id/post_id metadata).",
    )
    ap.add_argument(
        "--posts-summary",
        type=str,
        help="posts_summary.csv (or posts_summary_with_dfbench*.csv) to harvest posting IDs.",
    )
    ap.add_argument(
        "--ids-file",
        type=str,
        help="Optional newline-delimited list of posting IDs (platform:postid) or raw Reddit IDs.",
    )
    ap.add_argument(
        "--out-csv",
        type=str,
        default="propagation_timeseries.csv",
        help="Destination CSV; data is appended on every poll.",
    )
    ap.add_argument(
        "--interval",
        type=float,
        default=600.0,
        help="Seconds between polling rounds (default 10 minutes).",
    )
    ap.add_argument(
        "--jitter",
        type=float,
        default=60.0,
        help="Uniform jitter (+/- seconds) added to each sleep to avoid rigid scheduling.",
    )
    ap.add_argument(
        "--per-request-sleep",
        type=float,
        default=1.0,
        help="Delay (seconds) between individual post queries to respect rate limits (default 1.0).",
    )
    ap.add_argument(
        "--refresh-targets",
        action="store_true",
        help="Reload target list from input files before each poll (capture newly added posts).",
    )
    ap.add_argument(
        "--max-iterations",
        type=int,
        default=0,
        help="Optional limit on polling rounds (0 = infinite until Ctrl+C).",
    )
    ap.add_argument(
        "--status-json",
        type=str,
        default=None,
        help="Optional path to write watcher status metadata after each iteration.",
    )
    ap.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable per-iteration progress bar (default shows simple textual progress).",
    )
    return ap


def main() -> None:
    args = build_argparser().parse_args()
    tp.ensure_env()
    out_path = Path(args.out_csv).resolve()
    reddit = tp.init_reddit()

    def load_targets() -> List[tp.TargetPost]:
        return tp.resolve_targets(args)

    targets = load_targets()
    print(f"[watch] loaded {len(targets)} targets.")

    stop_requested = False

    def handle_signal(signum, frame):  # noqa: D401
        nonlocal stop_requested
        print(f"[watch] received signal {signum}, stopping after current cycle.")
        stop_requested = True

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    iteration = 0
    while True:
        iteration += 1
        if args.max_iterations and iteration > args.max_iterations:
            print("[watch] reached max_iterations, exiting.")
            break

        start_ts = time.time()
        if args.refresh_targets:
            targets = load_targets()
            print(f"[watch] refreshed targets → {len(targets)} posts.")

        snapshots = []
        show_progress = not args.no_progress
        if show_progress:
            sys.stdout.write(f"[watch] polling iteration {iteration}: 0/{len(targets)}    \r")
            sys.stdout.flush()
        for idx, target in enumerate(targets, start=1):
            try:
                snapshots.append(tp.fetch_submission(reddit, target))
            except Exception as exc:  # pragma: no cover
                print(f"[warn] fetch failed for {target.posting_id}: {exc}", file=sys.stderr)
            finally:
                # gentle throttle (~1 req/s) to respect API rate limits
                time.sleep(max(args.per_request_sleep, 0.0))
            if show_progress:
                sys.stdout.write(f"[watch] polling iteration {iteration}: {idx}/{len(targets)}    \r")
                sys.stdout.flush()

        if show_progress:
            sys.stdout.write("\n")
            sys.stdout.flush()

        if snapshots:
            tp.write_rows(out_path, snapshots)
            print(f"[watch] iteration {iteration}: wrote {len(snapshots)} rows -> {out_path}")
        else:
            print(f"[watch] iteration {iteration}: no snapshots captured.")

        if args.status_json:
            status = {
                "iteration": iteration,
                "snapshot_count": len(snapshots),
                "last_run_utc": time.time(),
                "targets": len(targets),
            }
            status_path = Path(args.status_json).resolve()
            status_path.parent.mkdir(parents=True, exist_ok=True)
            status_path.write_text(__import__("json").dumps(status, indent=2), encoding="utf-8")

        if stop_requested:
            break

        elapsed = time.time() - start_ts
        sleep_base = max(args.interval - elapsed, 1.0)
        jitter = random.uniform(-args.jitter, args.jitter)
        sleep_time = max(1.0, sleep_base + jitter)
        print(f"[watch] sleeping {sleep_time:.1f}s (base {sleep_base:.1f}s, jitter {jitter:.1f}s)")
        time.sleep(sleep_time)


if __name__ == "__main__":
    main()
