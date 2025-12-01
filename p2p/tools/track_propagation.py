# -*- coding: utf-8 -*-
"""
Poll Reddit for current engagement metrics of a post list and append to a time series table.

Usage example:
    python -m p2p.tools.track_propagation \
        --posts-jsonl tmp/reddit_seed_run_mass/reddit_posts.jsonl \
        --out-csv tmp/reddit_seed_run_mass/propagation_timeseries.csv

Run the command periodically (cron/systemd) to build a history of score/comment trajectories.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import praw


@dataclass
class TargetPost:
    posting_id: str
    post_id: str
    subreddit: Optional[str] = None
    author: Optional[str] = None
    created_utc: Optional[float] = None


FIELDS = [
    "snapshot_utc",
    "posting_id",
    "post_id",
    "subreddit",
    "author",
    "created_utc",
    "score",
    "upvote_ratio",
    "num_comments",
    "total_awards_received",
    "view_count",
    "upvotes",
    "downvotes",
]


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Track Reddit post engagement over time.")
    ap.add_argument(
        "--posts-jsonl",
        type=str,
        help="reddit_posts.jsonl produced by run_reddit_seed (posting_id/post_id metadata).",
    )
    ap.add_argument(
        "--posts-summary",
        type=str,
        help="posts_summary.csv (or posts_summary_with_dfbench*.csv) to harvest posting IDs.",
    )
    ap.add_argument(
        "--ids-file",
        type=str,
        help="Optional newline-delimited list of posting IDs (platform:postid) or plain Reddit post IDs.",
    )
    ap.add_argument(
        "--out-csv",
        type=str,
        default="propagation_timeseries.csv",
        help="Destination CSV path (appends if exists).",
    )
    ap.add_argument(
        "--dedupe",
        action="store_true",
        help="Skip posts that already have an entry for this snapshot_utc (only relevant if you pre-aggregate).",
    )
    return ap


def ensure_env() -> None:
    missing = [k for k in ("REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET") if not os.getenv(k)]
    if missing:
        raise RuntimeError(f"Missing Reddit credentials in environment: {', '.join(missing)}")


def load_from_jsonl(path: Path) -> List[TargetPost]:
    targets: Dict[str, TargetPost] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            posting_id = row.get("posting_id")
            post_id = row.get("post_id")
            if not posting_id or not post_id:
                platform = row.get("platform", "reddit")
                if not post_id:
                    continue
                posting_id = f"{platform}:{post_id}"
            targets.setdefault(
                posting_id,
                TargetPost(
                    posting_id=posting_id,
                    post_id=post_id,
                    subreddit=row.get("subreddit"),
                    author=row.get("author_name"),
                    created_utc=row.get("created_utc"),
                ),
            )
    return list(targets.values())


def load_from_summary(path: Path, existing: Dict[str, TargetPost]) -> None:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            posting_id = row.get("posting_id")
            if not posting_id:
                continue
            post_id = row.get("post_id")
            if not post_id:
                if ":" in posting_id:
                    post_id = posting_id.split(":", 1)[-1]
                else:
                    continue
            existing.setdefault(
                posting_id,
                TargetPost(
                    posting_id=posting_id,
                    post_id=post_id,
                    subreddit=row.get("subreddit"),
                    author=row.get("author_name"),
                    created_utc=float(row.get("created_utc") or 0) or None,
                ),
            )


def load_from_ids(path: Path, existing: Dict[str, TargetPost]) -> None:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            token = line.strip()
            if not token:
                continue
            if ":" in token:
                posting_id = token
                post_id = token.split(":", 1)[-1]
            else:
                post_id = token
                posting_id = f"reddit:{post_id}"
            existing.setdefault(posting_id, TargetPost(posting_id=posting_id, post_id=post_id))


def resolve_targets(args: argparse.Namespace) -> List[TargetPost]:
    targets: Dict[str, TargetPost] = {}
    if args.posts_jsonl:
        jsonl_path = Path(args.posts_jsonl).resolve()
        for item in load_from_jsonl(jsonl_path):
            targets[item.posting_id] = item
    if args.posts_summary:
        load_from_summary(Path(args.posts_summary).resolve(), targets)
    if args.ids_file:
        load_from_ids(Path(args.ids_file).resolve(), targets)
    if not targets:
        raise RuntimeError("No targets loaded. Provide at least one of --posts-jsonl/--posts-summary/--ids-file.")
    return list(targets.values())


def init_reddit() -> praw.Reddit:
    return praw.Reddit(
        client_id=os.environ["REDDIT_CLIENT_ID"],
        client_secret=os.environ["REDDIT_CLIENT_SECRET"],
        user_agent=os.getenv("REDDIT_USER_AGENT", "p2p-propagation-tracker/0.1"),
    )


def fetch_submission(reddit: praw.Reddit, target: TargetPost) -> Dict[str, object]:
    subm = reddit.submission(id=target.post_id)
    subm._fetch()
    upvotes = getattr(subm, "ups", None)
    downvotes = getattr(subm, "downs", None)
    return {
        "snapshot_utc": time.time(),
        "posting_id": target.posting_id,
        "post_id": target.post_id,
        "subreddit": getattr(subm, "subreddit", None).display_name if getattr(subm, "subreddit", None) else target.subreddit,
        "author": str(getattr(subm, "author", None)) if getattr(subm, "author", None) else target.author,
        "created_utc": float(getattr(subm, "created_utc", 0.0)) or target.created_utc,
        "score": getattr(subm, "score", None),
        "upvote_ratio": getattr(subm, "upvote_ratio", None),
        "num_comments": getattr(subm, "num_comments", None),
        "total_awards_received": getattr(subm, "total_awards_received", None),
        "view_count": getattr(subm, "view_count", None),
        "upvotes": upvotes,
        "downvotes": downvotes,
    }


def write_rows(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if not exists:
            writer.writeheader()
        for row in rows:
            formatted = {}
            for field in FIELDS:
                value = row.get(field)
                if isinstance(value, float):
                    formatted[field] = f"{value:.6f}".rstrip("0").rstrip(".")
                else:
                    formatted[field] = value if value is not None else ""
            writer.writerow(formatted)


def main() -> None:
    ap = build_argparser()
    args = ap.parse_args()
    ensure_env()
    targets = resolve_targets(args)
    reddit = init_reddit()

    snapshots: List[Dict[str, object]] = []
    for target in targets:
        try:
            snap = fetch_submission(reddit, target)
        except Exception as exc:  # pragma: no cover
            print(f"[warn] failed to fetch {target.posting_id}: {exc}")
            continue
        snapshots.append(snap)

    if not snapshots:
        print("[info] no snapshots recorded.")
        return

    out_path = Path(args.out_csv).resolve()
    write_rows(out_path, snapshots)
    print(f"[track] appended {len(snapshots)} rows -> {out_path}")


if __name__ == "__main__":
    main()
