# -*- coding: utf-8 -*-
"""
Generate dataset-level report JSON from aggregated Reddit tables.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Dict, List


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Summarize Reddit dataset statistics into a JSON report.")
    ap.add_argument("--posts-summary", required=True, help="Path to posts_summary.csv.")
    ap.add_argument("--content-map", required=True, help="Path to content_id_map.csv.")
    ap.add_argument("--kept-content", required=True, help="Path to kept_content.csv.")
    ap.add_argument("--dropped-content", required=True, help="Path to dropped_content.csv.")
    ap.add_argument("--quality-filters", required=True, help="Path to quality_filters.csv.")
    ap.add_argument("--content-canonical", required=True, help="Path to content_canonical.csv.")
    ap.add_argument("--out-json", required=True, help="Destination JSON report file.")
    ap.add_argument("--top-k", type=int, default=10, help="Top-K for subreddit/author rankings.")
    return ap


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def safe_int(value: str, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def safe_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def main():
    ap = build_argparser()
    args = ap.parse_args()

    posts_summary = read_csv_rows(Path(args.posts_summary))
    content_map_rows = read_csv_rows(Path(args.content_map))
    kept_rows = read_csv_rows(Path(args.kept_content))
    dropped_rows = read_csv_rows(Path(args.dropped_content))
    quality_rows = read_csv_rows(Path(args.quality_filters))
    canonical_rows = read_csv_rows(Path(args.content_canonical))

    post_count = len(posts_summary)
    media_total = len(content_map_rows)
    unique_content_total = len(kept_rows) + len(dropped_rows)

    posts_with_video = sum(1 for row in posts_summary if row.get("has_video") == "1")
    posts_with_image = sum(1 for row in posts_summary if row.get("has_image") == "1")
    posts_with_c2pa = sum(1 for row in posts_summary if row.get("c2pa_present") == "1")

    subreddit_counter = Counter(row.get("subreddit", "") for row in posts_summary if row.get("subreddit"))
    author_counter = Counter(row.get("author_name", "") for row in posts_summary if row.get("author_name"))

    score_values = [safe_int(row.get("score")) for row in posts_summary if row.get("score")]
    upvote_ratios = [safe_float(row.get("upvote_ratio")) for row in posts_summary if row.get("upvote_ratio")]

    c2pa_canonical = sum(1 for row in canonical_rows if row.get("c2pa_present") == "1")
    quality_issue_counts = Counter(row.get("issue", "") for row in quality_rows if row.get("issue"))
    fatal_issues = sum(1 for row in quality_rows if row.get("severity") == "fatal")
    warn_issues = sum(1 for row in quality_rows if row.get("severity") == "warn")

    report = {
        "posts": {
            "total": post_count,
            "with_video": posts_with_video,
            "with_image": posts_with_image,
            "with_c2pa_media": posts_with_c2pa,
            "score": {
                "mean": mean(score_values) if score_values else 0,
                "max": max(score_values) if score_values else 0,
                "min": min(score_values) if score_values else 0,
            },
            "upvote_ratio": {
                "mean": mean(upvote_ratios) if upvote_ratios else 0.0,
            },
            "top_subreddits": subreddit_counter.most_common(args.top_k),
            "top_authors": author_counter.most_common(args.top_k),
        },
        "media": {
            "total_items": media_total,
            "unique_content": unique_content_total,
            "kept_content": len(kept_rows),
            "dropped_content": len(dropped_rows),
        },
        "quality": {
            "issue_counts": quality_issue_counts,
            "fatal_issues": fatal_issues,
            "warning_issues": warn_issues,
        },
    }

    out_path = Path(args.out_json).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"[report] dataset_report -> {out_path}")


if __name__ == "__main__":
    main()
