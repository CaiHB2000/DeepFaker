#!/usr/bin/env python3
"""Search Reddit for each event keyword set and store JSONL posts."""
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import praw


def load_keywords(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def ensure_reddit():
    cid = os.getenv("REDDIT_CLIENT_ID")
    csec = os.getenv("REDDIT_CLIENT_SECRET")
    ua = os.getenv("REDDIT_USER_AGENT", "dpmd-reddit-scraper/0.1")
    if not cid or not csec:
        raise SystemExit("Missing REDDIT_CLIENT_ID/REDDIT_CLIENT_SECRET env vars")
    return praw.Reddit(client_id=cid, client_secret=csec, user_agent=ua)


def serialize_submission(subm):
    created = datetime.fromtimestamp(getattr(subm, "created_utc", 0) or 0, tz=timezone.utc).isoformat()
    media = []
    try:
        if getattr(subm, "is_gallery", False):
            gallery = getattr(subm, "gallery_data", {}) or {}
            meta = getattr(subm, "media_metadata", {}) or {}
            for item in gallery.get("items", []):
                mid = item.get("media_id")
                if mid and meta.get(mid):
                    s = meta[mid].get("s") or {}
                    media.append({
                        "url": (s.get("u") or "").replace("&amp;", "&"),
                        "width": s.get("x"),
                        "height": s.get("y"),
                        "kind": "image",
                    })
    except Exception:
        pass
    if getattr(subm, "preview", None):
        images = (subm.preview or {}).get("images", [])
        for img in images:
            src = img.get("source", {})
            if src:
                media.append({
                    "url": src.get("url", "").replace("&amp;", "&"),
                    "width": src.get("width"),
                    "height": src.get("height"),
                    "kind": "preview",
                })
    return {
        "platform": "reddit",
        "event_id": None,  # fill later
        "keyword": None,
        "id": subm.id,
        "full_name": subm.fullname,
        "permalink": f"https://www.reddit.com{subm.permalink}" if subm.permalink else None,
        "url": subm.url,
        "title": subm.title,
        "selftext": subm.selftext,
        "subreddit": str(subm.subreddit) if subm.subreddit else None,
        "author": getattr(subm.author, "name", None) if subm.author else None,
        "created_utc": created,
        "score": subm.score,
        "upvote_ratio": subm.upvote_ratio,
        "num_comments": subm.num_comments,
        "is_video": subm.is_video,
        "is_self": subm.is_self,
        "over_18": subm.over_18,
        "link_flair_text": subm.link_flair_text,
        "media": media,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keywords", default="datasets/dpmd_modalities/events/keywords.json")
    parser.add_argument("--out-dir", default="datasets/dpmd_modalities/raw_posts/reddit")
    parser.add_argument("--per-event-limit", type=int, default=40)
    parser.add_argument("--per-keyword-limit", type=int, default=10)
    parser.add_argument("--time-filter", default="week", choices=["hour","day","week","month","year","all"])
    parser.add_argument("--sleep", type=float, default=1.0, help="Delay between keyword queries")
    args = parser.parse_args()

    keywords_map = load_keywords(Path(args.keywords))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    reddit = ensure_reddit()

    for event_id, meta in keywords_map.items():
        words = meta.get("keywords", []) or []
        if not words:
            continue
        out_path = out_dir / f"{event_id}.jsonl"
        existing_ids = set()
        if out_path.exists():
            with out_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    existing_ids.add(obj.get("id"))
        collected = []
        for kw in words:
            if len(collected) >= args.per_event_limit:
                break
            try:
                submissions = reddit.subreddit("all").search(
                    kw,
                    sort="new",
                    time_filter=args.time_filter,
                    limit=args.per_keyword_limit)
            except Exception as exc:
                print(f"[warn] search failed for '{kw}': {exc}", file=sys.stderr)
                continue
            for subm in submissions:
                if len(collected) >= args.per_event_limit:
                    break
                if subm.id in existing_ids:
                    continue
                record = serialize_submission(subm)
                record["event_id"] = event_id
                record["keyword"] = kw
                collected.append(record)
                existing_ids.add(subm.id)
            time.sleep(args.sleep)
        if not collected:
            continue
        with out_path.open("a", encoding="utf-8") as fh:
            for rec in collected:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"Event {event_id}: collected {len(collected)} posts")


if __name__ == "__main__":
    main()
