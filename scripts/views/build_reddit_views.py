#!/usr/bin/env python3
"""Convert reddit raw JSONL posts into modality views (full + text-only)."""
import argparse
import hashlib
import json
import mimetypes
import pathlib
import re
import sys
from datetime import datetime, timezone
from typing import List, Tuple

import requests

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "dpmd-dataset-reddit-viewer/0.1"})


def iter_records(path: pathlib.Path):
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def sanitize_filename(url: str, fallback: str) -> str:
    parsed = url.split("?")[0].split("/")[-1]
    parsed = parsed[:80]
    if not parsed or "." not in parsed:
        parsed = fallback
    safe = re.sub(r"[^A-Za-z0-9._-]", "-", parsed)
    return safe or fallback


def download_media(url: str, dest: pathlib.Path) -> Tuple[bool, str]:
    try:
        resp = SESSION.get(url, timeout=30)
        resp.raise_for_status()
    except Exception as exc:
        return False, f"download_failed:{exc}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as fh:
        fh.write(resp.content)
    return True, "ok"


def build_views_for_event(raw_file: pathlib.Path, out_dir: pathlib.Path, media_limit: int):
    records = list(iter_records(raw_file))
    if not records:
        return 0
    count = 0
    for rec in records:
        post_id = rec.get("id") or f"post-{count}"
        event_id = rec.get("event_id")
        base_dir = out_dir / event_id / post_id
        base_dir.mkdir(parents=True, exist_ok=True)
        media_entries: List[dict] = []
        medias = rec.get("media") or []
        for idx, media in enumerate(medias[:media_limit]):
            url = media.get("url")
            if not url:
                continue
            filename = sanitize_filename(url, f"media-{idx}.bin")
            local_path = base_dir / "media" / filename
            ok, reason = download_media(url, local_path)
            if ok:
                media_entries.append(
                    {
                        "original_url": url,
                        "local_path": str(local_path.relative_to(out_dir)),
                        "width": media.get("width"),
                        "height": media.get("height"),
                        "kind": media.get("kind"),
                    }
                )
            else:
                media_entries.append(
                    {
                        "original_url": url,
                        "local_path": None,
                        "error": reason,
                    }
                )
        text_bundle = {
            "title": rec.get("title"),
            "selftext": rec.get("selftext"),
            "subreddit": rec.get("subreddit"),
            "permalink": rec.get("permalink"),
        }
        timestamp = datetime.now(timezone.utc).isoformat()
        full_view = {
            "event_id": event_id,
            "post_id": post_id,
            "view_id": f"{post_id}-full",
            "variant": "full_bundle",
            "modality": {
                "text": True,
                "image": bool(media_entries),
            },
            "text_bundle": text_bundle,
            "media": media_entries,
            "generated_at": timestamp,
        }
        text_only_view = {
            "event_id": event_id,
            "post_id": post_id,
            "view_id": f"{post_id}-textonly",
            "variant": "observed_view",
            "modality": {
                "text": True,
                "image": False,
            },
            "text_bundle": text_bundle,
            "missing_modalities": ["image"],
            "missing_reason": "simulated_drop_image",
            "generated_at": timestamp,
        }
        (base_dir / "full_bundle.json").write_text(json.dumps(full_view, ensure_ascii=False, indent=2), encoding="utf-8")
        (base_dir / "observed_textonly.json").write_text(json.dumps(text_only_view, ensure_ascii=False, indent=2), encoding="utf-8")
        count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="datasets/dpmd_modalities/raw_posts/reddit")
    parser.add_argument("--views-dir", default="datasets/dpmd_modalities/views")
    parser.add_argument("--media-limit", type=int, default=3)
    args = parser.parse_args()

    raw_dir = pathlib.Path(args.raw_dir)
    views_dir = pathlib.Path(args.views_dir)
    views_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for raw_file in raw_dir.glob("*.jsonl"):
        created = build_views_for_event(raw_file, views_dir, args.media_limit)
        if created:
            print(f"{raw_file.name}: built {created} posts")
            total += created
    print(f"Total reddit posts processed: {total}")


if __name__ == "__main__":
    main()
