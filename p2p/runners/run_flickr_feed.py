# -*- coding: utf-8 -*-
"""
Fetch Creative Commons licensed media from Flickr and emit tables compatible with
the existing Reddit ingestion pipeline (post_table.csv, media_manifest.csv, etc.).

This runner mirrors the structure of run_reddit_seed.py so downstream tooling
can be reused without modification.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import requests

FLICKR_REST_ENDPOINT = "https://api.flickr.com/services/rest/"
DEFAULT_EXTRAS = ",".join(
    [
        "description",
        "date_upload",
        "date_taken",
        "owner_name",
        "original_format",
        "last_update",
        "geo",
        "tags",
        "machine_tags",
        "views",
        "count_comments",
        "count_faves",
        "license",
        "media",
        "url_o",
        "url_k",
        "url_h",
        "url_l",
        "url_c",
        "url_z",
        "url_n",
        "width_o",
        "height_o",
        "width_k",
        "height_k",
        "width_h",
        "height_h",
        "width_l",
        "height_l",
        "width_c",
        "height_c",
        "width_z",
        "height_z",
        "width_n",
        "height_n",
    ]
)


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Discover Flickr media and emit DeepFakeBench-compatible tables.")
    ap.add_argument("--tags", type=str, default="news,reportage", help="Comma-separated Flickr tags to search.")
    ap.add_argument("--text", type=str, default="", help="Full-text search query (optional).")
    ap.add_argument("--licenses", type=str, default="1,2,3,4,5,6", help="Comma-separated license IDs (default: CC variants).")
    ap.add_argument("--min-upload-date", type=str, default=None, help="Minimum upload date (YYYY-MM-DD or UNIX timestamp).")
    ap.add_argument("--max-upload-date", type=str, default=None, help="Maximum upload date (YYYY-MM-DD or UNIX timestamp).")
    ap.add_argument("--media", type=str, default="photos", choices=["photos", "videos", "all"], help="Media type to search.")
    ap.add_argument("--safe-search", type=int, default=1, help="Flickr safe_search parameter (1=safe,2=moderate,3=restricted).")
    ap.add_argument("--sort", type=str, default="relevance", help="Flickr sort parameter (relevance, interestingness-desc, date-posted-desc, etc.).")
    ap.add_argument("--per-page", type=int, default=250, help="Items per page (max 500).")
    ap.add_argument("--max-pages", type=int, default=40, help="Maximum pages to fetch.")
    ap.add_argument("--limit", type=int, default=2000, help="Maximum total records to collect.")
    ap.add_argument("--min-views", type=int, default=0, help="Minimum view count filter.")
    ap.add_argument("--min-width", type=int, default=320, help="Minimum width requirement for the chosen media rendition.")
    ap.add_argument("--min-height", type=int, default=240, help="Minimum height requirement for the chosen media rendition.")
    ap.add_argument("--out_dir", type=str, default="tmp/flickr_seed", help="Output directory.")
    ap.add_argument("--download-media", action="store_true", help="Download media assets to local directory.")
    ap.add_argument("--media-dirname", type=str, default="media", help="Subdirectory under out_dir to store downloaded media.")
    ap.add_argument("--download-retries", type=int, default=3, help="Retry count per media asset when downloading.")
    ap.add_argument("--download-timeout", type=int, default=30, help="Per-request timeout (seconds) for media downloads.")
    ap.add_argument("--download-delay", type=float, default=0.5, help="Delay (seconds) between downloads to avoid rate limiting.")
    ap.add_argument("--api-key", type=str, default=None, help="Flickr API key (defaults to FLICKR_API_KEY env).")
    ap.add_argument("--api-secret", type=str, default=None, help="Flickr API secret (optional, defaults to FLICKR_API_SECRET env).")
    return ap


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_date_arg(value: Optional[str]) -> Optional[int]:
    if not value:
        return None
    value = value.strip()
    if not value:
        return None
    if value.isdigit():
        return int(value)
    try:
        struct = time.strptime(value, "%Y-%m-%d")
        return int(time.mktime(struct))
    except ValueError:
        return None


def choose_media_rendition(photo: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    candidates = [
        ("o", "original"),
        ("k", "large 2048"),
        ("h", "large 1600"),
        ("l", "large"),
        ("c", "medium 800"),
        ("z", "medium 640"),
        ("n", "small 320"),
    ]
    for suffix, _name in candidates:
        url = photo.get(f"url_{suffix}")
        width = photo.get(f"width_{suffix}")
        height = photo.get(f"height_{suffix}")
        if url and width and height:
            try:
                width = int(width)
                height = int(height)
            except ValueError:
                width, height = None, None
            return {
                "url": url,
                "width": width,
                "height": height,
                "format": suffix,
            }
    return None


def fetch_photos(args: argparse.Namespace) -> List[Dict[str, Any]]:
    api_key = args.api_key or os.getenv("FLICKR_API_KEY")
    if not api_key:
        raise RuntimeError("Missing Flickr API key (set FLICKR_API_KEY or --api-key).")

    params = {
        "method": "flickr.photos.search",
        "api_key": api_key,
        "format": "json",
        "nojsoncallback": 1,
        "extras": DEFAULT_EXTRAS,
        "license": args.licenses,
        "media": args.media,
        "sort": args.sort,
        "safe_search": args.safe_search,
        "per_page": min(max(args.per_page, 1), 500),
    }
    if args.tags:
        params["tags"] = args.tags
    if args.text:
        params["text"] = args.text

    min_upload = parse_date_arg(args.min_upload_date)
    max_upload = parse_date_arg(args.max_upload_date)
    if min_upload:
        params["min_upload_date"] = min_upload
    if max_upload:
        params["max_upload_date"] = max_upload

    records: List[Dict[str, Any]] = []
    seen_ids = set()
    page = 1
    while page <= args.max_pages and len(records) < args.limit:
        params["page"] = page
        try:
            resp = requests.get(FLICKR_REST_ENDPOINT, params=params, timeout=30)
            resp.raise_for_status()
        except requests.RequestException as exc:
            print(f"[warn] Flickr API request failed on page {page}: {exc}")
            break
        data = resp.json()
        photos = data.get("photos", {})
        photo_list = photos.get("photo", [])
        if not photo_list:
            break
        for photo in photo_list:
            pid = photo.get("id")
            if not pid or pid in seen_ids:
                continue
            rendition = choose_media_rendition(photo)
            if not rendition:
                continue
            width = rendition.get("width") or 0
            height = rendition.get("height") or 0
            if args.min_width and width and width < args.min_width:
                continue
            if args.min_height and height and height < args.min_height:
                continue
            views = int(photo.get("views") or 0)
            if views < args.min_views:
                continue
            records.append({"photo": photo, "rendition": rendition})
            seen_ids.add(pid)
            if len(records) >= args.limit:
                break
        if len(records) >= args.limit:
            break
        page += 1
        time.sleep(0.2)
    return records


def infer_content_type(url: str) -> str:
    parsed = urlparse(url)
    ext = Path(parsed.path).suffix.lower()
    if ext in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if ext == ".png":
        return "image/png"
    if ext == ".gif":
        return "image/gif"
    if ext == ".webp":
        return "image/webp"
    return "application/octet-stream"


def build_permalink(photo: Dict[str, Any]) -> str:
    owner = photo.get("owner")
    photo_id = photo.get("id")
    path_alias = photo.get("pathalias")
    if path_alias:
        return f"https://www.flickr.com/photos/{path_alias}/{photo_id}"
    return f"https://www.flickr.com/photos/{owner}/{photo_id}"


def sanitize_token(token: str) -> str:
    sanitized = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in token)
    return sanitized or "media"


def download_media(records: List[Dict[str, Any]], args: argparse.Namespace, out_dir: Path) -> Dict[Tuple[str, str], str]:
    media_dir = out_dir / args.media_dirname
    ensure_dir(media_dir)

    session = requests.Session()
    headers = {"User-Agent": os.getenv("FLICKR_USER_AGENT", "flickr-research-bot/0.1")}

    local_paths: Dict[Tuple[str, str], str] = {}
    stats = {"attempted": 0, "succeeded": 0, "skipped": 0, "failed": 0}

    for record in records:
        photo = record["photo"]
        rendition = record["rendition"]
        posting_id = f"flickr:{photo.get('id')}"
        post_dir = media_dir / sanitize_token(posting_id)
        ensure_dir(post_dir)
        url = rendition["url"]
        ext = Path(urlparse(url).path).suffix or ".jpg"
        dest = post_dir / f"001{ext}"
        key = (url, posting_id)

        if dest.exists() and dest.stat().st_size > 0:
            local_paths[key] = str(dest.relative_to(out_dir))
            stats["succeeded"] += 1
            stats["attempted"] += 1
            continue

        success = False
        for attempt in range(1, args.download_retries + 1):
            try:
                resp = session.get(url, headers=headers, timeout=args.download_timeout, stream=True)
                resp.raise_for_status()
                with open(dest, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                success = True
                break
            except requests.RequestException as exc:
                if attempt == args.download_retries:
                    print(f"[warn] download failed ({url}): {exc}")
                time.sleep(args.download_delay * attempt)
        stats["attempted"] += 1
        if success:
            local_paths[key] = str(dest.relative_to(out_dir))
            stats["succeeded"] += 1
            time.sleep(args.download_delay)
        else:
            local_paths[key] = ""
            stats["failed"] += 1

    print(
        f"[download] attempted={stats['attempted']} succeeded={stats['succeeded']} "
        f"failed={stats['failed']} skipped={stats['skipped']}"
    )
    return local_paths


def write_outputs(records: List[Dict[str, Any]], local_paths: Dict[Tuple[str, str], str], out_dir: Path) -> None:
    ensure_dir(out_dir)

    posts_jsonl = out_dir / "reddit_posts.jsonl"
    media_manifest = out_dir / "media_manifest.csv"
    post_table = out_dir / "post_table.csv"
    post_map = out_dir / "post_map.csv"
    seed_urls = out_dir / "seed_urls.txt"

    # posts_jsonl
    with posts_jsonl.open("w", encoding="utf-8") as f_jsonl, \
            media_manifest.open("w", encoding="utf-8", newline="") as f_manifest, \
            post_table.open("w", encoding="utf-8", newline="") as f_post_table, \
            post_map.open("w", encoding="utf-8", newline="") as f_post_map, \
            seed_urls.open("w", encoding="utf-8") as f_seed:

        manifest_fields = [
            "media_url",
            "content_type",
            "kind",
            "posting_id",
            "local_path",
            "source",
            "width",
            "height",
            "downloadable",
            "media_domain",
            "post_is_gallery",
            "post_is_video",
            "post_over_18",
            "post_hint",
        ]
        manifest_writer = csv.DictWriter(f_manifest, fieldnames=manifest_fields)
        manifest_writer.writeheader()

        post_table_writer = csv.DictWriter(f_post_table, fieldnames=["posting_id", "platform", "post_id", "url"])
        post_table_writer.writeheader()

        post_map_writer = csv.DictWriter(f_post_map, fieldnames=["canonical_url", "posting_id"])
        post_map_writer.writeheader()

        for record in records:
            photo = record["photo"]
            rendition = record["rendition"]
            photo_id = photo.get("id")
            posting_id = f"flickr:{photo_id}"
            permalink = build_permalink(photo)
            created_utc = int(photo.get("dateupload") or 0)
            owner = photo.get("owner")
            owner_name = photo.get("ownername")
            title = photo.get("title", "")
            description = ""
            if isinstance(photo.get("description"), dict):
                description = photo["description"].get("_content", "")
            tags = photo.get("tags", "")
            views = int(photo.get("views") or 0)
            count_comments = int(photo.get("count_comments") or 0)
            count_faves = int(photo.get("count_faves") or 0)
            media_url = rendition["url"]
            width = rendition.get("width")
            height = rendition.get("height")
            local_path = local_paths.get((media_url, posting_id), "")

            json_obj = {
                "platform": "flickr",
                "collection": "flickr_cc",
                "subreddit": "",  # kept for schema compatibility
                "post_id": photo_id,
                "posting_id": posting_id,
                "url": permalink,
                "permalink": permalink,
                "created_utc": created_utc,
                "author_id": owner,
                "author_name": owner_name,
                "title": title,
                "description": description,
                "tags": tags,
                "is_nsfw": False,
                "is_video": photo.get("media") == "video",
                "is_gallery": False,
                "score": views,
                "upvote_ratio": None,
                "num_comments": count_comments,
                "favorites": count_faves,
                "domain": "flickr.com",
                "post_hint": "image",
                "license": photo.get("license"),
                "media": [
                    {
                        "media_url": media_url,
                        "kind": "image",
                        "content_type": infer_content_type(media_url),
                        "width": width,
                        "height": height,
                        "downloadable": True,
                        "source": "flickr",
                    }
                ],
            }
            f_jsonl.write(json.dumps(json_obj, ensure_ascii=False) + "\n")

            manifest_writer.writerow(
                {
                    "media_url": media_url,
                    "content_type": infer_content_type(media_url),
                    "kind": "image",
                    "posting_id": posting_id,
                    "local_path": local_path,
                    "source": "flickr",
                    "width": "" if width is None else str(width),
                    "height": "" if height is None else str(height),
                    "downloadable": "1",
                    "media_domain": urlparse(media_url).netloc,
                    "post_is_gallery": "0",
                    "post_is_video": "1" if photo.get("media") == "video" else "0",
                    "post_over_18": "0",
                    "post_hint": "image",
                }
            )

            post_table_writer.writerow(
                {
                    "posting_id": posting_id,
                    "platform": "flickr",
                    "post_id": photo_id,
                    "url": permalink,
                }
            )
            post_map_writer.writerow({"canonical_url": permalink, "posting_id": posting_id})
            f_seed.write(media_url + "\n")


def main() -> None:
    ap = build_argparser()
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    ensure_dir(out_dir)

    print("[info] Fetching Flickr media...")
    records = fetch_photos(args)
    if not records:
        print("[warn] No records collected.")
        return

    local_paths: Dict[Tuple[str, str], str] = {}
    if args.download_media:
        local_paths = download_media(records, args, out_dir)
    else:
        for record in records:
            photo = record["photo"]
            rendition = record["rendition"]
            posting_id = f"flickr:{photo.get('id')}"
            local_paths[(rendition["url"], posting_id)] = ""

    write_outputs(records, local_paths, out_dir)

    print(f"[ok] collected {len(records)} photos → {out_dir}")
    print(f"  - post_table.csv: {out_dir / 'post_table.csv'}")
    print(f"  - media_manifest.csv: {out_dir / 'media_manifest.csv'}")
    print(f"  - reddit_posts.jsonl: {out_dir / 'reddit_posts.jsonl'}")
    print(f"  - seed_urls.txt: {out_dir / 'seed_urls.txt'}")


if __name__ == "__main__":
    main()
