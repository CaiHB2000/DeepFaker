# -*- coding: utf-8 -*-
"""
Fetch media from Wikimedia Commons (e.g. press/photojournalism categories) and emit
tables compatible with the DeepFakeBench ingestion pipeline.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path
from collections import deque
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse, quote

import requests

COMMONS_API = "https://commons.wikimedia.org/w/api.php"
LICENSE_ALLOWLIST = {
    "cc-by",
    "cc-by-sa",
    "cc0",
    "public domain",
    "cc-by-2.0",
    "cc-by-2.5",
    "cc-by-3.0",
    "cc-by-4.0",
    "cc-by-sa-2.0",
    "cc-by-sa-3.0",
    "cc-by-sa-4.0",
    "cc0 1.0",
    "gnu fdl",
}


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Discover Wikimedia Commons media and emit DeepFakeBench-compatible tables.")
    ap.add_argument(
        "--categories",
        type=str,
        default="Press photographs,Photojournalism",
        help="Comma-separated Wikimedia Commons categories (without 'Category:').",
    )
    ap.add_argument("--text-filter", type=str, default="", help="Optional substring filter applied to file titles (case insensitive).")
    ap.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Earliest upload date (YYYY-MM-DD). Defaults to 14 days ago.",
    )
    ap.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="Latest upload date (YYYY-MM-DD). Defaults to today.",
    )
    ap.add_argument("--limit", type=int, default=2000, help="Maximum number of media records to collect.")
    ap.add_argument("--per-category-limit", type=int, default=1500, help="Maximum records per category (before filtering).")
    ap.add_argument("--min-width", type=int, default=640, help="Minimum image width.")
    ap.add_argument("--min-height", type=int, default=480, help="Minimum image height.")
    ap.add_argument("--out_dir", type=str, default="tmp/wikimedia_seed", help="Output directory.")
    ap.add_argument("--download-media", action="store_true", help="Download media files to local directory.")
    ap.add_argument("--media-dirname", type=str, default="media", help="Subdirectory name for downloaded media.")
    ap.add_argument("--download-retries", type=int, default=3, help="Media download retries.")
    ap.add_argument("--download-timeout", type=int, default=30, help="Media download timeout (seconds).")
    ap.add_argument("--download-delay", type=float, default=0.5, help="Delay between downloads to avoid hammering servers.")
    return ap


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_date(date_str: Optional[str], default: Optional[int]) -> int:
    if date_str:
        try:
            dt = datetime.strptime(date_str.strip(), "%Y-%m-%d")
            return int(dt.timestamp())
        except ValueError:
            pass
    if default is not None:
        return default
    return int(time.time())


def sanitize_token(token: str) -> str:
    token = token.strip()
    sanitized = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in token)
    return sanitized or "media"


def infer_content_type(url: str) -> str:
    ext = Path(urlparse(url).path).suffix.lower()
    if ext in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if ext == ".png":
        return "image/png"
    if ext == ".gif":
        return "image/gif"
    if ext == ".webp":
        return "image/webp"
    return "application/octet-stream"


def license_allowed(extmetadata: Dict) -> bool:
    license_name = ""
    short = extmetadata.get("LicenseShortName", {})
    if isinstance(short, dict):
        license_name = short.get("value", "")
    elif isinstance(short, str):
        license_name = short
    license_name = license_name.lower()
    for allowed in LICENSE_ALLOWLIST:
        if allowed in license_name:
            return True
    return False


def fetch_category_media(
    root_category: str,
    start_ts: int,
    end_ts: int,
    per_category_limit: int,
    text_filter: str,
    min_width: int,
    min_height: int,
    limit_remaining: int,
    session: requests.Session,
) -> List[Dict]:
    collected: List[Dict] = []
    queue = deque([root_category])
    visited: set[str] = set()
    text_filter_lower = text_filter.lower()

    while queue and limit_remaining > 0:
        category = queue.popleft()
        if category in visited:
            continue
        visited.add(category)
        cm_continue: Optional[str] = None
        category_count = 0

        while limit_remaining > 0:
            params = {
                "action": "query",
                "format": "json",
                "generator": "categorymembers",
                "gcmtitle": f"Category:{category}",
                "gcmtype": "file|subcat",
                "gcmlimit": min(500, per_category_limit),
                "prop": "imageinfo",
                "iiprop": "url|mime|timestamp|size|extmetadata",
                "iiurlwidth": 1024,
                "iiurlheight": 1024,
                "iiextmetadatafilter": "LicenseShortName|Categories|DateTimeOriginal|ImageDescription|Artist",
            }
            if cm_continue:
                params["gcmcontinue"] = cm_continue

            try:
                resp = session.get(COMMONS_API, params=params, timeout=30)
                resp.raise_for_status()
            except requests.RequestException as exc:
                print(f"[warn] Wikimedia API request failed for {category}: {exc}")
                break

            data = resp.json()
            pages = data.get("query", {}).get("pages", {})
            if not pages:
                break

            for page in pages.values():
                title = page.get("title", "")
                ns = page.get("ns")

                if ns == 14 or title.startswith("Category:"):
                    subcat = title.replace("Category:", "", 1)
                    if subcat and subcat not in visited:
                        queue.append(subcat)
                    continue

                if not title.startswith("File:"):
                    continue
                if text_filter_lower and text_filter_lower not in title.lower():
                    continue

                imageinfo = page.get("imageinfo")
                if not imageinfo:
                    continue
                info = imageinfo[0]
                timestamp = info.get("timestamp")
                if not timestamp:
                    continue
                try:
                    upload_ts = int(datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%SZ").timestamp())
                except ValueError:
                    continue
                if upload_ts < start_ts or upload_ts > end_ts:
                    continue
                width = int(info.get("width") or 0)
                height = int(info.get("height") or 0)
                if width < min_width or height < min_height:
                    continue
                mime = info.get("mime", "")
                if not mime.startswith("image/"):
                    continue

                extmetadata = info.get("extmetadata", {}) or {}
                if not license_allowed(extmetadata):
                    continue
                url = info.get("url")
                if not url:
                    continue

                collected.append(
                    {
                        "title": title,
                        "timestamp": upload_ts,
                        "width": width,
                        "height": height,
                        "mime": mime,
                        "url": url,
                        "extmetadata": extmetadata,
                    }
                )
                limit_remaining -= 1
                category_count += 1
                if limit_remaining <= 0 or category_count >= per_category_limit:
                    break

            if limit_remaining <= 0 or category_count >= per_category_limit:
                break

            cm_continue = data.get("continue", {}).get("gcmcontinue")
            if not cm_continue:
                break
            time.sleep(0.2)

    return collected


def download_media(records: List[Dict], args: argparse.Namespace, out_dir: Path) -> Dict[str, str]:
    media_dir = out_dir / args.media_dirname
    ensure_dir(media_dir)
    local_map: Dict[str, str] = {}
    session = requests.Session()
    session.headers.update({"User-Agent": "wikimedia-research-bot/0.1"})
    session.trust_env = False
    stats = {"attempted": 0, "succeeded": 0, "failed": 0}
    for record in records:
        title = record["title"]
        url = record["url"]
        posting_id = f"wikimedia:{quote(title)}"
        post_dir = media_dir / sanitize_token(posting_id)
        ensure_dir(post_dir)
        ext = Path(urlparse(url).path).suffix or ".jpg"
        dest = post_dir / f"001{ext}"
        if dest.exists() and dest.stat().st_size > 0:
            local_map[title] = str(dest.relative_to(out_dir))
            stats["succeeded"] += 1
            stats["attempted"] += 1
            continue
        success = False
        for attempt in range(1, args.download_retries + 1):
            try:
                resp = session.get(url, timeout=args.download_timeout, stream=True)
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
            local_map[title] = str(dest.relative_to(out_dir))
            stats["succeeded"] += 1
            time.sleep(args.download_delay)
        else:
            local_map[title] = ""
            stats["failed"] += 1
    print(f"[download] attempted={stats['attempted']} succeeded={stats['succeeded']} failed={stats['failed']}")
    return local_map


def write_outputs(records: List[Dict], local_paths: Dict[str, str], out_dir: Path) -> None:
    ensure_dir(out_dir)
    posts_jsonl = out_dir / "reddit_posts.jsonl"
    media_manifest = out_dir / "media_manifest.csv"
    post_table = out_dir / "post_table.csv"
    post_map = out_dir / "post_map.csv"
    seed_urls = out_dir / "seed_urls.txt"

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

        for idx, record in enumerate(records):
            title = record["title"]
            filename = title.replace("File:", "")
            posting_id = f"wikimedia:{quote(title)}"
            permalink = f"https://commons.wikimedia.org/wiki/{quote(title)}"
            upload_ts = record["timestamp"]
            width = record["width"]
            height = record["height"]
            url = record["url"]
            local_path = local_paths.get(title, "")
            extmetadata = record["extmetadata"]
            description = ""
            desc_meta = extmetadata.get("ImageDescription", {})
            if isinstance(desc_meta, dict):
                description = desc_meta.get("value", "")
            author = extmetadata.get("Artist", {})
            author_name = ""
            if isinstance(author, dict):
                author_name = author.get("value", "")
            categories = extmetadata.get("Categories", {})
            categories_str = ""
            if isinstance(categories, dict):
                categories_str = categories.get("value", "")

            json_obj = {
                "platform": "wikimedia",
                "collection": "wikimedia_commons",
                "subreddit": "",
                "post_id": filename,
                "posting_id": posting_id,
                "url": permalink,
                "permalink": permalink,
                "created_utc": upload_ts,
                "author_id": "",
                "author_name": author_name,
                "title": filename,
                "description": description,
                "tags": categories_str,
                "is_nsfw": False,
                "is_video": False,
                "is_gallery": False,
                "score": 0,
                "upvote_ratio": None,
                "num_comments": 0,
                "domain": "commons.wikimedia.org",
                "post_hint": "image",
                "license": extmetadata.get("LicenseShortName", {}).get("value") if isinstance(extmetadata.get("LicenseShortName"), dict) else extmetadata.get("LicenseShortName"),
                "media": [
                    {
                        "media_url": url,
                        "kind": "image",
                        "content_type": infer_content_type(url),
                        "width": width,
                        "height": height,
                        "downloadable": True,
                        "source": "wikimedia_commons",
                    }
                ],
            }
            f_jsonl.write(json.dumps(json_obj, ensure_ascii=False) + "\n")

            manifest_writer.writerow(
                {
                    "media_url": url,
                    "content_type": infer_content_type(url),
                    "kind": "image",
                    "posting_id": posting_id,
                    "local_path": local_path,
                    "source": "wikimedia_commons",
                    "width": str(width),
                    "height": str(height),
                    "downloadable": "1",
                    "media_domain": urlparse(url).netloc,
                    "post_is_gallery": "0",
                    "post_is_video": "0",
                    "post_over_18": "0",
                    "post_hint": "image",
                }
            )

            post_table_writer.writerow(
                {
                    "posting_id": posting_id,
                    "platform": "wikimedia",
                    "post_id": filename,
                    "url": permalink,
                }
            )
            post_map_writer.writerow({"canonical_url": permalink, "posting_id": posting_id})
            f_seed.write(url + "\n")


def main() -> None:
    ap = build_argparser()
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    ensure_dir(out_dir)

    now = int(time.time())
    default_start = now - 14 * 86400
    start_ts = parse_date(args.start_date, default_start)
    end_ts = parse_date(args.end_date, now)
    if start_ts > end_ts:
        start_ts, end_ts = end_ts, start_ts

    categories = [cat.strip().replace(" ", "_") for cat in args.categories.split(",") if cat.strip()]
    if not categories:
        raise ValueError("At least one category must be specified.")

    print(f"[info] Fetching Wikimedia Commons media ({len(categories)} categories)...")
    session = requests.Session()
    session.headers.update({"User-Agent": "wikimedia-research-bot/0.1"})
    session.trust_env = False
    total_limit = args.limit
    records: List[Dict] = []
    for category in categories:
        if total_limit <= 0:
            break
        print(f"[info] Fetching category: {category}")
        fetched = fetch_category_media(
            root_category=category,
            start_ts=start_ts,
            end_ts=end_ts,
            per_category_limit=args.per_category_limit,
            text_filter=args.text_filter,
            min_width=args.min_width,
            min_height=args.min_height,
            limit_remaining=total_limit,
            session=session,
        )
        print(f"[info]  collected {len(fetched)} items from {category}")
        records.extend(fetched)
        total_limit -= len(fetched)

    if not records:
        print("[warn] No media collected.")
        return

    local_paths: Dict[str, str] = {}
    if args.download_media:
        local_paths = download_media(records, args, out_dir)
    else:
        for record in records:
            local_paths[record["title"]] = ""

    write_outputs(records, local_paths, out_dir)

    print(f"[ok] collected {len(records)} files → {out_dir}")
    print(f"  - post_table.csv: {out_dir / 'post_table.csv'}")
    print(f"  - media_manifest.csv: {out_dir / 'media_manifest.csv'}")
    print(f"  - reddit_posts.jsonl: {out_dir / 'reddit_posts.jsonl'}")
    print(f"  - seed_urls.txt: {out_dir / 'seed_urls.txt'}")


if __name__ == "__main__":
    main()
