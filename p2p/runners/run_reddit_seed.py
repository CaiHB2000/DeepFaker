# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, os, json, csv, time
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from urllib.parse import urlparse
from collections import defaultdict

import praw
import requests

from p2p.datasource.connectors.reddit_api import (
    iter_posts, PostRecord, MediaItem
)

def build_argparser():
    ap = argparse.ArgumentParser(description="Discover Reddit seeds via official API and emit framework tables.")
    ap.add_argument("--subs", type=str, default="pics,news,worldnews", help="Comma-separated subreddits")
    ap.add_argument("--days", type=int, default=7, help="Only include posts within last N days")
    ap.add_argument("--limit", type=int, default=60, help="Total posts target across all subs")
    ap.add_argument("--per-sub-limit", type=int, default=50, help="Max posts per subreddit before moving on")
    ap.add_argument("--query", type=str, default="", help="Optional search query")
    ap.add_argument("--min-score", type=int, default=0, help="Minimum post score")
    ap.add_argument("--out_dir", type=str, default="results/sample", help="Output directory")
    ap.add_argument("--emit-jsonl", action="store_true", help="Also write reddit_posts.jsonl rich metadata")
    ap.add_argument("--download-media", action="store_true", help="Download media assets and populate media_manifest.local_path")
    ap.add_argument("--media-dirname", type=str, default="media", help="Subdirectory under out_dir to store downloaded media")
    ap.add_argument("--download-retries", type=int, default=3, help="Retry count per media asset when downloading")
    ap.add_argument("--download-timeout", type=int, default=30, help="Per-request timeout (seconds) for media downloads")
    ap.add_argument("--download-delay", type=float, default=1.0, help="Delay (seconds) between successful media downloads to avoid CDN rate limits")
    ap.add_argument("--allow-post-hints", type=str, default="image,link,rich:video,hosted:video,gallery", help="Comma-separated whitelist of post_hint values (empty = allow all)")
    ap.add_argument("--allow-media-kinds", type=str, default="image,video_mp4,gif", help="Comma-separated whitelist of media kinds")
    ap.add_argument("--min-media-width", type=int, default=320, help="Minimum width for at least one media item when require-media is set")
    ap.add_argument("--min-media-height", type=int, default=240, help="Minimum height for at least one media item when require-media is set")
    ap.add_argument("--require-media", action="store_true", help="Keep only posts that have downloadable media")
    ap.add_argument("--exclude-flair", type=str, default="meme,fanart,art,ai art,meta,discussion", help="Comma-separated list of link flair texts to drop (case-insensitive substring match)")
    ap.add_argument("--exclude-title-keywords", type=str, default="meme,fanart,ai art,artstation,commission,drawing,cartoon,anime,comic,illustration", help="Comma-separated keywords; if title contains any, drop the post")
    ap.add_argument("--exclude-domains", type=str, default="deviantart.com,artstation.com,instagram.com,behance.net,redbubble.com,etsy.com", help="Comma-separated domain blacklist")
    ap.add_argument("--include-domains", type=str, default="", help="Comma-separated domain whitelist (empty = allow all)")
    ap.add_argument("--allow-nsfw", action="store_true", help="Allow NSFW posts (default: filter out)")
    return ap

def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)


def _parse_csv(value: str) -> List[str]:
    if not value:
        return []
    return [token.strip().lower() for token in value.split(",") if token.strip()]


def _domain_from_record(rec: PostRecord) -> str:
    domain = rec.domain or rec.url or rec.permalink or ""
    if domain.startswith("http"):
        try:
            return urlparse(domain).netloc.lower()
        except Exception:
            return ""
    return (rec.domain or "").lower()


def _should_keep(
    rec: PostRecord,
    allow_post_hints: List[str],
    allow_media_kinds: List[str],
    min_media_width: int,
    min_media_height: int,
    require_media: bool,
    exclude_flair_tokens: List[str],
    exclude_title_tokens: List[str],
    exclude_domains: List[str],
    include_domains: List[str],
    allow_nsfw: bool,
) -> Tuple[bool, str]:
    if not allow_nsfw and (rec.is_nsfw or rec.over_18):
        return False, "nsfw"

    post_hint = (rec.post_hint or "").lower()
    if allow_post_hints and post_hint not in allow_post_hints:
        return False, "post_hint"

    domain = _domain_from_record(rec)
    if include_domains and domain not in include_domains:
        return False, "domain_not_whitelisted"
    if exclude_domains and domain in exclude_domains:
        return False, "domain_blacklisted"

    flair = (rec.link_flair_text or "").lower()
    if exclude_flair_tokens and flair:
        for token in exclude_flair_tokens:
            if token and token in flair:
                return False, "flair_excluded"

    title = (rec.title or "").lower()
    if exclude_title_tokens and title:
        for token in exclude_title_tokens:
            if token and token in title:
                return False, "title_keyword"

    if require_media:
        medias = [m for m in rec.media if m.media_url]
        if not medias:
            return False, "no_media"

        if allow_media_kinds:
            medias = [m for m in medias if (m.kind or "").lower() in allow_media_kinds]
            if not medias:
                return False, "media_kind"

        size_ok = False
        for m in medias:
            width = int(m.width or 0)
            height = int(m.height or 0)
            if width == 0 and height == 0:
                size_ok = True
                break
            if (min_media_width <= 0 or width >= min_media_width) and (min_media_height <= 0 or height >= min_media_height):
                size_ok = True
                break
        if not size_ok:
            return False, "media_size"

    return True, ""


def write_jsonl(path: str, records: List[PostRecord]):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            obj = {
                "platform": r.platform,
                "subreddit": r.subreddit,
                "post_id": r.post_id,
                "posting_id": r.posting_id(),
                "url": r.url,
                "permalink": r.permalink,
                "created_utc": r.created_utc,
                "author_id": r.author_id,
                "author_name": r.author_name,
                "title": r.title,
                "is_nsfw": r.is_nsfw,
                "is_video": r.is_video,
                "is_gallery": r.is_gallery,
                "stickied": r.stickied,
                "crosspost_parent": r.crosspost_parent,
                "score": r.score,
                "ups": r.ups,
                "upvote_ratio": r.upvote_ratio,
                "num_comments": r.num_comments,
                "gilded": r.gilded,
                "domain": r.domain,
                "url_overridden_by_dest": r.url_overridden_by_dest,
                "post_hint": r.post_hint,
                "selftext": r.selftext,
                "selftext_html": r.selftext_html,
                "link_flair_text": r.link_flair_text,
                "link_flair_richtext": r.link_flair_richtext,
                "flair_template_id": r.flair_template_id,
                "author_flair_text": r.author_flair_text,
                "author_flair_css_class": r.author_flair_css_class,
                "author_flair_richtext": r.author_flair_richtext,
                "author_is_blocked": r.author_is_blocked,
                "author_premium": r.author_premium,
                "spoiler": r.spoiler,
                "locked": r.locked,
                "archived": r.archived,
                "pinned": r.pinned,
                "distinguished": r.distinguished,
                "removed_by_category": r.removed_by_category,
                "num_crossposts": r.num_crossposts,
                "is_original_content": r.is_original_content,
                "is_self": r.is_self,
                "view_count": r.view_count,
                "total_awards_received": r.total_awards_received,
                "whitelist_status": r.whitelist_status,
                "media_only": r.media_only,
                "edited_ts": r.edited_ts,
                "subreddit_subscribers": r.subreddit_subscribers,
                "thumbnail_url": r.thumbnail_url,
                "thumbnail_width": r.thumbnail_width,
                "thumbnail_height": r.thumbnail_height,
                "media": [m.__dict__ for m in r.media],
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def build_tables(records: List[PostRecord], out_dir: str, media_local_paths: Optional[Dict[Tuple[str, str], str]] = None):
    # 1) post_table.csv
    pt_path = str(Path(out_dir) / "post_table.csv")
    with open(pt_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["posting_id", "platform", "post_id", "url"])
        w.writeheader()
        seen = set()
        for r in records:
            pid = r.posting_id()
            if pid in seen: continue
            w.writerow({"posting_id": pid, "platform": r.platform, "post_id": r.post_id, "url": r.permalink or r.url})
            seen.add(pid)

    # 2) media_manifest.csv（多媒体全面落表）
    mm_path = str(Path(out_dir) / "media_manifest.csv")
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
    with open(mm_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=manifest_fields)
        w.writeheader()
        seen_media = set()
        for r in records:
            pid = r.posting_id()
            for m in r.media:
                key = (m.media_url, pid)
                if key in seen_media: continue
                media_domain = ""
                if m.media_url:
                    try:
                        media_domain = urlparse(m.media_url).netloc
                    except Exception:
                        media_domain = ""
                w.writerow({
                    "media_url": m.media_url,
                    "content_type": m.content_type or "",
                    "kind": m.kind,
                    "posting_id": pid,
                    "local_path": (media_local_paths or {}).get(key, ""),
                    "source": m.source or "",
                    "width": "" if m.width in (None, "") else str(m.width),
                    "height": "" if m.height in (None, "") else str(m.height),
                    "downloadable": "1" if m.downloadable else "0",
                    "media_domain": media_domain,
                    "post_is_gallery": "1" if r.is_gallery else "0",
                    "post_is_video": "1" if r.is_video else "0",
                    "post_over_18": "1" if getattr(r, "over_18", False) else "0",
                    "post_hint": r.post_hint or "",
                })
                seen_media.add(key)

    # 3) post_map.csv（URL → posting_id 映射，优先 permalink）
    pm_path = str(Path(out_dir) / "post_map.csv")
    with open(pm_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["canonical_url", "posting_id"])
        w.writeheader()
        seen_map = set()
        for r in records:
            for u in filter(None, [r.permalink, r.url]):
                key = (u, r.posting_id())
                if key in seen_map: continue
                w.writerow({"canonical_url": u, "posting_id": r.posting_id()})
                seen_map.add(key)

    return {"post_table": pt_path, "media_manifest": mm_path, "post_map": pm_path}


def sanitize_token(token: str) -> str:
    token = token.strip()
    sanitized = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in token)
    return sanitized or "media"


def infer_extension(media: MediaItem) -> str:
    path_ext = Path(urlparse(media.media_url or "").path).suffix.lower()
    if path_ext:
        return path_ext
    ct = (media.content_type or "").lower()
    if "jpeg" in ct or ct.endswith("/jpg"):
        return ".jpg"
    if ct.endswith("/png"):
        return ".png"
    if ct.endswith("/gif"):
        return ".gif"
    if ct.endswith("/mp4"):
        return ".mp4"
    if ct.endswith("/webp"):
        return ".webp"
    return ".bin"


def download_media(records: List[PostRecord], out_dir: str, media_dirname: str, retries: int, timeout: int, delay: float) -> Tuple[Dict[Tuple[str, str], str], Dict[str, Any]]:
    media_dir = Path(out_dir) / media_dirname
    media_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    headers = {
        "User-Agent": os.getenv("REDDIT_MEDIA_USER_AGENT", os.getenv("REDDIT_USER_AGENT", "deepfake-research-bot/0.1"))
    }

    local_paths: Dict[Tuple[str, str], str] = {}
    stats = {"attempted": 0, "succeeded": 0, "skipped": 0, "failed": []}

    for record in records:
        post_dir = media_dir / sanitize_token(record.posting_id())
        post_dir.mkdir(parents=True, exist_ok=True)
        for idx, media in enumerate(record.media, start=1):
            key = (media.media_url, record.posting_id())
            if not media.media_url or not media.media_url.startswith("http") or not media.downloadable:
                stats["skipped"] += 1
                local_paths[key] = ""
                continue

            ext = infer_extension(media)
            filename = f"{idx:03d}{ext}"
            dest_path = post_dir / filename

            if dest_path.exists() and dest_path.stat().st_size > 0:
                rel_path = dest_path.relative_to(out_dir)
                local_paths[key] = str(rel_path)
                stats["succeeded"] += 1
                stats["attempted"] += 1
                continue

            success = False
            for attempt in range(1, retries + 1):
                try:
                    resp = session.get(media.media_url, headers=headers, timeout=timeout, stream=True)
                    resp.raise_for_status()
                    with open(dest_path, "wb") as f:
                        for chunk in resp.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                    success = True
                    break
                except requests.RequestException as e:
                    if attempt == retries:
                        stats["failed"].append({"url": media.media_url, "posting_id": record.posting_id(), "error": str(e)})
                    else:
                        sleep_for = delay * attempt
                        print(f"[download] retry {attempt}/{retries} for {media.media_url} in {sleep_for:.1f}s ({e})")
                        time.sleep(sleep_for)
                except Exception as e:
                    stats["failed"].append({"url": media.media_url, "posting_id": record.posting_id(), "error": str(e)})
                    break
            stats["attempted"] += 1
            if success:
                rel_path = dest_path.relative_to(out_dir)
                local_paths[key] = str(rel_path)
                stats["succeeded"] += 1
                time.sleep(delay)
            else:
                local_paths[key] = ""
    return local_paths, stats

def main():
    ap = build_argparser(); args = ap.parse_args()
    ensure_dir(args.out_dir)

    cid = os.getenv("REDDIT_CLIENT_ID"); csec = os.getenv("REDDIT_CLIENT_SECRET")
    if not cid or not csec:
        print("[error] Missing REDDIT_CLIENT_ID/REDDIT_CLIENT_SECRET"); return

    reddit = praw.Reddit(
        client_id=cid,
        client_secret=csec,
        user_agent=os.getenv("REDDIT_USER_AGENT", "deepfake-research-bot/0.1"),
    )

    subs = [s.strip() for s in args.subs.split(",") if s.strip()]
    # 粗略分配每个 sub 的上限，避免一个大 sub 抢光额度
    per_sub_limit = max(1, args.per_sub_limit)
    global_cap = max(1, args.limit)

    allow_post_hints = set(_parse_csv(args.allow_post_hints))
    allow_media_kinds = set(_parse_csv(args.allow_media_kinds))
    exclude_flair_tokens = _parse_csv(args.exclude_flair)
    exclude_title_tokens = _parse_csv(args.exclude_title_keywords)
    exclude_domains = set(_parse_csv(args.exclude_domains))
    include_domains = set(_parse_csv(args.include_domains))

    records: List[PostRecord] = []
    seen = set()
    filter_stats = defaultdict(int)
    scanned = 0
    for rec in iter_posts(reddit, subs, args.query, args.days, args.min_score, per_sub_limit):
        scanned += 1
        if rec.post_id in seen:
            continue
        keep, reason = _should_keep(
            rec,
            allow_post_hints,
            allow_media_kinds,
            args.min_media_width,
            args.min_media_height,
            args.require_media,
            exclude_flair_tokens,
            exclude_title_tokens,
            exclude_domains,
            include_domains,
            args.allow_nsfw,
        )
        if not keep:
            filter_stats[reason or "filtered"] += 1
            continue
        records.append(rec)
        seen.add(rec.post_id)
        if len(records) >= global_cap:
            break

    media_local_paths: Dict[Tuple[str, str], str] = {}
    download_stats: Optional[Dict[str, Any]] = None
    if args.download_media:
        media_local_paths, download_stats = download_media(
            records,
            args.out_dir,
            args.media_dirname,
            args.download_retries,
            args.download_timeout,
            args.download_delay,
        )

    # 生成三张表 + 可选 JSONL
    paths = build_tables(records, args.out_dir, media_local_paths)
    if args.emit_jsonl:
        write_jsonl(str(Path(args.out_dir) / "reddit_posts.jsonl"), records)

    # 额外：为下游下载器生成简易 seeds（只直链可下载项）
    seed_path = str(Path(args.out_dir) / "seed_urls.txt")
    with open(seed_path, "w", encoding="utf-8") as f:
        for r in records:
            for m in r.media:
                if m.kind in ("image", "gif", "video_mp4") and (m.media_url or "").startswith("http"):
                    f.write(m.media_url + "\n")

    print(f"[ok] collected {len(records)} posts → {args.out_dir}")
    print("  - post_table.csv:", paths["post_table"])
    print("  - media_manifest.csv:", paths["media_manifest"])
    print("  - post_map.csv:", paths["post_map"])
    dropped = scanned - len(records)
    print(f"[filter] scanned={scanned} kept={len(records)} dropped={dropped}")
    if dropped:
        for reason, count in sorted(filter_stats.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"    • {reason}: {count}")
    if args.emit_jsonl:
        print("  - reddit_posts.jsonl written")
    print("  - seed_urls.txt (direct-download media) written")
    if download_stats:
        print(f"  - media downloads: {download_stats['succeeded']}/{download_stats['attempted']} succeeded, {download_stats['skipped']} skipped, {len(download_stats['failed'])} failed")
        if download_stats["failed"]:
            failed_log = Path(args.out_dir) / "media_download_failures.json"
            with open(failed_log, "w", encoding="utf-8") as f:
                json.dump(download_stats["failed"], f, ensure_ascii=False, indent=2)
            print(f"    • failed list saved -> {failed_log}")

if __name__ == "__main__":
    main()
