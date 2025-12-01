# -*- coding: utf-8 -*-
"""
Continuous Reddit poller that:
1. Streams newly published posts from configured subreddits/search queries.
2. Updates engagement metrics for all tracked posts.
3. Downloads media for newly tracked posts and extracts face crops.

Outputs (stored under ``--out-dir``):
    - ``posts_status.csv``: latest engagement snapshot for every tracked post.
    - ``face_detections.csv``: per-media face extraction summary.
    - ``media/`` + ``face_crops/`` directories with downloaded assets.
    - ``state/watch_reddit_faces.json``: internal state used to avoid reprocessing.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import signal
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse

import praw

from p2p.datasource.connectors.reddit_api import PostRecord, iter_posts
from p2p.runners.run_reddit_seed import download_media, sanitize_token
from p2p.tools import track_propagation as tp
from p2p.tools.filter_faces import crop_face, detect_faces, load_image


@dataclass
class FaceDetection:
    media_url: str
    media_local_path: str
    face_path: Optional[str]
    num_faces: int
    issues: List[str]


@dataclass
class PostState:
    posting_id: str
    post_id: str
    subreddit: Optional[str]
    author_name: Optional[str]
    title: Optional[str]
    created_utc: Optional[float]
    first_seen_utc: float
    last_snapshot_utc: Optional[float]
    metrics: Dict[str, Optional[float]]
    media_items: List[Dict[str, str]]
    faces: List[FaceDetection]
    next_refresh_utc: Optional[float] = None


@dataclass
class AllPostInfo:
    posting_id: str
    post_id: str
    subreddit: Optional[str]
    author_name: Optional[str]
    title: Optional[str]
    created_utc: Optional[float]
    first_seen_utc: float
    last_snapshot_utc: Optional[float]
    metrics: Dict[str, Optional[float]]
    face_crops: int
    has_faces: bool
    media_count: int
    next_refresh_utc: Optional[float] = None


DEFAULT_STATUS_FIELDS = [
    "posting_id",
    "post_id",
    "subreddit",
    "author_name",
    "title",
    "created_utc",
    "first_seen_utc",
    "last_snapshot_utc",
    "score",
    "upvote_ratio",
    "num_comments",
    "total_awards_received",
    "view_count",
    "face_crops",
    "media_count",
]

DEFAULT_STATUS_FIELDS_ALL = DEFAULT_STATUS_FIELDS + ["has_faces"]

FACE_INDEX_FIELDS = [
    "posting_id",
    "media_url",
    "media_local_path",
    "face_path",
    "num_faces",
    "issues",
]


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Continuously harvest Reddit posts and face crops.")
    ap.add_argument("--subs", type=str, default="pics,news,worldnews", help="Comma-separated subreddit list.")
    ap.add_argument("--query", type=str, default="", help="Optional search query (per subreddit).")
    ap.add_argument("--days", type=int, default=7, help="Only consider posts within the last N days (per poll).")
    ap.add_argument("--min-score", type=int, default=0, help="Drop posts with score below this threshold.")
    ap.add_argument("--per-sub-limit", type=int, default=30, help="Max new posts to inspect per subreddit on each iteration.")
    ap.add_argument("--max-new-posts", type=int, default=30, help="Global cap on new posts recorded per iteration.")
    ap.add_argument("--require-media", action="store_true", help="Keep only posts that expose downloadable media.")
    ap.add_argument("--allow-nsfw", action="store_true", help="Allow NSFW/over_18 posts (default filters them out).")
    ap.add_argument(
        "--allow-post-hints",
        type=str,
        default="image,rich:video,hosted:video,gallery,link",
        help="Comma-separated whitelist of post_hint values; empty string to allow all.",
    )
    ap.add_argument(
        "--max-refresh-batch",
        type=int,
        default=300,
        help="Maximum number of tracked posts to refresh per iteration (0 = no limit).",
    )
    ap.add_argument(
        "--allow-media-kinds",
        type=str,
        default="image,video_mp4,gif",
        help="Comma-separated whitelist of media kinds that must be present when --require-media is set.",
    )
    ap.add_argument("--min-media-width", type=int, default=320, help="Minimum width for at least one media item.")
    ap.add_argument("--min-media-height", type=int, default=240, help="Minimum height for at least one media item.")
    ap.add_argument("--exclude-flair", type=str, default="meme,fanart,art", help="Comma-separated flair substrings to skip.")
    ap.add_argument(
        "--exclude-title-keywords",
        type=str,
        default="meme,fanart,ai art,artstation,commission,drawing,anime",
        help="Comma-separated title substrings to skip.",
    )
    ap.add_argument("--exclude-domains", type=str, default="deviantart.com,artstation.com", help="Domain blacklist.")
    ap.add_argument("--include-domains", type=str, default="", help="Domain whitelist (empty = allow all).")

    ap.add_argument("--out-dir", type=str, required=True, help="Workspace directory for outputs/state.")
    ap.add_argument("--media-dirname", type=str, default="media", help="Relative directory to store downloaded media.")
    ap.add_argument("--face-dirname", type=str, default="face_crops", help="Relative directory for face crops.")
    ap.add_argument("--image-size", type=int, default=256, help="Face crop size (square).")
    ap.add_argument("--download-timeout", type=int, default=30, help="Media download timeout (seconds).")
    ap.add_argument("--download-retries", type=int, default=3, help="Retry count per media asset.")
    ap.add_argument("--download-delay", type=float, default=1.0, help="Sleep between successful downloads (seconds).")

    ap.add_argument(
        "--bootstrap-existing",
        action="store_true",
        help="Load previously harvested posts under out_dir (posts_summary.csv, media_inventory.csv, face_detection_summary.csv) into the watcher state.",
    )
    ap.add_argument(
        "--bootstrap-posts-summary",
        type=str,
        default=None,
        help="Optional explicit path to posts_summary.csv when bootstrapping existing data.",
    )
    ap.add_argument(
        "--bootstrap-media-inventory",
        type=str,
        default=None,
        help="Optional explicit path to media_inventory.csv when bootstrapping existing data.",
    )
    ap.add_argument(
        "--bootstrap-content-map",
        type=str,
        default=None,
        help="Optional explicit path to content_id_map.csv when bootstrapping existing data.",
    )
    ap.add_argument(
        "--bootstrap-face-summary",
        type=str,
        default=None,
        help="Optional explicit path to face_detection_summary.csv when bootstrapping existing data.",
    )
    ap.add_argument(
        "--bootstrap-posts-jsonl",
        type=str,
        default=None,
        help="Optional explicit path to reddit_posts.jsonl (provides titles/selftext for bootstrapped posts).",
    )

    ap.add_argument(
        "--max-post-age-hours",
        type=float,
        default=6.0,
        help="Skip newly discovered posts whose creation time距离当前超过该阈值（小时）。设置 <=0 表示不过滤。",
    )

    ap.add_argument("--interval", type=float, default=600.0, help="Seconds between polling rounds.")
    ap.add_argument("--jitter", type=float, default=60.0, help="Uniform jitter (+/- seconds) per sleep.")
    ap.add_argument("--per-request-sleep", type=float, default=1.0, help="Delay between engagement refresh requests.")
    ap.add_argument("--max-iterations", type=int, default=0, help="Optional limit on polling rounds (0 = infinite).")
    ap.add_argument("--status-json", type=str, default=None, help="Optional status file updated after each iteration.")
    ap.add_argument("--no-progress", action="store_true", help="Disable progress output while refreshing engagement.")
    return ap


def ensure_env() -> None:
    missing = [k for k in ("REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET") if not os.getenv(k)]
    if missing:
        raise RuntimeError(f"Missing Reddit API credentials: {', '.join(missing)}")


def init_reddit() -> praw.Reddit:
    return praw.Reddit(
        client_id=os.environ["REDDIT_CLIENT_ID"],
        client_secret=os.environ["REDDIT_CLIENT_SECRET"],
        user_agent=os.getenv("REDDIT_USER_AGENT", "p2p-reddit-face-watcher/0.1"),
    )


def parse_csv(value: str) -> List[str]:
    if not value:
        return []
    return [token.strip().lower() for token in value.split(",") if token.strip()]


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def compute_refresh_interval(entry: PostState, now: float) -> float:
    created = entry.created_utc or entry.first_seen_utc or now
    age_hours = max((now - created) / 3600.0, 0.0)
    if age_hours < 6.0:
        return 180.0  # refresh every 3 minutes
    if age_hours < 24.0:
        return 900.0  # every 15 minutes
    if age_hours < 72.0:
        return 3600.0  # every hour
    return 14400.0  # every 4 hours


def select_due_posts(state: Dict[str, PostState], now: float, max_batch: int) -> List[PostState]:
    if not state:
        return []
    due = []
    fallback_entry = None
    fallback_time = float("inf")
    for entry in state.values():
        next_refresh = entry.next_refresh_utc or 0.0
        if next_refresh <= now:
            due.append(entry)
        elif next_refresh < fallback_time:
            fallback_time = next_refresh
            fallback_entry = entry
    if not due and fallback_entry is not None:
        due = [fallback_entry]
    if max_batch and max_batch > 0 and len(due) > max_batch:
        due.sort(key=lambda e: (e.next_refresh_utc or 0.0))
        due = due[:max_batch]
    return due


def _safe_float(value: Optional[str]) -> Optional[float]:
    if value in (None, "", "None"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Optional[str]) -> Optional[int]:
    if value in (None, "", "None"):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _relative_path(base_dir: Path, candidate: Optional[str]) -> Optional[str]:
    if not candidate:
        return None
    try:
        rel = Path(candidate).resolve().relative_to(base_dir)
        return str(rel)
    except Exception:
        return candidate


def load_jsonl_metadata(path: Path) -> Dict[str, dict]:
    if not path.exists():
        return {}
    data: Dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            posting_id = obj.get("posting_id")
            if posting_id:
                data[posting_id] = obj
    return data


def domain_from_record(rec: PostRecord) -> str:
    url = rec.domain or rec.url or rec.permalink or ""
    if url.startswith("http"):
        try:
            host = urlparse(url).netloc.lower()
            if host.startswith("www."):
                host = host[4:]
            return host
        except Exception:
            return ""
    host = (rec.domain or "").lower()
    if host.startswith("www."):
        host = host[4:]
    return host


def should_keep(
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

    domain = domain_from_record(rec)
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


def load_state(path: Path) -> Tuple[Dict[str, PostState], Dict[str, AllPostInfo]]:
    if path.exists():
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, dict) and ("tracked_posts" in raw or "all_posts" in raw):
            tracked_payload = raw.get("tracked_posts", {})
            all_payload = raw.get("all_posts", {})
        else:
            tracked_payload = raw if isinstance(raw, dict) else {}
            all_payload = {}

        state: Dict[str, PostState] = {}
        for posting_id, payload in tracked_payload.items():
            faces = [
                FaceDetection(
                    media_url=item["media_url"],
                    media_local_path=item["media_local_path"],
                    face_path=item.get("face_path"),
                    num_faces=item.get("num_faces", 0),
                    issues=item.get("issues", []),
                )
                for item in payload.get("faces", [])
            ]
            state[posting_id] = PostState(
                posting_id=posting_id,
                post_id=payload["post_id"],
                subreddit=payload.get("subreddit"),
                author_name=payload.get("author_name"),
                title=payload.get("title"),
                created_utc=payload.get("created_utc"),
                first_seen_utc=payload.get("first_seen_utc", time.time()),
                last_snapshot_utc=payload.get("last_snapshot_utc"),
                metrics=payload.get("metrics", {}),
                media_items=payload.get("media_items", []),
                faces=faces,
                next_refresh_utc=payload.get("next_refresh_utc"),
            )
        all_posts: Dict[str, AllPostInfo] = {}
        for posting_id, payload in all_payload.items():
            all_posts[posting_id] = AllPostInfo(
                posting_id=posting_id,
                post_id=payload.get("post_id") or posting_id.split(":", 1)[-1],
                subreddit=payload.get("subreddit"),
                author_name=payload.get("author_name"),
                title=payload.get("title"),
                created_utc=payload.get("created_utc"),
                first_seen_utc=payload.get("first_seen_utc", time.time()),
                last_snapshot_utc=payload.get("last_snapshot_utc"),
                metrics=payload.get("metrics", {}),
                face_crops=int(payload.get("face_crops") or 0),
                has_faces=bool(payload.get("has_faces")),
                media_count=int(payload.get("media_count") or 0),
                next_refresh_utc=payload.get("next_refresh_utc"),
            )
        return state, all_posts
    return {}, {}


def serialize_state(state: Dict[str, PostState], all_posts: Dict[str, AllPostInfo]) -> Dict[str, dict]:
    payload_tracked: Dict[str, dict] = {}
    for posting_id, entry in state.items():
        payload_tracked[posting_id] = {
            "post_id": entry.post_id,
            "subreddit": entry.subreddit,
            "author_name": entry.author_name,
            "title": entry.title,
            "created_utc": entry.created_utc,
            "first_seen_utc": entry.first_seen_utc,
            "last_snapshot_utc": entry.last_snapshot_utc,
            "metrics": entry.metrics,
            "media_items": entry.media_items,
            "faces": [asdict(face) for face in entry.faces],
            "next_refresh_utc": entry.next_refresh_utc,
        }
    payload_all: Dict[str, dict] = {}
    for posting_id, info in all_posts.items():
        payload_all[posting_id] = {
            "post_id": info.post_id,
            "subreddit": info.subreddit,
            "author_name": info.author_name,
            "title": info.title,
            "created_utc": info.created_utc,
            "first_seen_utc": info.first_seen_utc,
            "last_snapshot_utc": info.last_snapshot_utc,
            "metrics": info.metrics,
            "face_crops": info.face_crops,
            "has_faces": info.has_faces,
            "media_count": info.media_count,
            "next_refresh_utc": info.next_refresh_utc,
        }
    return {"tracked_posts": payload_tracked, "all_posts": payload_all}


def write_posts_status_faces(path: Path, state: Dict[str, PostState]) -> None:
    import csv

    rows = []
    for posting_id, entry in state.items():
        metrics = entry.metrics
        rows.append({
            "posting_id": posting_id,
            "post_id": entry.post_id,
            "subreddit": entry.subreddit or "",
            "author_name": entry.author_name or "",
            "title": entry.title or "",
            "created_utc": _format_float(entry.created_utc),
            "first_seen_utc": _format_float(entry.first_seen_utc),
            "last_snapshot_utc": _format_float(entry.last_snapshot_utc),
            "score": _format_int(metrics.get("score")),
            "upvote_ratio": _format_float(metrics.get("upvote_ratio")),
            "num_comments": _format_int(metrics.get("num_comments")),
            "total_awards_received": _format_int(metrics.get("total_awards_received")),
            "view_count": _format_int(metrics.get("view_count")),
            "face_crops": str(len([face for face in entry.faces if face.face_path])),
            "media_count": str(len(entry.media_items)),
        })

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=DEFAULT_STATUS_FIELDS)
        writer.writeheader()
        for row in sorted(rows, key=lambda r: r["first_seen_utc"], reverse=True):
            writer.writerow(row)


def write_posts_status_all(path: Path, all_posts: Dict[str, AllPostInfo]) -> None:
    import csv

    rows = []
    for posting_id, info in all_posts.items():
        metrics = info.metrics
        rows.append({
            "posting_id": posting_id,
            "post_id": info.post_id,
            "subreddit": info.subreddit or "",
            "author_name": info.author_name or "",
            "title": info.title or "",
            "created_utc": _format_float(info.created_utc),
            "first_seen_utc": _format_float(info.first_seen_utc),
            "last_snapshot_utc": _format_float(info.last_snapshot_utc),
            "score": _format_int(metrics.get("score")),
            "upvote_ratio": _format_float(metrics.get("upvote_ratio")),
            "num_comments": _format_int(metrics.get("num_comments")),
            "total_awards_received": _format_int(metrics.get("total_awards_received")),
            "view_count": _format_int(metrics.get("view_count")),
            "face_crops": str(info.face_crops),
            "media_count": str(info.media_count),
            "has_faces": "1" if info.has_faces else "0",
        })

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=DEFAULT_STATUS_FIELDS_ALL)
        writer.writeheader()
        for row in sorted(rows, key=lambda r: r["first_seen_utc"], reverse=True):
            writer.writerow(row)


def write_faces_index(path: Path, state: Dict[str, PostState]) -> None:
    import csv

    rows: List[Dict[str, str]] = []
    for posting_id, entry in state.items():
        for face in entry.faces:
            rows.append({
                "posting_id": posting_id,
                "media_url": face.media_url,
                "media_local_path": face.media_local_path,
                "face_path": face.face_path or "",
                "num_faces": str(face.num_faces),
                "issues": ";".join(face.issues),
            })

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FACE_INDEX_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def download_new_media(
    out_dir: Path,
    records: List[PostRecord],
    media_dirname: str,
    retries: int,
    timeout: int,
    delay: float,
) -> Tuple[Dict[Tuple[str, str], str], Dict[str, object]]:
    if not records:
        return {}, {"attempted": 0, "succeeded": 0, "skipped": 0, "failed": []}
    local_paths, stats = download_media(
        records,
        str(out_dir),
        media_dirname,
        retries,
        timeout,
        delay,
    )
    return local_paths, stats


def extract_faces_for_post(
    base_dir: Path,
    face_dir: Path,
    record: PostRecord,
    media_local_map: Dict[Tuple[str, str], str],
    image_size: int,
) -> List[FaceDetection]:
    import cv2

    results: List[FaceDetection] = []
    posting_id = record.posting_id()
    face_dir.mkdir(parents=True, exist_ok=True)

    for idx, media in enumerate(record.media, start=1):
        key = (media.media_url, posting_id)
        rel_path = media_local_map.get(key, "")
        if not rel_path:
            results.append(FaceDetection(media.media_url, "", None, 0, ["no_local_path"]))
            continue

        src_path = (base_dir / rel_path).resolve()
        image = load_image(src_path)
        if image is None:
            results.append(FaceDetection(media.media_url, str(rel_path), None, 0, ["image_load_failed"]))
            continue

        faces = detect_faces(image)
        if not faces:
            results.append(FaceDetection(media.media_url, str(rel_path), None, 0, ["no_face"]))
            continue

        faces_sorted = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)
        best_face = faces_sorted[0]
        crop = crop_face(image, best_face, output_size=image_size)
        face_filename = f"{sanitize_token(posting_id)}_{idx:03d}.png"
        dest_path = face_dir / face_filename
        cv2.imwrite(str(dest_path), crop)
        results.append(FaceDetection(media.media_url, str(rel_path), str(dest_path.relative_to(base_dir)), len(faces), []))

    return results


def bootstrap_state_from_existing(
    out_dir: Path,
    state: Dict[str, PostState],
    all_posts: Dict[str, AllPostInfo],
    args: argparse.Namespace,
) -> None:
    if not args.bootstrap_existing:
        return

    import csv

    posts_path = (
        Path(args.bootstrap_posts_summary).resolve()
        if args.bootstrap_posts_summary
        else out_dir / "posts_summary.csv"
    )
    if not posts_path.exists():
        print(f"[bootstrap] posts_summary not found at {posts_path}, skipping bootstrap.")
        return

    media_inventory_path = (
        Path(args.bootstrap_media_inventory).resolve()
        if args.bootstrap_media_inventory
        else out_dir / "media_inventory.csv"
    )
    content_map_path = (
        Path(args.bootstrap_content_map).resolve()
        if args.bootstrap_content_map
        else out_dir / "content_id_map.csv"
    )
    face_summary_path = (
        Path(args.bootstrap_face_summary).resolve()
        if args.bootstrap_face_summary
        else out_dir / "face_detection_summary.csv"
    )
    jsonl_path = (
        Path(args.bootstrap_posts_jsonl).resolve()
        if args.bootstrap_posts_jsonl
        else out_dir / "reddit_posts.jsonl"
    )

    posts: Dict[str, dict] = {}
    with posts_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            posting_id = row.get("posting_id")
            post_id = row.get("post_id")
            if not posting_id or not post_id:
                continue
            posts[posting_id] = row

    if not posts:
        print("[bootstrap] posts_summary contained no valid rows; skipping bootstrap.")
        return

    media_by_post: Dict[str, List[dict]] = {}
    if media_inventory_path.exists():
        with media_inventory_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                posting_id = row.get("posting_id")
                if not posting_id:
                    continue
                media_by_post.setdefault(posting_id, []).append(row)
    else:
        print(f"[bootstrap] media_inventory not found at {media_inventory_path}; media info will be empty.")

    content_to_post: Dict[str, Tuple[str, str, str]] = {}
    post_to_content: Dict[str, List[str]] = {}
    if content_map_path.exists():
        with content_map_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                content_id = row.get("content_id")
                posting_id = row.get("posting_id")
                if not content_id or not posting_id:
                    continue
                local_path = row.get("local_path") or ""
                media_url = row.get("media_url") or ""
                content_to_post[content_id] = (posting_id, local_path, media_url)
                post_to_content.setdefault(posting_id, []).append(content_id)
    else:
        print(f"[bootstrap] content_id_map not found at {content_map_path}; face mapping may be incomplete.")

    faces_by_content: Dict[str, dict] = {}
    if face_summary_path.exists():
        with face_summary_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cid = row.get("content_id")
                if cid:
                    faces_by_content[cid] = row
    else:
        print(f"[bootstrap] face_detection_summary not found at {face_summary_path}; faces list will be empty.")

    jsonl_meta = load_jsonl_metadata(jsonl_path)
    now_ts = time.time()

    added_tracked = 0
    added_all = 0
    for posting_id, row in posts.items():
        post_id = row.get("post_id")
        metrics = {
            "score": _safe_int(row.get("score")),
            "upvote_ratio": _safe_float(row.get("upvote_ratio")),
            "num_comments": _safe_int(row.get("num_comments")),
            "total_awards_received": _safe_int(row.get("total_awards_received")),
            "view_count": _safe_int(row.get("view_count")),
        }
        media_entries: List[Dict[str, str]] = []
        for media_row in media_by_post.get(posting_id, []):
            media_entries.append(
                {
                    "media_url": media_row.get("media_url", ""),
                    "kind": media_row.get("kind", ""),
                    "downloadable": media_row.get("downloadable", ""),
                    "local_path": media_row.get("local_path", ""),
                }
            )

        faces: List[FaceDetection] = []
        face_count = 0
        for content_id in post_to_content.get(posting_id, []):
            _, local_path, media_url = content_to_post.get(content_id, (posting_id, "", ""))
            face_row = faces_by_content.get(content_id)
            if face_row:
                face_path = _relative_path(out_dir, face_row.get("face_path"))
                issues = [tok for tok in (face_row.get("issues") or "").split(";") if tok]
                has_face_crop = bool(face_path)
                if has_face_crop:
                    face_count += 1
                faces.append(
                    FaceDetection(
                        media_url=media_url,
                        media_local_path=local_path,
                        face_path=face_path,
                        num_faces=_safe_int(face_row.get("num_faces")) or 0,
                        issues=issues,
                    )
                )
            else:
                faces.append(
                    FaceDetection(
                        media_url=media_url,
                        media_local_path=local_path,
                        face_path=None,
                        num_faces=0,
                        issues=["no_face_summary"],
                    )
                )

        meta = jsonl_meta.get(posting_id, {})
        created = _safe_float(row.get("created_utc")) or _safe_float(meta.get("created_utc"))  # type: ignore[arg-type]
        first_seen = created or time.time()
        media_count = int(row.get("media_count") or 0)

        if posting_id not in all_posts:
            all_posts[posting_id] = AllPostInfo(
                posting_id=posting_id,
                post_id=post_id or posting_id.split(":", 1)[-1],
                subreddit=row.get("subreddit") or meta.get("subreddit"),
                author_name=row.get("author_name") or meta.get("author_name"),
                title=meta.get("title"),
                created_utc=created or meta.get("created_utc"),
                first_seen_utc=first_seen,
                last_snapshot_utc=None,
                metrics=metrics,
                face_crops=face_count,
                has_faces=face_count > 0,
                media_count=media_count,
                next_refresh_utc=None,
            )
            added_all += 1
        else:
            info = all_posts[posting_id]
            info.face_crops = face_count
            info.has_faces = face_count > 0
            info.media_count = info.media_count or media_count
            if not info.metrics:
                info.metrics = metrics
            if info.next_refresh_utc is None:
                temp_state = PostState(
                    posting_id=posting_id,
                    post_id=info.post_id,
                    subreddit=info.subreddit,
                    author_name=info.author_name,
                    title=info.title,
                    created_utc=info.created_utc,
                    first_seen_utc=info.first_seen_utc,
                    last_snapshot_utc=info.last_snapshot_utc,
                    metrics=info.metrics,
                    media_items=[],
                    faces=[],
                    next_refresh_utc=None,
                )
                info.next_refresh_utc = now_ts + compute_refresh_interval(temp_state, now_ts)

        if face_count <= 0 or posting_id in state:
            continue

        post_state = PostState(
            posting_id=posting_id,
            post_id=post_id or posting_id.split(":", 1)[-1],
            subreddit=row.get("subreddit") or meta.get("subreddit"),
            author_name=row.get("author_name") or meta.get("author_name"),
            title=meta.get("title"),
            created_utc=created or meta.get("created_utc"),
            first_seen_utc=first_seen,
            last_snapshot_utc=None,
            metrics=metrics,
            media_items=media_entries,
            faces=faces,
            next_refresh_utc=None,
        )
        post_state.next_refresh_utc = now_ts + compute_refresh_interval(post_state, now_ts)
        state[posting_id] = post_state
        ap_info = all_posts.get(posting_id)
        if ap_info:
            ap_info.next_refresh_utc = post_state.next_refresh_utc
        added_tracked += 1

    if added_tracked or added_all:
        print(f"[bootstrap] added {added_tracked} tracked posts and {added_all} all-post entries from existing outputs.")
    else:
        print("[bootstrap] no new posts loaded from existing outputs.")


def refresh_metrics(
    reddit: praw.Reddit,
    entries: List[PostState],
    all_posts: Dict[str, AllPostInfo],
    per_request_sleep: float,
    iteration: int,
    global_start: float,
    show_progress: bool,
) -> None:
    if not entries:
        return

    refresh_start = time.time()
    now = refresh_start
    total = len(entries)
    for idx, entry in enumerate(entries, start=1):
        target = tp.TargetPost(
            posting_id=entry.posting_id,
            post_id=entry.post_id,
            subreddit=entry.subreddit,
            author=entry.author_name,
            created_utc=entry.created_utc,
        )
        try:
            snap = tp.fetch_submission(reddit, target)
        except Exception as exc:
            print(f"[warn] refresh failed for {entry.posting_id}: {exc}", file=sys.stderr)
            time.sleep(max(per_request_sleep, 0.0))
            continue
        entry.metrics = {
            "score": snap.get("score"),
            "upvote_ratio": snap.get("upvote_ratio"),
            "num_comments": snap.get("num_comments"),
            "total_awards_received": snap.get("total_awards_received"),
            "view_count": snap.get("view_count"),
        }
        entry.last_snapshot_utc = snap.get("snapshot_utc")
        entry.subreddit = snap.get("subreddit") or entry.subreddit
        entry.author_name = snap.get("author") or entry.author_name
        entry.created_utc = snap.get("created_utc") or entry.created_utc
        current_time = time.time()
        entry.next_refresh_utc = current_time + compute_refresh_interval(entry, current_time)

        info = all_posts.get(entry.posting_id)
        face_crops_count = len([face for face in entry.faces if face.face_path])
        if info:
            info.metrics = dict(entry.metrics)
            info.last_snapshot_utc = entry.last_snapshot_utc
            info.subreddit = entry.subreddit
            info.author_name = entry.author_name
            info.created_utc = entry.created_utc
            info.face_crops = info.face_crops or face_crops_count
            info.has_faces = True
            info.media_count = info.media_count or len(entry.media_items)
            info.next_refresh_utc = entry.next_refresh_utc
        else:
            all_posts[entry.posting_id] = AllPostInfo(
                posting_id=entry.posting_id,
                post_id=entry.post_id,
                subreddit=entry.subreddit,
                author_name=entry.author_name,
                title=entry.title,
                created_utc=entry.created_utc,
                first_seen_utc=entry.first_seen_utc,
                last_snapshot_utc=entry.last_snapshot_utc,
                metrics=dict(entry.metrics),
                face_crops=face_crops_count,
                has_faces=True,
                media_count=len(entry.media_items),
                next_refresh_utc=entry.next_refresh_utc,
            )

        if show_progress:
            progress = (idx / total) * 100 if total else 100.0
            elapsed_refresh = format_duration(time.time() - refresh_start)
            elapsed_total = format_duration(time.time() - global_start)
            sys.stdout.write(
                f"[refresh] iter {iteration} {idx}/{total} ({progress:5.1f}%) "
                f"refresh {elapsed_refresh} total {elapsed_total}    \r"
            )
            sys.stdout.flush()
        time.sleep(max(per_request_sleep, 0.0))
    if show_progress:
        total_elapsed = format_duration(time.time() - global_start)
        sys.stdout.write(f"[refresh] iter {iteration} complete total {total_elapsed}\n")
        sys.stdout.flush()


def poll_new_posts(
    reddit: praw.Reddit,
    args: argparse.Namespace,
    seen_post_ids: Iterable[str],
) -> List[PostRecord]:
    subs = [token.strip() for token in args.subs.split(",") if token.strip()]
    allow_post_hints = parse_csv(args.allow_post_hints)
    allow_media_kinds = parse_csv(args.allow_media_kinds)
    exclude_flair_tokens = parse_csv(args.exclude_flair)
    exclude_title_tokens = parse_csv(args.exclude_title_keywords)
    exclude_domains = parse_csv(args.exclude_domains)
    include_domains = parse_csv(args.include_domains)

    seen = set(seen_post_ids)
    new_records: List[PostRecord] = []
    now_ts = time.time()
    max_age_hours = args.max_post_age_hours if args.max_post_age_hours and args.max_post_age_hours > 0 else None
    for rec in iter_posts(reddit, subs, args.query, args.days, args.min_score, args.per_sub_limit):
        if rec.post_id in seen:
            continue
        if max_age_hours is not None and rec.created_utc:
            age_hours = max((now_ts - rec.created_utc) / 3600.0, 0.0)
            if age_hours > max_age_hours:
                continue
        keep, reason = should_keep(
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
            continue
        new_records.append(rec)
        seen.add(rec.post_id)
        if args.max_new_posts and len(new_records) >= args.max_new_posts:
            break
    return new_records


def _format_float(value: Optional[float]) -> str:
    if value in (None, ""):
        return ""
    try:
        return f"{float(value):.6f}".rstrip("0").rstrip(".")
    except (TypeError, ValueError):
        return ""


def _format_int(value: Optional[float]) -> str:
    if value in (None, ""):
        return ""
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return ""


def main() -> None:
    ap = build_argparser()
    args = ap.parse_args()

    ensure_env()
    reddit = init_reddit()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    state_dir = out_dir / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    state_path = state_dir / "watch_reddit_faces.json"

    face_dir = out_dir / args.face_dirname

    state, all_posts = load_state(state_path)
    print(f"[init] loaded {len(state)} tracked posts and {len(all_posts)} total posts from state.")

    bootstrap_state_from_existing(out_dir, state, all_posts, args)

    stop_requested = False

    def handle_signal(signum, frame):  # noqa: D401
        nonlocal stop_requested
        print(f"[signal] received {signum}, will exit after current iteration.")
        stop_requested = True

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    iteration = 0
    global_start = time.time()
    while True:
        iteration += 1
        if args.max_iterations and iteration > args.max_iterations:
            print("[loop] reached max_iterations; exiting.")
            break

        start_ts = time.time()
        try:
            seen_ids = {info.post_id for info in all_posts.values() if info.post_id}
            seen_ids.update(entry.post_id for entry in state.values() if entry.post_id)
            new_records = poll_new_posts(reddit, args, seen_ids)
        except Exception as exc:
            print(f"[warn] failed to poll new posts: {exc}", file=sys.stderr)
            new_records = []

        if new_records:
            print(f"[loop] discovered {len(new_records)} new posts.")
            media_map, stats = download_new_media(
                out_dir,
                new_records,
                args.media_dirname,
                args.download_retries,
                args.download_timeout,
                args.download_delay,
            )
            if stats.get("attempted"):
                print(f"[media] downloads: {stats['succeeded']}/{stats['attempted']} succeeded, {stats['skipped']} skipped")
                if stats.get("failed"):
                    print(f"[media] {len(stats['failed'])} assets failed to download.")
            for rec in new_records:
                posting_id = rec.posting_id()
                now_ts = time.time()
                faces = extract_faces_for_post(out_dir, face_dir, rec, media_map, args.image_size)
                face_count = sum(1 for face in faces if face.face_path)
                metrics = {
                    "score": rec.score,
                    "upvote_ratio": rec.upvote_ratio,
                    "num_comments": rec.num_comments,
                    "total_awards_received": rec.total_awards_received,
                    "view_count": rec.view_count,
                }
                all_posts[posting_id] = AllPostInfo(
                    posting_id=posting_id,
                    post_id=rec.post_id,
                    subreddit=rec.subreddit,
                    author_name=rec.author_name,
                    title=rec.title,
                    created_utc=rec.created_utc,
                    first_seen_utc=now_ts,
                    last_snapshot_utc=now_ts,
                    metrics=metrics,
                    face_crops=face_count,
                    has_faces=face_count > 0,
                    media_count=len(rec.media),
                    next_refresh_utc=None,
                )
                if face_count > 0:
                    post_state = PostState(
                        posting_id=posting_id,
                        post_id=rec.post_id,
                        subreddit=rec.subreddit,
                        author_name=rec.author_name,
                        title=rec.title,
                        created_utc=rec.created_utc,
                        first_seen_utc=now_ts,
                        last_snapshot_utc=now_ts,
                        metrics=metrics,
                        media_items=[
                            {
                                "media_url": media.media_url,
                                "kind": media.kind,
                                "downloadable": "1" if media.downloadable else "0",
                                "local_path": media_map.get((media.media_url, posting_id), ""),
                            }
                            for media in rec.media
                        ],
                        faces=faces,
                        next_refresh_utc=None,
                    )
                    post_state.next_refresh_utc = now_ts + 120.0
                    state[posting_id] = post_state
                    all_posts[posting_id].next_refresh_utc = post_state.next_refresh_utc
                else:
                    print(f"[filter] skipped tracking {posting_id} (no face crops detected).")
        else:
            print("[loop] no new posts discovered.")

        due_entries = select_due_posts(state, time.time(), args.max_refresh_batch)
        if due_entries:
            refresh_metrics(
                reddit,
                due_entries,
                all_posts,
                args.per_request_sleep,
                iteration,
                global_start,
                show_progress=not args.no_progress,
            )
        else:
            print("[refresh] no posts due for refresh; skipping.")

        posts_status_faces = out_dir / "posts_status_faces.csv"
        write_posts_status_faces(posts_status_faces, state)
        # Backwards compatible alias
        write_posts_status_faces(out_dir / "posts_status.csv", state)
        posts_status_all = out_dir / "posts_status_all.csv"
        write_posts_status_all(posts_status_all, all_posts)
        faces_index_path = out_dir / "face_detections.csv"
        write_faces_index(faces_index_path, state)

        state_path.write_text(
            json.dumps(serialize_state(state, all_posts), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        total_elapsed = format_duration(time.time() - global_start)
        iteration_elapsed = format_duration(time.time() - start_ts)
        print(
            f"[state] iteration {iteration} finished in {iteration_elapsed} "
            f"(total {total_elapsed}); tracked={len(state)} all={len(all_posts)}"
        )
        print(f"[state] persisted tracked/all → {state_path}")

        if args.status_json:
            status_payload = {
                "iteration": iteration,
                "tracked_posts": len(state),
                "all_posts": len(all_posts),
                "last_run_utc": time.time(),
            }
            status_path = Path(args.status_json).resolve()
            status_path.parent.mkdir(parents=True, exist_ok=True)
            status_path.write_text(json.dumps(status_payload, indent=2), encoding="utf-8")

        if stop_requested:
            print("[loop] stop requested; exiting.")
            break

        elapsed = time.time() - start_ts
        sleep_base = max(args.interval - elapsed, 1.0)
        jitter = random.uniform(-args.jitter, args.jitter)
        sleep_time = max(1.0, sleep_base + jitter)
        print(f"[sleep] sleeping {sleep_time:.1f}s (base {sleep_base:.1f}s, jitter {jitter:.1f}s)")
        time.sleep(sleep_time)


if __name__ == "__main__":
    main()
