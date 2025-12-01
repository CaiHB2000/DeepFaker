# -*- coding: utf-8 -*-
"""
Aggregate Reddit seed outputs into inventory & quality reports.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse


def _parse_float(value) -> Optional[float]:
    if value in (None, "", "None"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_float(value: Optional[float], precision: int = 4) -> str:
    if value is None:
        return ""
    formatted = f"{value:.{precision}f}"
    if "." in formatted:
        formatted = formatted.rstrip("0").rstrip(".")
    return formatted


def _format_int(value: Optional[float]) -> str:
    if value is None:
        return ""
    try:
        return str(int(round(float(value))))
    except (TypeError, ValueError):
        return ""


def _bool_to_flag(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y"}:
            return "1"
        if lowered in {"0", "false", "no", "n"}:
            return "0"
        return ""
    return "1" if bool(value) else "0"


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Align posts, manifest, and provenance into inventory tables.")
    ap.add_argument("--posts_jsonl", required=True, help="Path to reddit_posts.jsonl emitted from run_reddit_seed.")
    ap.add_argument("--manifest", required=True, help="Path to media_manifest.csv.")
    ap.add_argument("--prov", required=True, help="Path to provenance_media.csv.")
    ap.add_argument("--out_dir", required=True, help="Output directory for generated tables.")
    ap.add_argument(
        "--media-root",
        type=str,
        default=None,
        help="Optional root directory to resolve relative local_path entries (defaults to manifest parent).",
    )
    ap.add_argument(
        "--post-table",
        type=str,
        default=None,
        help="Optional path to post_table.csv for fallback metadata.",
    )
    ap.add_argument(
        "--post-map",
        type=str,
        default=None,
        help="Optional path to post_map.csv (not required today, reserved for future use).",
    )
    ap.add_argument(
        "--content-map",
        type=str,
        default=None,
        help="Optional path to content_id_map.csv for joining content IDs.",
    )
    return ap


def read_posts(path: Path) -> Dict[str, Dict]:
    posts: Dict[str, Dict] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            pid = data.get("posting_id")
            if pid:
                posts[pid] = data
    return posts


def extract_subreddit(url: str) -> str:
    try:
        parsed = urlparse(url)
        parts = parsed.path.split("/")
        if len(parts) >= 3 and parts[1] == "r":
            return parts[2]
    except Exception:
        pass
    return ""


def read_post_table(path: Path) -> Dict[str, Dict]:
    posts: Dict[str, Dict] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            posting_id = row.get("posting_id")
            if not posting_id:
                continue
            url = row.get("url", "")
            posts[posting_id] = {
                "posting_id": posting_id,
                "platform": row.get("platform", ""),
                "post_id": row.get("post_id", ""),
                "url": url,
                "subreddit": extract_subreddit(url),
            }
    return posts


def read_manifest(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def read_provenance(path: Path) -> Dict[str, Dict[str, str]]:
    prov_by_key: Dict[str, Dict[str, str]] = {}
    prov_by_local: Dict[str, Dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            posting_id = row.get("posting_id", "")
            media_url = row.get("media_url", "")
            key = f"{posting_id}||{media_url}"
            prov_by_key[key] = row
            local_path = row.get("local_path", "")
            if local_path:
                prov_by_local[local_path] = row
    return {"by_key": prov_by_key, "by_local": prov_by_local}


def compute_sha256(path: Path) -> Optional[str]:
    h = hashlib.sha256()
    try:
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except FileNotFoundError:
        return None
    except Exception:
        return None


def resolve_local_path(local_path: str, media_root: Path) -> Optional[Path]:
    if not local_path:
        return None
    lp = Path(local_path)
    if lp.is_absolute():
        return lp if lp.exists() else None
    candidate = media_root / lp
    return candidate if candidate.exists() else None


def main():
    ap = build_argparser()
    args = ap.parse_args()

    posts_path = Path(args.posts_jsonl).resolve()
    manifest_path = Path(args.manifest).resolve()
    prov_path = Path(args.prov).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    media_root = Path(args.media_root).resolve() if args.media_root else manifest_path.parent

    post_table_path = Path(args.post_table).resolve() if args.post_table else manifest_path.with_name("post_table.csv")
    post_table_meta: Dict[str, Dict] = {}
    if post_table_path.exists():
        post_table_meta = read_post_table(post_table_path)
        print(f"[info] post_table metadata loaded ({len(post_table_meta)} rows)")

    if posts_path.exists():
        posts = read_posts(posts_path)
        if post_table_meta:
            for pid, meta in post_table_meta.items():
                base = posts.setdefault(pid, {})
                for key, value in meta.items():
                    if key not in base or base.get(key) in ("", None):
                        base[key] = value
    else:
        print(f"[warn] posts_jsonl not found at {posts_path}, falling back to post_table metadata")
        posts = post_table_meta or {}
    manifest_rows = read_manifest(manifest_path)
    prov_maps = read_provenance(prov_path)

    prov_by_key = prov_maps["by_key"]
    prov_by_local = prov_maps["by_local"]

    content_map_path = Path(args.content_map).resolve() if args.content_map else manifest_path.with_name("content_id_map.csv")
    content_by_local: Dict[str, str] = {}
    content_by_post: Dict[str, set] = defaultdict(set)
    if content_map_path.exists():
        with content_map_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cid = row.get("content_id", "")
                local = row.get("local_path", "")
                posting = row.get("posting_id", "")
                if cid:
                    if local:
                        content_by_local[local] = cid
                    if posting:
                        content_by_post.setdefault(posting, set()).add(cid)

    sha_cache: Dict[Path, Optional[str]] = {}
    stat_cache: Dict[Path, Optional[int]] = {}

    inventory_rows: List[Dict[str, str]] = []
    qc_rows: List[Dict[str, str]] = []

    def _init_post_stats():
        return {
            "media_count": 0,
            "has_video": False,
            "has_image": False,
            "has_gif": False,
            "c2pa_present": False,
            "content_ids": set(),
            "downloadable_count": 0,
            "reddit_hosted_count": 0,
            "source_domains": set(),
            "size_samples": [],
            "resolution_samples": [],
            "max_width": 0.0,
            "max_height": 0.0,
        }

    post_stats = defaultdict(_init_post_stats)

    for row in manifest_rows:
        posting_id = row.get("posting_id", "")
        media_url = row.get("media_url", "")
        local_path = row.get("local_path", "") or ""
        kind = row.get("kind", "")
        content_type = row.get("content_type", "")
        source = row.get("source", "")
        media_domain = row.get("media_domain", "")
        if not media_domain and media_url:
            try:
                media_domain = urlparse(media_url).netloc
            except Exception:
                media_domain = ""
        width_raw = row.get("width", "")
        height_raw = row.get("height", "")
        width_val = _parse_float(width_raw)
        height_val = _parse_float(height_raw)
        raw_downloadable = row.get("downloadable", "")
        if raw_downloadable in ("", None):
            downloadable_flag = kind not in ("video_dash", "video_hls") and bool(media_url)
        else:
            downloadable_flag = str(raw_downloadable).strip().lower() in {"1", "true", "yes"}
        post_is_gallery = row.get("post_is_gallery", "")
        post_is_video = row.get("post_is_video", "")
        post_over_18 = row.get("post_over_18", "")
        post_hint = row.get("post_hint", "")
        stats = post_stats[posting_id]

        resolved = resolve_local_path(local_path, media_root)
        stat_key: Optional[Path] = resolved

        size_bytes = ""
        sha256_hex = ""
        exists = resolved is not None
        errors: List[str] = []

        if local_path == "":
            errors.append("missing_local_path")
        if resolved:
            if stat_key not in stat_cache:
                try:
                    stat_cache[stat_key] = stat_key.stat().st_size
                except Exception:
                    stat_cache[stat_key] = None
            size_val = stat_cache[stat_key]
            if size_val is not None:
                size_bytes = str(size_val)
                if size_val == 0:
                    errors.append("size_zero")
                stats["size_samples"].append(size_val)
            else:
                errors.append("stat_failed")
            if stat_key not in sha_cache:
                sha_cache[stat_key] = compute_sha256(stat_key)
            sha_val = sha_cache[stat_key]
            if sha_val:
                sha256_hex = sha_val
            else:
                errors.append("sha256_failed")
        else:
            if local_path:
                errors.append("file_missing")

        prov_row = prov_by_local.get(local_path) or prov_by_key.get(f"{posting_id}||{media_url}") or {}
        posts_row = posts.get(posting_id, {})
        if posts_row.get("subreddit", "") == "" and posts_row.get("url"):
            posts_row["subreddit"] = extract_subreddit(posts_row["url"])

        c2pa_present = prov_row.get("c2pa_present", "")
        c2pa_valid = prov_row.get("c2pa_valid", "")
        pdq_hash = prov_row.get("pdq_hash", "")
        vpdq_frames = prov_row.get("vpdq_frames", "")
        fingerprint_errors = prov_row.get("fingerprint_errors", "")
        pdq_error = prov_row.get("pdq_error", "")
        vpdq_error = prov_row.get("vpdq_error", "")
        c2pa_error = prov_row.get("c2pa_error", "")

        def extend_errors(raw: str, skip: Optional[Iterable[str]] = None):
            if not raw:
                return
            skip_set = set(skip or [])
            for token in raw.split(";"):
                token = token.strip()
                if not token:
                    continue
                if token in skip_set:
                    continue
                errors.append(token)

        extend_errors(fingerprint_errors)
        extend_errors(pdq_error)
        extend_errors(vpdq_error)
        extend_errors(c2pa_error, skip={"no_claim"})

        errors = sorted(set(errors))

        content_id_val = content_by_local.get(local_path, "")

        inventory_rows.append({
            "posting_id": posting_id,
            "media_url": media_url,
            "local_path": local_path,
            "size_bytes": size_bytes,
            "sha256": sha256_hex,
            "kind": kind,
            "content_type": content_type,
            "source": source,
            "media_domain": media_domain,
            "width": "" if width_raw in (None, "", "None") else str(width_raw),
            "height": "" if height_raw in (None, "", "None") else str(height_raw),
            "downloadable": "1" if downloadable_flag else "0",
            "post_is_gallery": _bool_to_flag(post_is_gallery),
            "post_is_video": _bool_to_flag(post_is_video),
            "post_over_18": _bool_to_flag(post_over_18),
            "post_hint": post_hint,
            "subreddit": posts_row.get("subreddit", ""),
            "author_name": posts_row.get("author_name", ""),
            "author_id": posts_row.get("author_id", ""),
            "platform": posts_row.get("platform", ""),
            "post_id": posts_row.get("post_id", ""),
            "post_url": posts_row.get("url", ""),
            "created_utc": str(posts_row.get("created_utc", "")),
            "score": str(posts_row.get("score", "")),
            "upvote_ratio": str(posts_row.get("upvote_ratio", "")),
            "c2pa_present": c2pa_present,
            "c2pa_valid": c2pa_valid,
            "pdq_hash": pdq_hash,
            "vpdq_frames": vpdq_frames,
            "errors": ";".join(errors),
            "content_id": content_id_val,
        })

        if errors:
            qc_rows.append({
                "posting_id": posting_id,
                "media_url": media_url,
                "local_path": local_path,
                "errors": ";".join(errors),
            })

        stats["media_count"] += 1
        if kind.startswith("video"):
            stats["has_video"] = True
        if kind == "image":
            stats["has_image"] = True
        if kind == "gif":
            stats["has_gif"] = True
        if c2pa_present == "1":
            stats["c2pa_present"] = True
        if content_id_val:
            stats["content_ids"].add(content_id_val)
        if downloadable_flag:
            stats["downloadable_count"] += 1
        domain_key = ""
        if media_domain:
            domain_key = media_domain.lower()
        elif source:
            domain_key = source.lower()
        if domain_key:
            stats["source_domains"].add(domain_key)
            if domain_key.endswith("redd.it") or domain_key.endswith("reddit.com"):
                stats["reddit_hosted_count"] += 1
        if width_val and height_val:
            stats["resolution_samples"].append(width_val * height_val)
            stats["max_width"] = max(stats["max_width"], width_val)
            stats["max_height"] = max(stats["max_height"], height_val)

    inventory_path = out_dir / "media_inventory.csv"
    qc_path = out_dir / "qc_missing.csv"
    posts_summary_path = out_dir / "posts_summary.csv"

    write_csv(inventory_path, inventory_rows, [
        "posting_id",
        "media_url",
        "local_path",
        "size_bytes",
        "sha256",
        "kind",
        "content_type",
        "source",
        "media_domain",
        "width",
        "height",
        "downloadable",
        "post_is_gallery",
        "post_is_video",
        "post_over_18",
        "post_hint",
        "subreddit",
        "author_name",
        "author_id",
        "platform",
        "post_id",
        "post_url",
        "created_utc",
        "score",
        "upvote_ratio",
        "c2pa_present",
        "c2pa_valid",
        "pdq_hash",
        "vpdq_frames",
        "errors",
        "content_id",
    ])

    write_csv(qc_path, qc_rows, ["posting_id", "media_url", "local_path", "errors"])

    posts_summary_rows: List[Dict[str, str]] = []
    for posting_id, stats in post_stats.items():
        posts_row = posts.get(posting_id, {})
        content_ids_joined = "|".join(sorted(stats["content_ids"])) if stats["content_ids"] else ""
        media_count = stats["media_count"]
        downloadable_ratio = (stats["downloadable_count"] / media_count) if media_count else None
        reddit_hosted_ratio = (stats["reddit_hosted_count"] / media_count) if media_count else None
        size_samples = stats["size_samples"]
        resolution_samples = stats["resolution_samples"]
        median_size = statistics.median(size_samples) if size_samples else None
        max_size = max(size_samples) if size_samples else None
        median_resolution = statistics.median(resolution_samples) if resolution_samples else None
        max_resolution = max(resolution_samples) if resolution_samples else None
        max_width = stats["max_width"] if stats["max_width"] > 0 else None
        max_height = stats["max_height"] if stats["max_height"] > 0 else None
        media_domains_joined = "|".join(sorted(d for d in stats["source_domains"] if d))
        posts_summary_rows.append({
            "posting_id": posting_id,
            "subreddit": posts_row.get("subreddit", ""),
            "author_name": posts_row.get("author_name", ""),
            "author_id": posts_row.get("author_id", ""),
            "platform": posts_row.get("platform", ""),
            "post_id": posts_row.get("post_id", ""),
            "post_url": posts_row.get("url", ""),
            "created_utc": str(posts_row.get("created_utc", "")),
            "media_count": str(stats["media_count"]),
            "has_video": "1" if stats["has_video"] else "0",
            "has_image": "1" if stats["has_image"] else "0",
            "has_gif": "1" if stats["has_gif"] else "0",
            "c2pa_present": "1" if stats["c2pa_present"] else "0",
            "score": str(posts_row.get("score", "")),
            "upvote_ratio": str(posts_row.get("upvote_ratio", "")),
            "num_comments": str(posts_row.get("num_comments", "")),
            "content_ids": content_ids_joined,
            "over_18": _bool_to_flag(posts_row.get("over_18")),
            "spoiler": _bool_to_flag(posts_row.get("spoiler")),
            "is_original_content": _bool_to_flag(posts_row.get("is_original_content")),
            "is_self": _bool_to_flag(posts_row.get("is_self")),
            "is_gallery": _bool_to_flag(posts_row.get("is_gallery")),
            "author_premium": _bool_to_flag(posts_row.get("author_premium")),
            "domain": posts_row.get("domain", ""),
            "media_domains": media_domains_joined,
            "downloadable_ratio": _format_float(downloadable_ratio),
            "reddit_hosted_ratio": _format_float(reddit_hosted_ratio),
            "median_size_bytes": _format_int(median_size),
            "max_size_bytes": _format_int(max_size),
            "median_resolution_px": _format_int(median_resolution),
            "max_resolution_px": _format_int(max_resolution),
            "max_width": _format_int(max_width),
            "max_height": _format_int(max_height),
        })

    write_csv(posts_summary_path, posts_summary_rows, [
        "posting_id",
        "subreddit",
        "author_name",
        "author_id",
        "platform",
        "post_id",
        "post_url",
        "created_utc",
        "media_count",
        "has_video",
        "has_image",
        "has_gif",
        "c2pa_present",
        "score",
        "upvote_ratio",
        "num_comments",
        "content_ids",
        "over_18",
        "spoiler",
        "is_original_content",
        "is_self",
        "is_gallery",
        "author_premium",
        "domain",
        "media_domains",
        "downloadable_ratio",
        "reddit_hosted_ratio",
        "median_size_bytes",
        "max_size_bytes",
        "median_resolution_px",
        "max_resolution_px",
        "max_width",
        "max_height",
    ])

    print(f"[aggregate] media_inventory -> {inventory_path} ({len(inventory_rows)} rows)")
    print(f"[aggregate] qc_missing -> {qc_path} ({len(qc_rows)} rows)")
    print(f"[aggregate] posts_summary -> {posts_summary_path} ({len(posts_summary_rows)} rows)")


def write_csv(path: Path, rows: Iterable[Dict[str, str]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


if __name__ == "__main__":
    main()
