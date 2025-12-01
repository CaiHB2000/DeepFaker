# -*- coding: utf-8 -*-
"""
Apply quality rules to clustered Reddit media and emit keep/drop manifests.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from PIL import Image  # type: ignore
except Exception:  # pragma: no cover
    Image = None

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None


IMAGE_KINDS = {"image", "gif"}
VIDEO_KINDS = {"video_mp4"}
FATAL_ISSUES = {
    "missing_file",
    "size_too_small",
    "image_short_edge_lt_256",
    "image_read_failed",
    "video_read_failed",
    "video_duration_lt_2s",
    "video_resolution_lt_360p",
}


@dataclass
class MediaInfo:
    content_id: str
    local_path: str
    kind: str
    size_bytes: int
    relation: str
    source_url: str


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Evaluate media quality and produce kept/dropped content manifests.")
    ap.add_argument("--inventory", required=True, help="Path to media_inventory.csv.")
    ap.add_argument("--content-map", required=True, help="Path to content_id_map.csv.")
    ap.add_argument("--canonical", required=True, help="Path to content_canonical.csv.")
    ap.add_argument("--media-root", required=True, help="Root directory for resolving local_path.")
    ap.add_argument("--out_dir", required=True, help="Directory to write quality outputs.")
    ap.add_argument("--image-min-edge", type=int, default=256, help="Minimum allowed short edge for images.")
    ap.add_argument("--image-min-bytes", type=int, default=10_240, help="Minimum file size (bytes) for images/gifs.")
    ap.add_argument("--video-min-edge", type=int, default=360, help="Minimum allowed short edge for videos.")
    ap.add_argument("--video-min-duration", type=float, default=2.0, help="Minimum duration (seconds) for videos.")
    return ap


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_inventory(path: Path) -> Dict[str, MediaInfo]:
    info_by_local: Dict[str, MediaInfo] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            local_path = row.get("local_path", "")
            if not local_path:
                continue
            try:
                size_bytes = int(row.get("size_bytes") or 0)
            except ValueError:
                size_bytes = 0
            info_by_local[local_path] = MediaInfo(
                content_id="",
                local_path=local_path,
                kind=row.get("kind", ""),
                size_bytes=size_bytes,
                relation="",
                source_url=row.get("source_post_url", "") or row.get("media_url", ""),
            )
    return info_by_local


def load_content_map(path: Path, info_by_local: Dict[str, MediaInfo]) -> Dict[str, List[MediaInfo]]:
    content_map: Dict[str, List[MediaInfo]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            local_path = row.get("local_path", "")
            content_id = row.get("content_id", "")
            if not content_id:
                continue
            entry = info_by_local.get(local_path)
            if entry is None:
                entry = MediaInfo(
                    content_id="",
                    local_path=local_path,
                    kind=row.get("kind", ""),
                    size_bytes=int(row.get("size_bytes") or 0),
                    relation=row.get("relation", ""),
                    source_url=row.get("media_url", ""),
                )
                info_by_local[local_path] = entry
            entry.content_id = content_id
            entry.relation = row.get("relation", "")
            joined = content_map.setdefault(content_id, [])
            joined.append(entry)
    return content_map


def load_canonical(path: Path) -> Dict[str, Dict[str, str]]:
    canonical = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            canonical[row.get("content_id", "")] = row
    return canonical


def read_image_geometry(path: Path) -> Optional[Tuple[int, int]]:
    if Image is not None:
        try:
            with Image.open(path) as img:
                return img.size  # (width, height)
        except Exception:
            return None
    if cv2 is not None:
        img = cv2.imread(str(path))
        if img is None:
            return None
        h, w = img.shape[:2]
        return (int(w), int(h))
    return None


def read_video_geometry(path: Path) -> Optional[Tuple[float, float, float]]:
    if cv2 is None:
        return None
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0.0
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0.0
        duration = frame_count / fps if fps > 0 else 0.0
        return float(width), float(height), float(duration)
    except Exception:
        return None
    finally:
        cap.release()


def main():
    ap = build_argparser()
    args = ap.parse_args()

    inventory_path = Path(args.inventory).resolve()
    content_map_path = Path(args.content_map).resolve()
    canonical_path = Path(args.canonical).resolve()
    media_root = Path(args.media_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    ensure_dir(out_dir)

    info_by_local = load_inventory(inventory_path)
    content_map = load_content_map(content_map_path, info_by_local)
    canonical_meta = load_canonical(canonical_path)

    quality_rows: List[Dict[str, str]] = []
    kept_rows: List[Dict[str, str]] = []
    dropped_rows: List[Dict[str, str]] = []

    content_status: Dict[str, Dict[str, List[str]]] = {}
    content_metrics: Dict[str, Dict[str, object]] = {}

    for content_id, media_list in content_map.items():
        status = content_status.setdefault(content_id, {"fatal": [], "warn": []})
        canonical_local = (canonical_meta.get(content_id) or {}).get("canonical_local_path", "")
        for media in media_list:
            rel_path = media.local_path
            abs_path = media_root / rel_path if rel_path else None
            issues = []
            warnings = []
            if not rel_path or not abs_path or not abs_path.exists():
                issues.append(("missing_file", "local file missing"))
            else:
                if media.kind in IMAGE_KINDS:
                    if media.size_bytes and media.size_bytes < args.image_min_bytes:
                        issues.append(("size_too_small", f"{media.size_bytes}B < {args.image_min_bytes}B"))
                    geometry = read_image_geometry(abs_path)
                    if geometry is None:
                        issues.append(("image_read_failed", "cannot decode image"))
                    else:
                        width, height = geometry
                        short_edge = min(width, height)
                        if short_edge < args.image_min_edge:
                            issues.append(("image_short_edge_lt_256", f"short_edge={short_edge}"))
                        ratio = max(width, height) / max(1, short_edge)
                        if ratio > 4:
                            warnings.append(("image_aspect_extreme", f"aspect_ratio={ratio:.2f}"))
                        if media.relation == "canonical":
                            content_metrics[content_id] = {
                                "width": width,
                                "height": height,
                                "duration": "",
                                "size_bytes": media.size_bytes,
                                "kind": media.kind,
                                "local_path": rel_path,
                            }
                elif media.kind in VIDEO_KINDS:
                    geometry = read_video_geometry(abs_path)
                    if geometry is None:
                        issues.append(("video_read_failed", "cannot decode video"))
                    else:
                        width, height, duration = geometry
                        short_edge = min(width, height)
                        if short_edge < args.video_min_edge:
                            issues.append(("video_resolution_lt_360p", f"short_edge={short_edge:.0f}"))
                        if duration and duration < args.video_min_duration:
                            issues.append(("video_duration_lt_2s", f"duration={duration:.2f}s"))
                        if media.relation == "canonical":
                            content_metrics[content_id] = {
                                "width": int(width),
                                "height": int(height),
                                "duration": f"{duration:.2f}",
                                "size_bytes": media.size_bytes,
                                "kind": media.kind,
                                "local_path": rel_path,
                            }
                else:
                    if media.relation == "canonical":
                        content_metrics.setdefault(content_id, {
                            "width": "",
                            "height": "",
                            "duration": "",
                            "size_bytes": media.size_bytes,
                            "kind": media.kind,
                            "local_path": rel_path,
                        })

            severity = "fatal"
            for issue, detail in issues:
                quality_rows.append({
                    "content_id": content_id,
                    "local_path": rel_path,
                    "issue": issue,
                    "severity": "fatal",
                    "detail": detail,
                })
                status["fatal"].append(issue)
                if media.relation == "canonical" and issue in FATAL_ISSUES:
                    status["drop_canonical"] = True
            for warn, detail in warnings:
                quality_rows.append({
                    "content_id": content_id,
                    "local_path": rel_path,
                    "issue": warn,
                    "severity": "warn",
                    "detail": detail,
                })
                status["warn"].append(warn)

        if content_id not in content_metrics:
            canonical_row = canonical_meta.get(content_id, {})
            content_metrics[content_id] = {
                "width": "",
                "height": "",
                "duration": "",
                "size_bytes": canonical_row.get("size_bytes", ""),
                "kind": canonical_row.get("kind", ""),
                "local_path": canonical_row.get("canonical_local_path", ""),
            }

    for content_id, status in content_status.items():
        metrics = content_metrics.get(content_id, {})
        canonical_path = metrics.get("local_path", "")
        row = {
            "content_id": content_id,
            "canonical_local_path": canonical_path,
            "kind": metrics.get("kind", ""),
            "size_bytes": metrics.get("size_bytes", ""),
            "width": metrics.get("width", ""),
            "height": metrics.get("height", ""),
            "duration": metrics.get("duration", ""),
            "issues": ";".join(sorted(set(status.get("warn", [])))),
        }
        if status.get("drop_canonical"):
            row["reasons"] = ";".join(sorted(set(status.get("fatal", []))))
            dropped_rows.append(row)
        else:
            kept_rows.append(row)

    quality_path = out_dir / "quality_filters.csv"
    kept_path = out_dir / "kept_content.csv"
    dropped_path = out_dir / "dropped_content.csv"

    write_csv(quality_path, quality_rows, ["content_id", "local_path", "issue", "severity", "detail"])
    write_csv(kept_path, kept_rows, ["content_id", "canonical_local_path", "kind", "size_bytes", "width", "height", "duration", "issues"])
    write_csv(dropped_path, dropped_rows, ["content_id", "canonical_local_path", "kind", "size_bytes", "width", "height", "duration", "issues", "reasons"])

    print(f"[quality] quality_filters -> {quality_path} ({len(quality_rows)} issues)")
    print(f"[quality] kept_content -> {kept_path} ({len(kept_rows)} kept)")
    print(f"[quality] dropped_content -> {dropped_path} ({len(dropped_rows)} dropped)")


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


if __name__ == "__main__":
    main()
