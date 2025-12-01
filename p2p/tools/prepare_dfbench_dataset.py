# -*- coding: utf-8 -*-
"""
Prepare DeepFakeBench-compatible dataset assets from scraped Reddit media.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Convert Reddit media into DeepFakeBench dataset format.")
    ap.add_argument("--content-csv", required=True, help="Path to content_canonical.csv or kept_content.csv.")
    ap.add_argument("--media-root", required=True, help="Root directory containing media local_path files.")
    ap.add_argument(
        "--dataset-root",
        type=str,
        default="DeepFakeBench/datasets/rgb/reddit_canonical",
        help="Output dataset root (frames will be stored under <root>/frames/<content_id>/).",
    )
    ap.add_argument(
        "--dataset-name",
        type=str,
        default="reddit_canonical",
        help="Dataset name to use inside the generated JSON.",
    )
    ap.add_argument(
        "--label-name",
        type=str,
        default="reddit_unknown",
        help="Label key to use in dataset JSON (must exist in test_config label_dict).",
    )
    ap.add_argument(
        "--max-frames",
        type=int,
        default=32,
        help="Maximum number of frames to extract per video.",
    )
    ap.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="Output size (square) for stored frames.",
    )
    ap.add_argument(
        "--dataset-json",
        type=str,
        default="DeepFakeBench/preprocessing/dataset_json/reddit_canonical.json",
        help="Path to write the dataset JSON file.",
    )
    ap.add_argument(
        "--mapping-json",
        type=str,
        default=None,
        help="Optional path to write auxiliary mapping (content_id -> frames).",
    )
    return ap


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_content_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def save_image(src_path: Path, dst_path: Path, size: int) -> bool:
    try:
        with Image.open(src_path) as img:
            img = img.convert("RGB")
            if img.width != size or img.height != size:
                img = img.resize((size, size), Image.BILINEAR)
            ensure_dir(dst_path.parent)
            img.save(dst_path, format="PNG")
        return True
    except Exception as e:
        print(f"[warn] save_image failed ({src_path}) -> {dst_path}: {e}")
        return False


def extract_video_frames(src_path: Path, dst_dir: Path, max_frames: int, size: int) -> Tuple[List[Path], List[str]]:
    ensure_dir(dst_dir)
    cap = cv2.VideoCapture(str(src_path))
    if not cap.isOpened():
        return [], [f"video_open_failed:{src_path}"]

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    issues: List[str] = []
    saved_paths: List[Path] = []

    if total_frames <= 0:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        duration = cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0
        issues.append(f"video_unknown_length:{fps}:{duration}")
        frame_indices = list(range(max_frames))
    else:
        desired = min(max_frames, total_frames)
        if desired <= 0:
            issues.append("video_no_frames")
            cap.release()
            return [], issues
        frame_indices = sorted(set(np.linspace(0, total_frames - 1, desired, dtype=int).tolist()))

    current_idx = 0
    target_pointer = 0
    target_indices = frame_indices
    for idx in range(max(frame_indices) + 1):
        ret, frame = cap.read()
        if not ret:
            issues.append(f"frame_read_failed_{idx}")
            continue
        if idx < target_indices[target_pointer]:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (size, size), interpolation=cv2.INTER_CUBIC)
        frame_path = dst_dir / f"{target_pointer:03d}.png"
        Image.fromarray(frame_resized).save(frame_path, format="PNG")
        saved_paths.append(frame_path)

        target_pointer += 1
        if target_pointer >= len(target_indices):
            break
        current_idx += 1

    cap.release()
    if not saved_paths:
        issues.append("video_no_frames_saved")
    return saved_paths, issues


def prepare_dataset(
    rows: List[Dict[str, str]],
    media_root: Path,
    dataset_root: Path,
    dataset_name: str,
    label_name: str,
    max_frames: int,
    image_size: int,
) -> Tuple[Dict, Dict[str, Dict[str, object]]]:
    frames_root = dataset_root / "frames"
    ensure_dir(frames_root)

    dataset_entries: Dict[str, Dict[str, object]] = {}
    stats: Dict[str, Dict[str, object]] = defaultdict(dict)

    for row in rows:
        content_id = row.get("content_id") or row.get("contentId")
        local_path = row.get("canonical_local_path") or row.get("local_path")
        kind = (row.get("kind") or "").lower()
        if not content_id or not local_path:
            continue

        src = media_root / local_path
        if not src.exists():
            stats[content_id]["issues"] = stats[content_id].get("issues", []) + [f"missing:{local_path}"]
            continue

        out_dir = frames_root / content_id
        if out_dir.exists():
            shutil.rmtree(out_dir)
        frame_list: List[Path] = []
        issues: List[str] = []

        ext = src.suffix.lower()
        if ext in IMAGE_EXTS:
            dst = out_dir / "000.png"
            success = save_image(src, dst, image_size)
            if success:
                frame_list = [dst]
            else:
                issues.append("image_convert_failed")
        elif ext in VIDEO_EXTS:
            frame_list, issues = extract_video_frames(src, out_dir, max_frames, image_size)
        else:
            dst = out_dir / src.name
            ensure_dir(out_dir)
            shutil.copy2(src, dst)
            frame_list = [dst]
            issues.append(f"unknown_ext:{ext}")

        if not frame_list:
            stats[content_id]["issues"] = stats[content_id].get("issues", []) + issues
            continue

        frames_abs = [str(frame.resolve()) for frame in frame_list]
        dataset_entries[content_id] = {
            "label": label_name,
            "frames": frames_abs,
        }
        stats_entry = stats[content_id]
        if issues:
            stats_entry["issues"] = stats_entry.get("issues", []) + issues
        stats_entry["frames"] = len(frame_list)
        stats_entry["source"] = str(src.resolve())
        stats_entry["frame_dir"] = str(out_dir.resolve())
        stats_entry["frame_paths"] = frames_abs

    dataset_json = {
        dataset_name: {
            label_name: {
                "test": dataset_entries,
            }
        }
    }
    return dataset_json, stats


def main():
    ap = build_argparser()
    args = ap.parse_args()

    content_rows = read_content_csv(Path(args.content_csv))
    media_root = Path(args.media_root).resolve()
    dataset_root = Path(args.dataset_root).resolve()

    ensure_dir(dataset_root)

    dataset_json, stats = prepare_dataset(
        content_rows,
        media_root,
        dataset_root,
        dataset_name=args.dataset_name,
        label_name=args.label_name,
        max_frames=args.max_frames,
        image_size=args.image_size,
    )

    dataset_json_path = Path(args.dataset_json).resolve()
    ensure_dir(dataset_json_path.parent)
    with dataset_json_path.open("w", encoding="utf-8") as f:
        json.dump(dataset_json, f, ensure_ascii=False, indent=2)
    print(f"[prepare] dataset json -> {dataset_json_path}")

    if args.mapping_json:
        mapping_path = Path(args.mapping_json).resolve()
        ensure_dir(mapping_path.parent)
        with mapping_path.open("w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"[prepare] stats mapping -> {mapping_path}")


if __name__ == "__main__":
    main()
