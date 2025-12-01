# -*- coding: utf-8 -*-
"""
Detect faces in collected media and produce face crops for DeepFakeBench.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

try:
    from retinaface import RetinaFace  # type: ignore

    _RETINAFACE_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    RetinaFace = None
    _RETINAFACE_AVAILABLE = False


@dataclass
class FaceResult:
    content_id: str
    source_path: Path
    face_path: Optional[Path]
    num_faces: int
    issues: List[str]


CASCADE_PATH = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"


def load_image(path: Path) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    img = cv2.imread(str(path))
    if img is None:
        return None
    return img


def _detect_faces_retinaface(
    image: np.ndarray,
    score_threshold: float = 0.9,
    min_size: int = 60,
) -> List[Tuple[int, int, int, int]]:
    if not _RETINAFACE_AVAILABLE:
        return []
    try:
        results = RetinaFace.detect_faces(image)
    except Exception:
        return []
    if not isinstance(results, dict):
        return []
    boxes: List[Tuple[int, int, int, int]] = []
    for face in results.values():
        if not isinstance(face, dict):
            continue
        score = face.get("score", 0.0) or 0.0
        if score < score_threshold:
            continue
        facial_area = face.get("facial_area") or ()
        if len(facial_area) != 4:
            continue
        x1, y1, x2, y2 = facial_area
        w = int(max(x2 - x1, 0))
        h = int(max(y2 - y1, 0))
        if min(w, h) < min_size:
            continue
        boxes.append((int(x1), int(y1), w, h))
    return boxes


def detect_faces(
    image: np.ndarray,
    scale_factor: float = 1.1,
    min_neighbors: int = 5,
    min_size: int = 80,
    retinaface_score: float = 0.9,
) -> List[Tuple[int, int, int, int]]:
    if _RETINAFACE_AVAILABLE:
        boxes = _detect_faces_retinaface(image, score_threshold=retinaface_score, min_size=min_size)
        if boxes:
            return boxes
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(str(CASCADE_PATH))
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=max(min_neighbors, 6),
        minSize=(min_size, min_size),
    )
    return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]


def crop_face(image: np.ndarray, bbox: Tuple[int, int, int, int], output_size: int = 256) -> np.ndarray:
    x, y, w, h = bbox
    cx = x + w // 2
    cy = y + h // 2
    half = max(w, h) // 2
    half = int(half * 1.2)  # expand a bit to include context
    h_img, w_img = image.shape[:2]
    left = max(cx - half, 0)
    right = min(cx + half, w_img)
    top = max(cy - half, 0)
    bottom = min(cy + half, h_img)
    crop = image[top:bottom, left:right]
    if crop.size == 0:
        crop = image[max(y, 0):min(y+h, h_img), max(x, 0):min(x+w, w_img)]
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(crop_rgb).resize((output_size, output_size), Image.BILINEAR)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def process_content(row: dict, media_root: Path, face_dir: Path, image_size: int) -> FaceResult:
    content_id = row.get("content_id") or row.get("contentId")
    local_path = row.get("canonical_local_path") or row.get("local_path") or row.get("source_path")
    issues: List[str] = []
    if not content_id or not local_path:
        issues.append("missing_path")
        return FaceResult(content_id or "", Path(), None, 0, issues)

    src = media_root / local_path
    image = load_image(src)
    if image is None:
        issues.append("image_load_failed")
        return FaceResult(content_id, src, None, 0, issues)

    faces = detect_faces(image)
    if not faces:
        issues.append("no_face")
        return FaceResult(content_id, src, None, 0, issues)

    # pick largest face
    faces_sorted = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)
    best_face = faces_sorted[0]
    crop = crop_face(image, best_face, output_size=image_size)

    face_dir.mkdir(parents=True, exist_ok=True)
    face_path = face_dir / f"{content_id}_000.png"
    cv2.imwrite(str(face_path), crop)

    return FaceResult(content_id, src, face_path, len(faces), issues)


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Detect faces and produce crops for DeepFakeBench")
    ap.add_argument("--work-dir", required=True, help="Base work directory (e.g., tmp/reddit_seed_realnews)")
    ap.add_argument("--kept-csv", default=None, help="Input kept_content.csv (defaults to <work-dir>/kept_content.csv)")
    ap.add_argument("--media-root", default=None, help="Media root (defaults to work dir)")
    ap.add_argument("--out-kept", default="kept_content_faces.csv", help="Filtered kept CSV output")
    ap.add_argument("--face-dir", default="face_crops", help="Relative directory to store face crops")
    ap.add_argument("--image-size", type=int, default=256, help="Output face size")
    return ap


def main() -> None:
    ap = build_argparser()
    args = ap.parse_args()
    work_dir = Path(args.work_dir).resolve()
    kept_csv = Path(args.kept_csv) if args.kept_csv else work_dir / "kept_content.csv"
    media_root = Path(args.media_root) if args.media_root else work_dir
    out_kept = work_dir / args.out_kept
    face_dir = work_dir / args.face_dir

    if not kept_csv.exists():
        raise FileNotFoundError(f"kept csv not found: {kept_csv}")

    with kept_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    fieldnames = reader.fieldnames or []
    if "canonical_local_path" not in fieldnames:
        fieldnames = list(fieldnames) + ["canonical_local_path"]
    fieldnames_faces = fieldnames + ["num_faces", "face_source"]

    face_results: List[FaceResult] = []
    for row in rows:
        result = process_content(row, media_root, face_dir, args.image_size)
        face_results.append(result)

    # write filtered kept csv
    with out_kept.open("w", encoding="utf-8", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames_faces)
        writer.writeheader()
        for row, result in zip(rows, face_results):
            if result.face_path is None:
                continue
            row_face = dict(row)
            rel_path = result.face_path.relative_to(media_root)
            row_face["canonical_local_path"] = str(rel_path)
            row_face["num_faces"] = result.num_faces
            face_source = str(result.source_path) if result.source_path.exists() else ""
            row_face["face_source"] = face_source
            writer.writerow(row_face)

    # summary csv
    summary_path = work_dir / "face_detection_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as f_sum:
        writer = csv.DictWriter(f_sum, fieldnames=["content_id", "source_path", "face_path", "num_faces", "issues"])
        writer.writeheader()
        for result in face_results:
            writer.writerow({
                "content_id": result.content_id,
                "source_path": str(result.source_path),
                "face_path": str(result.face_path) if result.face_path else "",
                "num_faces": result.num_faces,
                "issues": ";".join(result.issues),
            })

    no_face_path = work_dir / "no_face_content.csv"
    with no_face_path.open("w", encoding="utf-8", newline="") as f_nf:
        writer = csv.DictWriter(f_nf, fieldnames=["content_id", "source_path", "issues"])
        writer.writeheader()
        for result in face_results:
            if result.face_path is None:
                writer.writerow({
                    "content_id": result.content_id,
                    "source_path": str(result.source_path),
                    "issues": ";".join(result.issues),
                })

    print(f"[faces] processed {len(face_results)} contents")
    print(f"[faces] crops -> {face_dir}")
    print(f"[faces] filtered kept csv -> {out_kept}")
    print(f"[faces] summary -> {summary_path}")
    print(f"[faces] no-face list -> {no_face_path}")


if __name__ == "__main__":
    main()
