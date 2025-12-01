# -*- coding: utf-8 -*-
"""
Cluster downloaded Reddit media into content IDs using PDQ (images) and vPDQ (videos).
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


IMAGE_KINDS = {"image", "gif"}
VIDEO_KINDS = {"video_mp4"}


@dataclass
class MediaEntry:
    idx: int
    posting_id: str
    media_url: str
    local_path: str
    kind: str
    sha256: str
    size_bytes: int
    pdq_hash: Optional[str]
    vpdq_frames: int
    c2pa_present: bool
    c2pa_valid: Optional[str]
    c2pa_vendor: Optional[str]
    post_url: Optional[str]
    extras: Dict[str, str] = field(default_factory=dict)
    vpdq_hashes: List[str] = field(default_factory=list)


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Cluster Reddit media by PDQ/vPDQ to assign content IDs.")
    ap.add_argument("--inventory", required=True, help="Path to media_inventory.csv.")
    ap.add_argument(
        "--provenance-jsonl",
        type=str,
        default=None,
        help="Optional provenance_media.jsonl to read detailed vPDQ frames.",
    )
    ap.add_argument("--out_dir", required=True, help="Directory to write clustering outputs.")
    ap.add_argument("--pdq-threshold", type=int, default=15, help="Maximum PDQ Hamming distance for clustering.")
    ap.add_argument(
        "--vpdq-strong",
        type=float,
        default=0.8,
        help="Similarity threshold (0-1) for strong video matches (exact).",
    )
    ap.add_argument(
        "--vpdq-weak",
        type=float,
        default=0.3,
        help="Similarity threshold (0-1) for weak video matches (partial relations).",
    )
    ap.add_argument(
        "--id-prefix",
        type=str,
        default="rdtC_",
        help="Prefix for generated content IDs.",
    )
    return ap


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_vpdq_hashes(jsonl_path: Optional[Path]) -> Dict[str, List[str]]:
    frames_by_local: Dict[str, List[str]] = {}
    if not jsonl_path or not jsonl_path.exists():
        return frames_by_local
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            manifest = data.get("media_manifest") or {}
            analysis = data.get("analysis") or {}
            fp = (analysis.get("fingerprints") or {})
            vpdq = fp.get("vpdq") or {}
            frames = vpdq.get("frames") or []
            local_path = manifest.get("local_path") or ""
            if local_path and frames:
                frames_by_local[local_path] = [str(fr.get("hash_hex")) for fr in frames if fr.get("hash_hex")]
    return frames_by_local


def load_inventory(path: Path, frames_by_local: Dict[str, List[str]]) -> List[MediaEntry]:
    entries: List[MediaEntry] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            size_val = 0
            try:
                size_val = int(row.get("size_bytes") or 0)
            except ValueError:
                size_val = 0
            vpdq_frames = 0
            try:
                vpdq_frames = int(row.get("vpdq_frames") or 0)
            except ValueError:
                vpdq_frames = 0
            entry = MediaEntry(
                idx=idx,
                posting_id=row.get("posting_id", ""),
                media_url=row.get("media_url", ""),
                local_path=row.get("local_path", ""),
                kind=row.get("kind", ""),
                sha256=row.get("sha256", ""),
                size_bytes=size_val,
                pdq_hash=(row.get("pdq_hash") or "").lower() or None,
                vpdq_frames=vpdq_frames,
                c2pa_present=row.get("c2pa_present") == "1",
                c2pa_valid=row.get("c2pa_valid"),
                c2pa_vendor=row.get("c2pa_vendor"),
                post_url=row.get("post_url"),
                extras={
                    "content_type": row.get("content_type", ""),
                    "author_name": row.get("author_name", ""),
                    "author_id": row.get("author_id", ""),
                    "platform": row.get("platform", ""),
                    "post_id": row.get("post_id", ""),
                },
                vpdq_hashes=frames_by_local.get(row.get("local_path", "") or "", []),
            )
            entries.append(entry)
    return entries


def pdq_hamming(hash_a: str, hash_b: str) -> int:
    if len(hash_a) != len(hash_b):
        return 256
    try:
        a = int(hash_a, 16)
        b = int(hash_b, 16)
    except ValueError:
        return 256
    xor = a ^ b
    return bin(xor).count("1")


def cluster_pdq(entries: List[MediaEntry], indices: List[int], threshold: int) -> Tuple[List[List[int]], Dict[Tuple[int, int], int]]:
    dist_map: Dict[Tuple[int, int], int] = {}
    graph: Dict[int, List[int]] = {i: [] for i in indices}
    for i_pos, idx_i in enumerate(indices):
        for idx_j in indices[i_pos + 1 :]:
            ha = entries[idx_i].pdq_hash
            hb = entries[idx_j].pdq_hash
            if not ha or not hb:
                continue
            dist = pdq_hamming(ha, hb)
            key = (min(idx_i, idx_j), max(idx_i, idx_j))
            dist_map[key] = dist
            if dist <= threshold:
                graph[idx_i].append(idx_j)
                graph[idx_j].append(idx_i)
    components: List[List[int]] = []
    visited = set()
    for idx in indices:
        if idx in visited:
            continue
        comp = []
        queue = deque([idx])
        visited.add(idx)
        while queue:
            cur = queue.popleft()
            comp.append(cur)
            for nxt in graph.get(cur, []):
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append(nxt)
        components.append(comp)
    return components, dist_map


def vpdq_similarity(hashes_a: List[str], hashes_b: List[str]) -> float:
    if not hashes_a or not hashes_b:
        return 0.0
    set_a = set(hashes_a)
    set_b = set(hashes_b)
    if not set_a or not set_b:
        return 0.0
    inter = len(set_a & set_b)
    denom = max(len(set_a), len(set_b))
    if denom == 0:
        return 0.0
    return inter / denom


def cluster_vpdq(entries: List[MediaEntry], indices: List[int], weak_thresh: float) -> Tuple[List[List[int]], Dict[Tuple[int, int], float]]:
    sim_map: Dict[Tuple[int, int], float] = {}
    graph: Dict[int, List[int]] = {i: [] for i in indices}
    for i_pos, idx_i in enumerate(indices):
        for idx_j in indices[i_pos + 1 :]:
            ha = entries[idx_i].vpdq_hashes
            hb = entries[idx_j].vpdq_hashes
            if not ha or not hb:
                continue
            sim = vpdq_similarity(ha, hb)
            key = (min(idx_i, idx_j), max(idx_i, idx_j))
            sim_map[key] = sim
            if sim >= weak_thresh:
                graph[idx_i].append(idx_j)
                graph[idx_j].append(idx_i)
    components: List[List[int]] = []
    visited = set()
    for idx in indices:
        if idx in visited:
            continue
        comp = []
        queue = deque([idx])
        visited.add(idx)
        while queue:
            cur = queue.popleft()
            comp.append(cur)
            for nxt in graph.get(cur, []):
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append(nxt)
        components.append(comp)
    return components, sim_map


def choose_canonical(entries: List[MediaEntry], component: List[int]) -> int:
    def sort_key(idx: int):
        entry = entries[idx]
        c2pa_score = 1 if entry.c2pa_present else 0
        return (
            -c2pa_score,
            -entry.size_bytes,
            -entry.vpdq_frames,
            entry.local_path or entry.media_url,
        )

    return min(component, key=sort_key)


def format_score(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{value:.4f}"


def build_content_ids(entries: List[MediaEntry], components: List[List[int]], dist_map, strong_thresh, weak_thresh, is_video: bool) -> List[Dict[str, object]]:
    results: List[Dict[str, object]] = []
    for comp in components:
        canonical_idx = choose_canonical(entries, comp)
        canonical_entry = entries[canonical_idx]
        for idx in comp:
            entry = entries[idx]
            if idx == canonical_idx:
                relation = "canonical"
                match = 1.0
            else:
                if is_video:
                    key = (min(idx, canonical_idx), max(idx, canonical_idx))
                    sim = dist_map.get(key)
                    if sim is None:
                        # fallback to best similarity with any other member
                        sim = max(
                            (dist_map.get((min(idx, other), max(idx, other))) or 0.0)
                            for other in comp
                            if other != idx
                        ) if len(comp) > 1 else 0.0
                    match = sim
                    if sim is None:
                        relation = "unknown"
                    elif sim >= strong_thresh:
                        relation = "exact"
                    elif sim >= weak_thresh:
                        relation = "partial"
                    else:
                        relation = "related"
                else:
                    key = (min(idx, canonical_idx), max(idx, canonical_idx))
                    dist = dist_map.get(key)
                    if dist is None:
                        dist = min(
                            dist_map.get((min(idx, other), max(idx, other)))
                            for other in comp
                            if other != idx
                        )
                    match = 1.0 - (dist / 256.0)
                    relation = "exact" if dist <= strong_thresh else "related"

            results.append({
                "entry_idx": idx,
                "canonical_idx": canonical_idx,
                "match_score": match,
                "relation": relation,
            })
    return results


def assign_content_ids(entries: List[MediaEntry], pdq_threshold: int, vpdq_strong: float, vpdq_weak: float, id_prefix: str):
    assigned: Dict[int, str] = {}
    clusters: Dict[str, List[int]] = {}
    cluster_meta: Dict[str, Dict[str, object]] = {}
    content_counter = 0

    pdq_indices = [e.idx for e in entries if e.idx not in assigned and e.pdq_hash and e.kind in IMAGE_KINDS]
    pdq_components, pdq_dist_map = cluster_pdq(entries, pdq_indices, pdq_threshold)
    pdq_info = build_content_ids(entries, pdq_components, pdq_dist_map, pdq_threshold, pdq_threshold, is_video=False)

    for comp, info_rows in zip(pdq_components, chunk_list(pdq_info, [len(c) for c in pdq_components])):
        content_counter += 1
        content_id = f"{id_prefix}{content_counter:06d}"
        canonical_idx = next(row["canonical_idx"] for row in info_rows if entries[row["entry_idx"]].idx == row["canonical_idx"])
        for row in info_rows:
            idx = row["entry_idx"]
            assigned[idx] = content_id
        clusters[content_id] = comp
        cluster_meta[content_id] = {"canonical_idx": canonical_idx, "info_rows": info_rows, "is_video": False}

    video_indices = [
        e.idx for e in entries
        if e.idx not in assigned and e.kind in VIDEO_KINDS and e.vpdq_hashes
    ]
    video_components, vpdq_sim_map = cluster_vpdq(entries, video_indices, vpdq_weak)
    video_info = build_content_ids(entries, video_components, vpdq_sim_map, vpdq_strong, vpdq_weak, is_video=True)

    for comp, info_rows in zip(video_components, chunk_list(video_info, [len(c) for c in video_components])):
        content_counter += 1
        content_id = f"{id_prefix}{content_counter:06d}"
        canonical_idx = next(row["canonical_idx"] for row in info_rows if row["entry_idx"] == row["canonical_idx"])
        for row in info_rows:
            idx = row["entry_idx"]
            assigned[idx] = content_id
        clusters[content_id] = comp
        cluster_meta[content_id] = {"canonical_idx": canonical_idx, "info_rows": info_rows, "is_video": True}

    for entry in entries:
        if entry.idx in assigned:
            continue
        content_counter += 1
        content_id = f"{id_prefix}{content_counter:06d}"
        assigned[entry.idx] = content_id
        clusters[content_id] = [entry.idx]
        cluster_meta[content_id] = {
            "canonical_idx": entry.idx,
            "info_rows": [{
                "entry_idx": entry.idx,
                "canonical_idx": entry.idx,
                "match_score": 1.0,
                "relation": "canonical",
            }],
            "is_video": entry.kind in VIDEO_KINDS,
        }

    return assigned, clusters, cluster_meta


def chunk_list(info_rows: List[Dict[str, object]], sizes: Iterable[int]) -> List[List[Dict[str, object]]]:
    result = []
    idx = 0
    for size in sizes:
        result.append(info_rows[idx : idx + size])
        idx += size
    return result


def write_content_id_map(entries: List[MediaEntry], assigned: Dict[int, str], cluster_meta: Dict[str, Dict[str, object]], out_path: Path):
    fieldnames = [
        "content_id",
        "posting_id",
        "local_path",
        "media_url",
        "kind",
        "sha256",
        "pdq_hash",
        "match_score",
        "relation",
        "source_post_url",
        "size_bytes",
        "vpdq_frames",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for content_id, meta in cluster_meta.items():
            for row in meta["info_rows"]:
                entry = entries[row["entry_idx"]]
                writer.writerow({
                    "content_id": content_id,
                    "posting_id": entry.posting_id,
                    "local_path": entry.local_path,
                    "media_url": entry.media_url,
                    "kind": entry.kind,
                    "sha256": entry.sha256,
                    "pdq_hash": entry.pdq_hash or "",
                    "match_score": format_score(row.get("match_score")),
                    "relation": row.get("relation", ""),
                    "source_post_url": entry.post_url or "",
                    "size_bytes": entry.size_bytes,
                    "vpdq_frames": entry.vpdq_frames,
                })


def write_content_canonical(entries: List[MediaEntry], cluster_meta: Dict[str, Dict[str, object]], out_path: Path):
    fieldnames = [
        "content_id",
        "canonical_local_path",
        "kind",
        "media_url",
        "sha256",
        "size_bytes",
        "pdq_hash",
        "vpdq_frames",
        "c2pa_present",
        "c2pa_valid",
        "c2pa_vendor",
        "width",
        "height",
        "duration",
        "has_audio",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for content_id, meta in cluster_meta.items():
            canonical_idx = meta["canonical_idx"]
            entry = entries[canonical_idx]
            writer.writerow({
                "content_id": content_id,
                "canonical_local_path": entry.local_path,
                "kind": entry.kind,
                "media_url": entry.media_url,
                "sha256": entry.sha256,
                "size_bytes": entry.size_bytes,
                "pdq_hash": entry.pdq_hash or "",
                "vpdq_frames": entry.vpdq_frames,
                "c2pa_present": "1" if entry.c2pa_present else "0",
                "c2pa_valid": entry.c2pa_valid or "",
                "c2pa_vendor": entry.c2pa_vendor or "",
                "width": "",
                "height": "",
                "duration": "",
                "has_audio": "",
            })


def main():
    ap = build_argparser()
    args = ap.parse_args()

    inventory_path = Path(args.inventory).resolve()
    out_dir = Path(args.out_dir).resolve()
    ensure_dir(out_dir)

    provenance_jsonl = Path(args.provenance_jsonl).resolve() if args.provenance_jsonl else None
    frames_by_local = load_vpdq_hashes(provenance_jsonl)
    entries = load_inventory(inventory_path, frames_by_local)

    assigned, clusters, cluster_meta = assign_content_ids(
        entries,
        pdq_threshold=args.pdq_threshold,
        vpdq_strong=args.vpdq_strong,
        vpdq_weak=args.vpdq_weak,
        id_prefix=args.id_prefix,
    )

    map_path = out_dir / "content_id_map.csv"
    canonical_path = out_dir / "content_canonical.csv"
    write_content_id_map(entries, assigned, cluster_meta, map_path)
    write_content_canonical(entries, cluster_meta, canonical_path)

    print(f"[dedupe] content_id_map -> {map_path} ({sum(len(v['info_rows']) for v in cluster_meta.values())} rows)")
    print(f"[dedupe] content_canonical -> {canonical_path} ({len(cluster_meta)} content IDs)")


if __name__ == "__main__":
    main()
