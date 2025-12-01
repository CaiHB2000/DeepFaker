#!/usr/bin/env python
"""
Filter Fakeddit samples to ensure图文齐全，并按指定策略（子板块/时间）严格划分。

用法示例：
python dynamic_distill/data_prep/fakeddit_strict_split.py \
  --raw-dir datasets/fakeddit/raw/multimodal_only_samples \
  --processed-dir datasets/fakeddit/processed \
  --images-dir datasets/fakeddit/images \
  --output-dir datasets/fakeddit/processed_strict/subreddit \
  --strategy subreddit \
  --train-ratio 0.7 --val-ratio 0.15 --test-ratio 0.15
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from tqdm import tqdm


REQUIRED_RAW_COLS = ["id", "subreddit", "created_utc"]


@dataclass
class Sample:
    guid: str
    text: str
    label: int
    image_path: Path
    subreddit: str
    created_utc: float


def load_raw_metadata(raw_dir: Path, cache_path: Path) -> Dict[str, Dict[str, str]]:
    if cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    metadata: Dict[str, Dict[str, str]] = {}
    for split_tag in ("train", "validate", "test_public"):
        tsv = raw_dir / f"multimodal_{split_tag}.tsv"
        if not tsv.exists():
            raise FileNotFoundError(f"Missing raw tsv: {tsv}")
        df = pd.read_table(tsv, usecols=REQUIRED_RAW_COLS)
        for _, row in df.iterrows():
            guid = str(row["id"])
            subreddit = str(row.get("subreddit", "")).strip()
            created = float(row.get("created_utc", 0.0) or 0.0)
            metadata[guid] = {"subreddit": subreddit, "created_utc": created}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle)
    return metadata


def hashed_image_path(image_dir: Path, guid: str) -> Path:
    import hashlib

    digest = hashlib.sha256(guid.encode("utf-8")).hexdigest()
    return image_dir / f"{digest}.jpg"


def collect_samples(processed_dir: Path, images_dir: Path, metadata: Dict[str, Dict[str, str]]) -> List[Sample]:
    records: List[Sample] = []
    for split in ("train", "val", "test"):
        # allow scheme subfolders, else flat
        csv_path = processed_dir / f"fakeddit_{split}.csv"
        if not csv_path.exists():
            # try under 6way/3way/2way
            for scheme in ("6way", "3way", "2way"):
                alt = processed_dir / scheme / f"fakeddit_{split}.csv"
                if alt.exists():
                    csv_path = alt
                    break
        if not csv_path.exists():
            raise FileNotFoundError(f"Processed CSV missing: {csv_path}")
        df = pd.read_csv(csv_path, usecols=["guid", "text", "label"])
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"scan {split}"):
            guid = str(row["guid"])
            meta = metadata.get(guid)
            if not meta:
                continue
            text = str(row["text"]).strip()
            if not text:
                continue
            img_path = hashed_image_path(images_dir, guid)
            if not img_path.exists():
                continue
            try:
                label = int(row["label"])
            except ValueError:
                continue
            records.append(
                Sample(
                    guid=guid,
                    text=text,
                    label=label,
                    image_path=img_path,
                    subreddit=meta.get("subreddit", ""),
                    created_utc=float(meta.get("created_utc", 0.0) or 0.0),
                )
            )
    return records


def split_by_subreddit(samples: List[Sample], ratios: Tuple[float, float, float], seed: int):
    import random

    totals = {"train": 0, "val": 0, "test": 0}
    target = {
        "train": ratios[0] * len(samples),
        "val": ratios[1] * len(samples),
        "test": ratios[2] * len(samples),
    }
    groups: Dict[str, List[Sample]] = {}
    for sample in samples:
        key = sample.subreddit or "unknown"
        groups.setdefault(key, []).append(sample)

    keys = list(groups.keys())
    random.Random(seed).shuffle(keys)
    assignment: Dict[str, str] = {}
    for key in keys:
        sizes = {split: target[split] - totals[split] for split in totals}
        split = max(sizes, key=sizes.get)
        assignment[key] = split
        totals[split] += len(groups[key])
    buckets = {"train": [], "val": [], "test": []}
    for key, split in assignment.items():
        buckets[split].extend(groups[key])
    return buckets


def split_by_time(samples: List[Sample], ratios: Tuple[float, float, float]):
    samples_sorted = sorted(samples, key=lambda s: s.created_utc)
    n = len(samples_sorted)
    train_end = int(ratios[0] * n)
    val_end = train_end + int(ratios[1] * n)
    return {
        "train": samples_sorted[:train_end],
        "val": samples_sorted[train_end:val_end],
        "test": samples_sorted[val_end:],
    }


def write_split(buckets: Dict[str, List[Sample]], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, data in buckets.items():
        df = pd.DataFrame(
            [
                {
                    "guid": s.guid,
                    "text": s.text,
                    "label": s.label,
                    "image_path": s.image_path.as_posix(),
                    "subreddit": s.subreddit,
                    "created_utc": s.created_utc,
                }
                for s in data
            ]
        )
        df.to_csv(out_dir / f"fakeddit_{split}.csv", index=False)
        print(f"[write] {split}: {len(df)} samples -> {out_dir / f'fakeddit_{split}.csv'}")


def main():
    parser = argparse.ArgumentParser(description="Generate strict Fakeddit splits with text+image.")
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--strategy", choices=["subreddit", "time"], required=True)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ratios = (args.train_ratio, args.val_ratio, args.test_ratio)
    if not abs(sum(ratios) - 1.0) < 1e-6:
        parser.error("train+val+test ratios must sum to 1.0")

    metadata_cache = args.processed_dir / "metadata_cache.json"
    metadata = load_raw_metadata(args.raw_dir, metadata_cache)
    samples = collect_samples(args.processed_dir, args.images_dir, metadata)
    print(f"[info] collected {len(samples)} samples with text+image")

    if args.strategy == "subreddit":
        buckets = split_by_subreddit(samples, ratios, args.seed)
    else:
        buckets = split_by_time(samples, ratios)

    write_split(buckets, args.output_dir)


if __name__ == "__main__":
    main()
